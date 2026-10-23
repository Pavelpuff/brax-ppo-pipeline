"""Entry point for parallel PPO training with W&B tracking."""

import argparse
import time

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

import wandb
from src.data.env import make_env
from src.models.networks import PPOAgent
from src.utils.config import PPOConfig
from src.utils.core import compute_gae, init_running_norm, normalize, ppo_loss, update_running_norm
from src.utils.types import RolloutInfo, RunnerState, Transition


class PPOTrainer:
    def __init__(self, config: PPOConfig, env, agent: PPOAgent):
        self.config = config
        self.env = env
        self.agent = agent

    def init_runner(self, rng: jax.Array) -> RunnerState:
        rng, init_rng, reset_rng = jax.random.split(rng, 3)
        env_state = self.env.reset(reset_rng)
        obs = env_state.obs
        obs_dim = obs.shape[-1]

        params = self.agent.init(init_rng, jnp.zeros((1, obs_dim)))

        def learning_rate_schedule(step):
            frac = 1.0 - step / (
                self.config.num_updates * self.config.update_epochs * self.config.num_minibatches
            )
            return self.config.lr * frac

        tx = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(
                learning_rate_schedule if self.config.anneal_lr else self.config.lr, eps=1e-5
            ),
        )

        train_state = TrainState.create(apply_fn=self.agent.apply, params=params, tx=tx)

        return RunnerState(
            train_state=train_state,
            env_state=env_state,
            obs=obs,
            rng=rng,
            obs_norm=init_running_norm((obs_dim,)),
            rew_norm=init_running_norm(()),
            running_returns=jnp.zeros((self.config.num_envs,), dtype=jnp.float32),
            episode_returns=jnp.zeros((self.config.num_envs,), dtype=jnp.float32),
            episode_lengths=jnp.zeros((self.config.num_envs,), dtype=jnp.int32),
        )

    def make_update_fn(self):
        cfg = self.config
        env = self.env
        agent = self.agent

        def env_step(carry, _):
            runner_state = carry

            if cfg.normalize_obs:
                obs_norm = update_running_norm(runner_state.obs_norm, runner_state.obs)
                obs_in = normalize(runner_state.obs, obs_norm, eps=cfg.norm_eps, clip=cfg.obs_clip)
            else:
                obs_norm = runner_state.obs_norm
                obs_in = runner_state.obs

            rng, act_rng = jax.random.split(runner_state.rng)
            action, _, log_prob = agent.act(runner_state.train_state.params, obs_in, act_rng)
            value = agent.apply(runner_state.train_state.params, obs_in, method=agent.critic)

            next_env_state = env.step(runner_state.env_state, action)
            next_obs = next_env_state.obs
            reward = next_env_state.reward
            done = next_env_state.done.astype(jnp.float32)

            return_for_norm = runner_state.running_returns * cfg.gamma + reward
            if cfg.normalize_reward:
                rew_norm = update_running_norm(runner_state.rew_norm, reward)
                reward_out = reward / jnp.sqrt(jnp.maximum(rew_norm.var, cfg.norm_eps))
                reward_out = jnp.clip(reward_out, -cfg.reward_clip, cfg.reward_clip)
            else:
                rew_norm = runner_state.rew_norm
                reward_out = reward

            next_running_returns = return_for_norm * (1.0 - done)
            new_episode_returns = runner_state.episode_returns + reward
            new_episode_lengths = runner_state.episode_lengths + 1
            completed_return = jnp.where(done > 0.0, new_episode_returns, 0.0)
            completed_length = jnp.where(done > 0.0, new_episode_lengths, 0)

            episode_returns = new_episode_returns * (1.0 - done)
            episode_lengths = (new_episode_lengths * (1.0 - done)).astype(jnp.int32)

            transition = Transition(
                obs=obs_in,
                action=action,
                reward=reward_out,
                done=done,
                value=value,
                log_prob=log_prob,
            )
            rollout_info = RolloutInfo(
                completed_return=completed_return, completed_length=completed_length, completed=done
            )

            next_carry = RunnerState(
                train_state=runner_state.train_state,
                env_state=next_env_state,
                obs=next_obs,
                rng=rng,
                obs_norm=obs_norm,
                rew_norm=rew_norm,
                running_returns=next_running_returns,
                episode_returns=episode_returns,
                episode_lengths=episode_lengths,
            )
            return next_carry, (transition, rollout_info)

        def update_minibatch(train_state, minibatch):
            batch, adv_batch, target_batch = minibatch

            def loss_on_params(params):
                return ppo_loss(params, agent, batch, adv_batch, target_batch, cfg)

            grad_fn = jax.value_and_grad(loss_on_params, has_aux=True)
            (loss, aux), grads = grad_fn(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, (loss, aux)

        def update_epoch(carry, _):
            train_state, flat_traj, flat_adv, flat_targets, rng = carry
            rng, perm_rng = jax.random.split(rng)
            permutation = jax.random.permutation(perm_rng, cfg.batch_size)

            shuffled_traj = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), flat_traj
            )
            shuffled_adv = jnp.take(flat_adv, permutation, axis=0)
            shuffled_targets = jnp.take(flat_targets, permutation, axis=0)

            minibatches = (
                jax.tree_util.tree_map(
                    lambda x: x.reshape((cfg.num_minibatches, cfg.minibatch_size) + x.shape[1:]),
                    shuffled_traj,
                ),
                shuffled_adv.reshape((cfg.num_minibatches, cfg.minibatch_size)),
                shuffled_targets.reshape((cfg.num_minibatches, cfg.minibatch_size)),
            )
            train_state, loss_info = jax.lax.scan(update_minibatch, train_state, minibatches)
            return (train_state, flat_traj, flat_adv, flat_targets, rng), loss_info

        def update_step(runner_state):
            state = runner_state
            state, (traj, rollout_info) = jax.lax.scan(
                env_step, state, None, length=cfg.rollout_length
            )

            last_obs_in = (
                normalize(state.obs, state.obs_norm, eps=cfg.norm_eps, clip=cfg.obs_clip)
                if cfg.normalize_obs
                else state.obs
            )
            last_value = agent.apply(state.train_state.params, last_obs_in, method=agent.critic)

            advantages, targets = compute_gae(traj, last_value, cfg.gamma, cfg.gae_lambda)
            flat_traj = jax.tree_util.tree_map(
                lambda x: x.reshape((cfg.batch_size,) + x.shape[2:]), traj
            )
            flat_adv = advantages.reshape((cfg.batch_size,))
            flat_targets = targets.reshape((cfg.batch_size,))

            (train_state, _, _, _, rng), loss_info = jax.lax.scan(
                update_epoch,
                (state.train_state, flat_traj, flat_adv, flat_targets, state.rng),
                None,
                length=cfg.update_epochs,
            )

            num_completed = jnp.maximum(jnp.sum(rollout_info.completed), 1.0)
            mean_return = jnp.sum(rollout_info.completed_return) / num_completed
            losses, aux = loss_info
            actor_losses, value_losses, entropies = aux

            metrics = {
                "mean_episode_return": mean_return,
                "loss": jnp.mean(losses),
                "actor_loss": jnp.mean(actor_losses),
                "value_loss": jnp.mean(value_losses),
                "entropy": jnp.mean(entropies),
            }

            next_runner_state = RunnerState(
                train_state=train_state,
                env_state=state.env_state,
                obs=state.obs,
                rng=rng,
                obs_norm=state.obs_norm,
                rew_norm=state.rew_norm,
                running_returns=state.running_returns,
                episode_returns=state.episode_returns,
                episode_lengths=state.episode_lengths,
            )
            return next_runner_state, metrics

        return jax.jit(update_step)


class PPOEvaluator:
    def __init__(self, config: PPOConfig, eval_env, agent: PPOAgent):
        self.config = config
        self.eval_env = eval_env
        self.agent = agent

    def make_eval_fn(self):
        cfg = self.config
        eval_env = self.eval_env
        agent = self.agent

        def evaluate(train_state, obs_norm, rng: jax.Array):
            rng, reset_rng = jax.random.split(rng)
            env_state = eval_env.reset(reset_rng)
            obs = env_state.obs
            episode_returns = jnp.zeros((obs.shape[0],), dtype=jnp.float32)
            total_completed_return = jnp.array(0.0, dtype=jnp.float32)
            total_completed_count = jnp.array(0.0, dtype=jnp.float32)

            def eval_step(carry, _):
                (
                    env_state,
                    obs,
                    episode_returns,
                    total_completed_return,
                    total_completed_count,
                    rng,
                ) = carry

                obs_in = (
                    normalize(obs, obs_norm, eps=cfg.norm_eps, clip=cfg.obs_clip)
                    if cfg.normalize_obs
                    else obs
                )
                rng, act_rng = jax.random.split(rng)

                # Используем детерминированное действие для валидации
                _, action, _ = agent.act(train_state.params, obs_in, act_rng)

                next_env_state = eval_env.step(env_state, action)
                reward = next_env_state.reward
                done = next_env_state.done.astype(jnp.float32)
                next_obs = next_env_state.obs

                episode_returns = episode_returns + reward
                completed_return = jnp.where(done > 0.0, episode_returns, 0.0)
                total_completed_return = total_completed_return + jnp.sum(completed_return)
                total_completed_count = total_completed_count + jnp.sum(done)

                episode_returns = episode_returns * (1.0 - done)

                next_carry = (
                    next_env_state,
                    next_obs,
                    episode_returns,
                    total_completed_return,
                    total_completed_count,
                    rng,
                )
                return next_carry, None

            final_carry, _ = jax.lax.scan(
                eval_step,
                (
                    env_state,
                    obs,
                    episode_returns,
                    total_completed_return,
                    total_completed_count,
                    rng,
                ),
                None,
                length=cfg.eval_horizon,
            )
            _, _, _, total_completed_return, total_completed_count, _ = final_carry

            mean_return = total_completed_return / jnp.maximum(total_completed_count, 1.0)
            return {"eval_return": mean_return, "eval_episodes": total_completed_count}

        return jax.jit(evaluate)


def main(args: argparse.Namespace) -> None:
    config = PPOConfig(
        env_name=args.env_name,
        ent_coef=args.ent_coef,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
    )

    wandb.init(project="brax-ppo-industrial", config=vars(config))
    print(f"[*] Default JAX Backend: {jax.default_backend().upper()}")

    env = make_env(config)
    eval_env = make_env(config, num_envs=config.eval_num_envs)
    agent = PPOAgent(hidden_sizes=config.hidden_sizes, action_dim=env.action_size)

    trainer = PPOTrainer(config, env, agent)
    evaluator = PPOEvaluator(config, eval_env, agent)

    update_fn = trainer.make_update_fn()
    eval_fn = evaluator.make_eval_fn()

    rng = jax.random.PRNGKey(config.seed)
    rng, train_rng, eval_rng = jax.random.split(rng, 3)
    runner_state = trainer.init_runner(train_rng)

    print(
        f"[*] Starting training loop. Env: {config.env_name}, "
        f"Total Timesteps: {config.total_timesteps}"
    )
    eval_every_updates = max(1, config.eval_every_env_steps // config.batch_size)
    start_time = time.time()

    for update_idx in range(config.num_updates):
        runner_state, metrics = update_fn(runner_state)
        log_metrics = {k: float(jax.device_get(v)) for k, v in metrics.items()}
        log_metrics["env_steps"] = (update_idx + 1) * config.batch_size

        # Evaluation phase
        do_eval = (update_idx % eval_every_updates == 0) or ((update_idx + 1) == config.num_updates)
        if do_eval:
            eval_rng, eval_step_rng = jax.random.split(eval_rng)
            eval_metrics = eval_fn(runner_state.train_state, runner_state.obs_norm, eval_step_rng)
            log_metrics["eval_return"] = float(jax.device_get(eval_metrics["eval_return"]))

        wandb.log(log_metrics)

        if update_idx % 50 == 0:
            eval_str = (
                f" | Eval Return: {log_metrics.get('eval_return', 'N/A'):.2f}" if do_eval else ""
            )
            print(
                f"Update {update_idx:04d}/{config.num_updates} | "
                f"Train Return: {log_metrics['mean_episode_return']:.2f}{eval_str}"
            )

    elapsed = time.time() - start_time
    print(f"[*] Training finished in {elapsed:.2f} seconds.")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO in Brax Environment")
    parser.add_argument("--env_name", type=str, default="ant", help="Brax environment name")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--total_timesteps", type=int, default=10_000_000, help="Total environment steps"
    )
    parser.add_argument(
        "--num_envs", type=int, default=1024, help="Number of parallel environments"
    )

    args = parser.parse_args()
    main(args)
