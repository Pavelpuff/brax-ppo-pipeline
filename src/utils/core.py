"""Core mathematical operations: GAE, PPO Loss, and Running Norms."""

import jax
import jax.numpy as jnp

from src.utils.config import PPOConfig
from src.utils.types import RunningNorm, Transition


def init_running_norm(shape: tuple) -> RunningNorm:
    return RunningNorm(
        mean=jnp.zeros(shape, dtype=jnp.float32),
        var=jnp.ones(shape, dtype=jnp.float32),
        count=jnp.array(1e-4, dtype=jnp.float32),
    )


def update_running_norm(norm: RunningNorm, x: jnp.ndarray) -> RunningNorm:
    batch_count = x.shape[0]
    total_count = norm.count + batch_count
    mean_batch = jnp.mean(x, axis=0)
    var_batch = jnp.var(x, axis=0)
    mean_diff = mean_batch - norm.mean
    new_mean = norm.mean + mean_diff * (batch_count / total_count)
    sum_sq_old = norm.var * norm.count
    sum_sq_new = var_batch * batch_count
    sum_sq_combined = (
        sum_sq_old + sum_sq_new + (mean_diff**2) * (norm.count * batch_count / total_count)
    )
    new_var = sum_sq_combined / total_count
    return RunningNorm(mean=new_mean, var=new_var, count=total_count)


def normalize(
    x: jnp.ndarray, norm: RunningNorm, eps: float = 1e-6, clip: float | None = None
) -> jnp.ndarray:
    y = (x - norm.mean) / jnp.sqrt(jnp.maximum(norm.var, eps))
    if clip is not None:
        y = jnp.clip(y, -clip, clip)
    return y


def compute_gae(
    traj: Transition, last_value: jnp.ndarray, gamma: float, gae_lambda: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    def scan_gae(carry, t):
        gae, next_value = carry
        not_done = 1.0 - traj.done[t]
        delta = traj.reward[t] + gamma * next_value * not_done - traj.value[t]
        gae = delta + gamma * gae_lambda * not_done * gae
        return (gae, traj.value[t]), gae

    T = traj.reward.shape[0]
    time_idx = jnp.arange(T - 1, -1, -1)
    _, advantages_rev = jax.lax.scan(scan_gae, (jnp.zeros_like(last_value), last_value), time_idx)
    advantages = jnp.flip(advantages_rev, axis=0)
    targets = advantages + traj.value
    return advantages, targets


def ppo_loss(
    params, agent, batch: Transition, advantages: jnp.ndarray, targets: jnp.ndarray, cfg: PPOConfig
) -> tuple[jnp.ndarray, tuple]:
    log_prob, entropy = agent.get_action_log_prob_and_entropy(params, batch.obs, batch.action)
    value = agent.apply(params, batch.obs, method=agent.critic)

    if cfg.normalize_advantage:
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    ratio = jnp.exp(log_prob - batch.log_prob)
    unclipped_obj = ratio * advantages
    clipped_obj = jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages
    actor_loss = -jnp.mean(jnp.minimum(unclipped_obj, clipped_obj))
    value_loss = jnp.mean(jnp.square(value - targets))
    entropy_loss = jnp.mean(entropy)

    total_loss = actor_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss
    return total_loss, (actor_loss, value_loss, entropy_loss)
