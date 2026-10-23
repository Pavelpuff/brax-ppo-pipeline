"""Flax-based Actor and Critic networks for Continuous PPO."""

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class CriticNet(nn.Module):
    hidden_sizes: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        hidden_init = orthogonal(np.sqrt(2))
        critic_head_init = orthogonal(1.0)
        x = obs
        for hidden_size in self.hidden_sizes:
            x = nn.Dense(hidden_size, kernel_init=hidden_init, bias_init=constant(0.0))(x)
            x = nn.tanh(x)
        value = nn.Dense(1, kernel_init=critic_head_init, bias_init=constant(0.0))(x)
        return jnp.squeeze(value, axis=-1)


class ActorNet(nn.Module):
    hidden_sizes: tuple[int, ...] = (256, 256)
    action_dim: int = 1

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        hidden_init = orthogonal(np.sqrt(2))
        actor_head_init = orthogonal(0.01)
        x = obs
        for hidden_size in self.hidden_sizes:
            x = nn.Dense(hidden_size, kernel_init=hidden_init, bias_init=constant(0.0))(x)
            x = nn.tanh(x)
        mean = nn.Dense(self.action_dim, kernel_init=actor_head_init, bias_init=constant(0.0))(x)
        log_std = nn.Dense(self.action_dim, kernel_init=actor_head_init, bias_init=constant(0.0))(x)
        return mean, log_std


class PPOAgent(nn.Module):
    hidden_sizes: tuple[int, ...] = (256, 256)
    action_dim: int = 1

    def setup(self) -> None:
        self.actor_net = ActorNet(hidden_sizes=self.hidden_sizes, action_dim=self.action_dim)
        self.critic_net = CriticNet(hidden_sizes=self.hidden_sizes)

    def critic(self, obs: jnp.ndarray) -> jnp.ndarray:
        return self.critic_net(obs)

    def actor(self, obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.actor_net(obs)

    def _apply_squashing_function(
        self, mean: jnp.ndarray, action: jnp.ndarray, logp_action: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        tanh_logp_action = logp_action - jnp.sum(
            2 * (jnp.log(2) - action - nn.softplus(-2 * action)), axis=-1
        )
        return jnp.tanh(action), jnp.tanh(mean), tanh_logp_action

    def act(
        self, params, obs: jnp.ndarray, rng: jax.Array
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        mean, log_std = self.apply(params, obs, method=self.actor)
        std = jnp.exp(log_std)
        dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
        action = dist.sample(seed=rng)
        logp_action = dist.log_prob(action)
        return self._apply_squashing_function(mean, action, logp_action)

    def get_action_log_prob_and_entropy(
        self, params, obs: jnp.ndarray, squashed_action: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        mean, log_std = self.apply(params, obs, method=self.actor)
        std = jnp.exp(log_std)
        dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
        action = jnp.arctanh(jnp.clip(squashed_action, -1.0 + 1e-6, 1.0 - 1e-6))
        log_prob = dist.log_prob(action)
        _, _, squashed_log_prob = self._apply_squashing_function(mean, action, log_prob)
        return squashed_log_prob, dist.entropy()

    def __call__(self, obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        mean, log_std = self.actor(obs)
        value = self.critic(obs)
        return mean, log_std, value
