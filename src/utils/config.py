"""Hyperparameters and configuration for PPO."""

from dataclasses import dataclass


@dataclass
class PPOConfig:
    env_name: str = "ant"
    backend: str = "spring"
    total_timesteps: int = 10_000_000
    num_envs: int = 1024
    rollout_length: int = 32
    update_epochs: int = 16
    num_minibatches: int = 16
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01  # Увеличено для лучшей эксплорации
    max_grad_norm: float = 0.5
    episode_length: int = 1000
    hidden_sizes: tuple[int, ...] = (256, 256)
    anneal_lr: bool = True
    normalize_advantage: bool = True
    normalize_obs: bool = True
    normalize_reward: bool = True
    norm_eps: float = 1e-6
    obs_clip: float = 10.0
    reward_clip: float = 10.0
    max_log_ratio: float = 20.0
    eval_every_env_steps: int = 500_000
    eval_num_envs: int = 1024
    eval_horizon: int = 1000
    seed: int = 42

    def __post_init__(self) -> None:
        self.batch_size = self.num_envs * self.rollout_length
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.num_updates = self.total_timesteps // self.batch_size
