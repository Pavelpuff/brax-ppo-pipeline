"""Brax environment factory with vectorization and auto-reset."""
from brax import envs
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper, VmapWrapper
from src.utils.config import PPOConfig

def make_env(config: PPOConfig, num_envs: int | None = None) -> object:
    """Creates a vectorized, auto-resetting Brax environment."""
    env = envs.get_environment(config.env_name, backend=config.backend)
    env = EpisodeWrapper(env, episode_length=config.episode_length, action_repeat=1)
    env = VmapWrapper(env, batch_size=config.num_envs if num_envs is None else num_envs)
    env = AutoResetWrapper(env)
    return env