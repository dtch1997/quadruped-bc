from typing import Any

from gymnasium.envs.registration import (
    register,
)

register(
    id="MyAnt-v4",
    entry_point="quadruped_bc.envs.ant:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
