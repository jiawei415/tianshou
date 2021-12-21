"""Env package."""

from tianshou.env.maenv import MultiAgentEnv
from tianshou.env.venvs import (
    BaseVectorEnv,
    DummyVectorEnv,
    RayVectorEnv,
    ShmemVectorEnv,
    SubprocVectorEnv,
)

__all__ = [
    "BaseVectorEnv",
    "DummyVectorEnv",
    "SubprocVectorEnv",
    "ShmemVectorEnv",
    "RayVectorEnv",
    "MultiAgentEnv",
]


from gym.envs.registration import register

register(
    id='DeepSea-v0',
    entry_point='tianshou.env.deepsea:DeepSea',
)

register(
    id='CustomizeMDP-v1',
    entry_point='tianshou.env.customizemdp:CustomizeMDPV1',
)

register(
    id='CustomizeMDP-v2',
    entry_point='tianshou.env.customizemdp:CustomizeMDPV2',
)
