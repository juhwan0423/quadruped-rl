"""Gym registrations for the Go2 pitch task."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Velocity-Flat-Go2-Pitch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2PitchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PitchPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Go2-Pitch-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2PitchEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PitchPPORunnerCfg",
    },
)
