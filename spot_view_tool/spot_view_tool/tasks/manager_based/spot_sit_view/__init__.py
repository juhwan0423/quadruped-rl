"""Gym registrations for Spot sit-view tasks."""

import gymnasium as gym

from . import agents


_COMMON_KWARGS = {
    "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
    "disable_env_checker": True,
    "kwargs": {
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpotSitViewEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotSitViewPPORunnerCfg",
    },
}

_POSTURE_VIEW_KWARGS = {
    "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
    "disable_env_checker": True,
    "kwargs": {
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpotPostureViewEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotPostureViewPPORunnerCfg",
    },
}

_POSTURE_FAST_KWARGS = {
    "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
    "disable_env_checker": True,
    "kwargs": {
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpotPostureFastEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotPostureFastPPORunnerCfg",
    },
}


gym.register(id="Isaac-Spot-Sit-View-V0", **_COMMON_KWARGS)
gym.register(
    id="Isaac-Spot-Sit-View-Play-V0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpotSitViewEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotSitViewPPORunnerCfg",
    },
)

gym.register(id="Isaac-Spot-Posture-View-V0", **_POSTURE_VIEW_KWARGS)
gym.register(
    id="Isaac-Spot-Posture-View-Play-V0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpotPostureViewEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotPostureViewPPORunnerCfg",
    },
)

gym.register(id="Isaac-Spot-Posture-Fast-V0", **_POSTURE_FAST_KWARGS)
gym.register(
    id="Isaac-Spot-Posture-Fast-Play-V0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpotPostureFastEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotPostureFastPPORunnerCfg",
    },
)

# Isaac Lab convention uses lower-case v0. Keep aliases so either spelling works.
gym.register(id="Isaac-Spot-Sit-View-v0", **_COMMON_KWARGS)
gym.register(
    id="Isaac-Spot-Sit-View-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpotSitViewEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotSitViewPPORunnerCfg",
    },
)

gym.register(id="Isaac-Spot-Posture-View-v0", **_POSTURE_VIEW_KWARGS)
gym.register(
    id="Isaac-Spot-Posture-View-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpotPostureViewEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotPostureViewPPORunnerCfg",
    },
)

gym.register(id="Isaac-Spot-Posture-Fast-v0", **_POSTURE_FAST_KWARGS)
gym.register(
    id="Isaac-Spot-Posture-Fast-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpotPostureFastEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotPostureFastPPORunnerCfg",
    },
)
