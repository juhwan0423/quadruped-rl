"""RSL-RL PPO configuration for the Go2 pitch task."""

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
)


@configclass
class Go2PitchPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # separate log directory so runs do not mix with the stock Go2 flat policy
        self.experiment_name = "go2_pitch"
        # the extra command dimension makes the task harder than plain flat walking
        self.max_iterations = 1000
        self.save_interval = 50
