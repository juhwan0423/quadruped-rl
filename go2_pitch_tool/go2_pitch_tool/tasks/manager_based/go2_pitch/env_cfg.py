"""Go2 flat locomotion with an extra body-pitch command.

The environment inherits everything from ``Isaac-Velocity-Flat-Unitree-Go2-v0`` and only
changes what the pitch command needs:

* commands: adds a 1-dim ``body_pitch`` term next to ``base_velocity``
* observations: adds the pitch command to the policy group (48 -> 49 dims)
* rewards: pitch tracking replaces the flat-orientation penalty, roll stays penalized
* curriculum: the pitch range widens as training progresses
"""

from __future__ import annotations

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg, CurriculumCfg, RewardsCfg

import go2_pitch_tool.tasks.manager_based.go2_pitch.mdp as mdp


@configclass
class Go2PitchCommandsCfg(CommandsCfg):
    """Velocity command from the base task plus an independent body-pitch command."""

    body_pitch = mdp.BodyPitchCommandCfg(
        asset_name="robot",
        # resampled faster than the velocity command (10 s) so the policy actually has to
        # learn pitch transitions rather than a few near-static poses per episode
        resampling_time_range=(3.0, 5.0),
        rel_level_envs=0.2,
        ranges=mdp.BodyPitchCommandCfg.Ranges(pitch=(-0.15, 0.15)),
    )


@configclass
class Go2PitchObservationsCfg:
    """Policy observations, written out in full so the layout is explicit for deployment."""

    @configclass
    class PolicyCfg(ObsGroup):
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))  # 3
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))  # 3
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))  # 3
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})  # 3
        pitch_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "body_pitch"})  # 1
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))  # 12
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))  # 12
        actions = ObsTerm(func=mdp.last_action)  # 12

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class Go2PitchRewardsCfg(RewardsCfg):
    """Base locomotion rewards plus pitch tracking and a roll-only upright penalty."""

    track_body_pitch_exp = RewTerm(
        func=mdp.track_body_pitch_exp,
        weight=1.0,
        params={"command_name": "body_pitch", "std": 0.2},
    )
    body_roll_l2 = RewTerm(func=mdp.body_roll_l2, weight=-2.5)


@configclass
class Go2PitchCurriculumCfg(CurriculumCfg):
    """Terrain curriculum from the base task (disabled on flat) plus the pitch range ramp."""

    pitch_levels = CurrTerm(
        func=mdp.pitch_command_levels,
        params={"command_name": "body_pitch", "start_range": 0.15, "end_range": 0.4, "num_steps": 4000},
    )


@configclass
class Go2PitchEnvCfg(UnitreeGo2FlatEnvCfg):
    commands: Go2PitchCommandsCfg = Go2PitchCommandsCfg()
    observations: Go2PitchObservationsCfg = Go2PitchObservationsCfg()
    rewards: Go2PitchRewardsCfg = Go2PitchRewardsCfg()
    curriculum: Go2PitchCurriculumCfg = Go2PitchCurriculumCfg()

    def __post_init__(self):
        # runs the Go2 rough + flat post-init (robot asset, action scale, event and reward
        # tuning, flat terrain, no height scanner)
        super().__post_init__()

        # the flat task pins the body level with a weight of -2.5, which directly fights the
        # pitch command -- roll is penalized separately by body_roll_l2 instead
        self.rewards.flat_orientation_l2.weight = 0.0

        # pitching the body changes the base height a lot; keep the joint limit penalty on so
        # the policy does not reach the commanded pitch by bottoming out the knees
        self.rewards.dof_pos_limits.weight = -0.1

        # leaning while turning at full speed is a lot to ask at once; the yaw rate is
        # narrowed slightly to keep the early training signal clean
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)


@configclass
class Go2PitchEnvCfg_PLAY(Go2PitchEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # play at the full pitch range instead of wherever the curriculum happens to start
        self.curriculum.pitch_levels = None
        self.commands.body_pitch.ranges.pitch = (-0.4, 0.4)
