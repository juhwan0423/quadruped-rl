from __future__ import annotations

import copy

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import spot_view_tool.tasks.manager_based.spot_sit_view.mdp as mdp
from spot_view_tool.tasks.manager_based.spot_sit_view.mdp.events import SIT_JOINT_TARGETS
from isaaclab_assets.robots.spot import SPOT_CFG


# The 0.34 m viewer pose is visually low, but it can start with foot
# interpenetration once reset noise and random policy actions are applied.
SIT_BASE_HEIGHT = 0.43
STAND_BASE_HEIGHT = 0.52


@configclass
class SpotSitViewSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.CuboidCfg(
            size=(120.0, 120.0, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.025)),
        collision_group=-1,
    )

    robot = copy.deepcopy(SPOT_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.8, 0.8, 0.8)),
    )


@configclass
class SpotSitViewActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        offset=SIT_JOINT_TARGETS,
        use_default_offset=False,
    )


@configclass
class SpotPostureViewActionsCfg:
    joint_pos = mdp.PostureJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        command_name="posture_view",
        scale=0.25,
    )


@configclass
class SpotSitViewObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.02, n_max=0.02)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        body_angle_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "body_angles"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class SpotPostureViewObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.02, n_max=0.02)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        posture_view_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "posture_view"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class SpotSitViewCommandsCfg:
    body_angles = mdp.BodyAngleCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.0, 4.0),
        debug_vis=False,
        ranges=mdp.BodyAngleCommandCfg.Ranges(
            roll=(-0.15, 0.15),
            pitch=(-0.18, 0.18),
            yaw=(-0.35, 0.35),
        ),
    )


@configclass
class SpotPostureViewCommandsCfg:
    posture_view = mdp.PostureViewCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.5, 5.0),
        debug_vis=False,
        ranges=mdp.PostureViewCommandCfg.Ranges(
            posture=(0.0, 1.0),
            roll=(0.0, 0.0),
            pitch=(-0.45, 0.15),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class SpotPostureFastCommandsCfg:
    posture_view = mdp.PostureViewCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.6, 1.2),
        debug_vis=False,
        ranges=mdp.PostureViewCommandCfg.Ranges(
            posture=(0.0, 1.0),
            roll=(0.0, 0.0),
            pitch=(-0.45, 0.15),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class SpotSitViewEventsCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.0),
            "dynamic_friction_range": (0.7, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 32,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_to_sit_height,
        mode="reset",
        params={
            "base_height": SIT_BASE_HEIGHT,
            "roll_range": (-0.03, 0.03),
            "pitch_range": (-0.03, 0.03),
            "yaw_range": (-0.25, 0.25),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_to_sit_pose,
        mode="reset",
        params={
            "position_noise": 0.03,
            "velocity_noise": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class SpotPostureViewEventsCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.0),
            "dynamic_friction_range": (0.7, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 32,
        },
    )
    reset_robot = EventTerm(
        func=mdp.reset_robot_to_random_posture,
        mode="reset",
        params={
            "posture_range": (0.0, 1.0),
            "sit_base_height": SIT_BASE_HEIGHT,
            "stand_base_height": STAND_BASE_HEIGHT,
            "roll_range": (-0.04, 0.04),
            "pitch_range": (-0.04, 0.04),
            "yaw_range": (-0.35, 0.35),
            "position_noise": 0.04,
            "velocity_noise": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class SpotSitViewRewardsCfg:
    body_angle_tracking = RewTerm(
        func=mdp.body_angle_command_exp,
        weight=3.0,
        params={
            "command_name": "body_angles",
            "std": 0.35,
            "roll_weight": 1.0,
            "pitch_weight": 1.0,
            "yaw_weight": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    sit_joint_pose = RewTerm(func=mdp.sit_joint_pose_exp, weight=5.0, params={"std": 0.25})
    base_height = RewTerm(func=mdp.base_height_exp, weight=2.0, params={"target_height": SIT_BASE_HEIGHT, "std": 0.08})
    upright = RewTerm(func=mdp.upright_exp, weight=1.0, params={"std": 0.35})
    feet_contact = RewTerm(
        func=mdp.feet_contact_reward,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 1.0},
    )
    feet_below_ground = RewTerm(
        func=mdp.feet_below_ground_penalty,
        weight=-3.0,
        params={
            "min_height": -0.05,
            "scale": 0.03,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    feet_contact_force = RewTerm(
        func=mdp.feet_contact_force_penalty,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "max_force": 180.0,
            "scale": 120.0,
        },
    )
    base_stability = RewTerm(func=mdp.base_stability_l2, weight=-0.3)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001, params={"asset_cfg": SceneEntityCfg("robot")})
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class SpotPostureViewRewardsCfg:
    posture_angle_tracking = RewTerm(
        func=mdp.posture_view_angle_command_exp,
        weight=5.0,
        params={
            "command_name": "posture_view",
            "std": 0.25,
            "roll_weight": 0.0,
            "pitch_weight": 4.0,
            "yaw_weight": 0.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    posture_joint_pose = RewTerm(
        func=mdp.posture_joint_pose_exp,
        weight=3.0,
        params={"command_name": "posture_view", "std": 0.35, "asset_cfg": SceneEntityCfg("robot")},
    )
    posture_base_height = RewTerm(
        func=mdp.posture_base_height_exp,
        weight=2.0,
        params={
            "command_name": "posture_view",
            "sit_height": SIT_BASE_HEIGHT,
            "stand_height": STAND_BASE_HEIGHT,
            "std": 0.09,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    upright = RewTerm(func=mdp.upright_exp, weight=0.2, params={"std": 0.45})
    feet_contact = RewTerm(
        func=mdp.feet_contact_reward,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 1.0},
    )
    feet_below_ground = RewTerm(
        func=mdp.feet_below_ground_penalty,
        weight=-4.0,
        params={
            "min_height": -0.03,
            "scale": 0.03,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    feet_contact_force = RewTerm(
        func=mdp.feet_contact_force_penalty,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "max_force": 180.0,
            "scale": 120.0,
        },
    )
    base_stability = RewTerm(func=mdp.base_stability_l2, weight=-0.3)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001, params={"asset_cfg": SceneEntityCfg("robot")})
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class SpotPostureFastRewardsCfg:
    posture_angle_tracking = RewTerm(
        func=mdp.posture_view_angle_command_exp,
        weight=5.0,
        params={
            "command_name": "posture_view",
            "std": 0.25,
            "roll_weight": 0.0,
            "pitch_weight": 4.0,
            "yaw_weight": 0.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    posture_joint_pose = RewTerm(
        func=mdp.posture_joint_pose_exp,
        weight=4.0,
        params={"command_name": "posture_view", "std": 0.32, "asset_cfg": SceneEntityCfg("robot")},
    )
    posture_base_height = RewTerm(
        func=mdp.posture_base_height_exp,
        weight=2.5,
        params={
            "command_name": "posture_view",
            "sit_height": SIT_BASE_HEIGHT,
            "stand_height": STAND_BASE_HEIGHT,
            "std": 0.09,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    upright = RewTerm(func=mdp.upright_exp, weight=0.15, params={"std": 0.45})
    feet_contact = RewTerm(
        func=mdp.feet_contact_reward,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 1.0},
    )
    feet_below_ground = RewTerm(
        func=mdp.feet_below_ground_penalty,
        weight=-4.0,
        params={
            "min_height": -0.03,
            "scale": 0.03,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    feet_contact_force = RewTerm(
        func=mdp.feet_contact_force_penalty,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "max_force": 220.0,
            "scale": 140.0,
        },
    )
    base_stability = RewTerm(func=mdp.base_stability_l2, weight=-0.2)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0005, params={"asset_cfg": SceneEntityCfg("robot")})
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-0.0001, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class SpotSitViewTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="body"), "threshold": 5.0},
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 1.0},
    )
    feet_below_ground = DoneTerm(
        func=mdp.feet_below_ground,
        params={"min_height": -0.14, "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
    )


@configclass
class SpotSitViewEnvCfg(ManagerBasedRLEnvCfg):
    scene: SpotSitViewSceneCfg = SpotSitViewSceneCfg(num_envs=512, env_spacing=2.5)
    observations: SpotSitViewObservationsCfg = SpotSitViewObservationsCfg()
    actions: SpotSitViewActionsCfg = SpotSitViewActionsCfg()
    commands: SpotSitViewCommandsCfg = SpotSitViewCommandsCfg()
    rewards: SpotSitViewRewardsCfg = SpotSitViewRewardsCfg()
    terminations: SpotSitViewTerminationsCfg = SpotSitViewTerminationsCfg()
    events: SpotSitViewEventsCfg = SpotSitViewEventsCfg()
    curriculum = None
    viewer = ViewerCfg(eye=(3.0, 3.0, 1.4), origin_type="world")

    def __post_init__(self):
        self.decimation = 10
        self.episode_length_s = 8.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        self.scene.contact_forces.update_period = self.sim.dt


class SpotSitViewEnvCfg_PLAY(SpotSitViewEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False


@configclass
class SpotPostureViewEnvCfg(ManagerBasedRLEnvCfg):
    scene: SpotSitViewSceneCfg = SpotSitViewSceneCfg(num_envs=512, env_spacing=2.5)
    observations: SpotPostureViewObservationsCfg = SpotPostureViewObservationsCfg()
    actions: SpotPostureViewActionsCfg = SpotPostureViewActionsCfg()
    commands: SpotPostureViewCommandsCfg = SpotPostureViewCommandsCfg()
    rewards: SpotPostureViewRewardsCfg = SpotPostureViewRewardsCfg()
    terminations: SpotSitViewTerminationsCfg = SpotSitViewTerminationsCfg()
    events: SpotPostureViewEventsCfg = SpotPostureViewEventsCfg()
    curriculum = None
    viewer = ViewerCfg(eye=(3.0, 3.0, 1.4), origin_type="world")

    def __post_init__(self):
        self.decimation = 10
        self.episode_length_s = 10.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        self.scene.contact_forces.update_period = self.sim.dt


class SpotPostureViewEnvCfg_PLAY(SpotPostureViewEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False


@configclass
class SpotPostureFastEnvCfg(ManagerBasedRLEnvCfg):
    scene: SpotSitViewSceneCfg = SpotSitViewSceneCfg(num_envs=512, env_spacing=2.5)
    observations: SpotPostureViewObservationsCfg = SpotPostureViewObservationsCfg()
    actions: SpotPostureViewActionsCfg = SpotPostureViewActionsCfg()
    commands: SpotPostureFastCommandsCfg = SpotPostureFastCommandsCfg()
    rewards: SpotPostureFastRewardsCfg = SpotPostureFastRewardsCfg()
    terminations: SpotSitViewTerminationsCfg = SpotSitViewTerminationsCfg()
    events: SpotPostureViewEventsCfg = SpotPostureViewEventsCfg()
    curriculum = None
    viewer = ViewerCfg(eye=(3.0, 3.0, 1.4), origin_type="world")

    def __post_init__(self):
        self.decimation = 10
        self.episode_length_s = 6.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        self.scene.contact_forces.update_period = self.sim.dt


class SpotPostureFastEnvCfg_PLAY(SpotPostureFastEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
