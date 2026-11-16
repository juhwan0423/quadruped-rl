from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


SIT_JOINT_TARGETS = {
    "fl_hx": 0.10,
    "fr_hx": -0.10,
    "hl_hx": 0.12,
    "hr_hx": -0.12,
    "fl_hy": 0.85,
    "fr_hy": 0.85,
    "hl_hy": 1.45,
    "hr_hy": 1.45,
    "fl_kn": -1.65,
    "fr_kn": -1.65,
    "hl_kn": -2.25,
    "hr_kn": -2.25,
}

LOW_CROUCH_JOINT_TARGETS = {
    "fl_hx": 0.10,
    "fr_hx": -0.10,
    "hl_hx": 0.12,
    "hr_hx": -0.12,
    "fl_hy": 0.88,
    "fr_hy": 0.88,
    "hl_hy": 1.30,
    "hr_hy": 1.30,
    "fl_kn": -1.65,
    "fr_kn": -1.65,
    "hl_kn": -1.95,
    "hr_kn": -1.95,
}

HIGH_CROUCH_JOINT_TARGETS = {
    "fl_hx": 0.10,
    "fr_hx": -0.10,
    "hl_hx": 0.10,
    "hr_hx": -0.10,
    "fl_hy": 0.90,
    "fr_hy": 0.90,
    "hl_hy": 1.15,
    "hr_hy": 1.15,
    "fl_kn": -1.55,
    "fr_kn": -1.55,
    "hl_kn": -1.65,
    "hr_kn": -1.65,
}


def _target_joint_tensor(asset: Articulation, env_ids: torch.Tensor, targets: dict[str, float]) -> torch.Tensor:
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    for joint_name, target in targets.items():
        joint_pos[:, asset.joint_names.index(joint_name)] = target
    return joint_pos


def posture_joint_target(asset: Articulation, posture: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Interpolate stand/high-crouch/low-crouch/sit keyframes.

    ``posture`` uses 0.0 for seated and 1.0 for standing.
    """
    if env_ids is None:
        target = asset.data.default_joint_pos.clone()
    else:
        target = asset.data.default_joint_pos[env_ids].clone()

    posture = posture.reshape(-1).clamp(0.0, 1.0)
    sit = target.clone()
    low = target.clone()
    high = target.clone()
    for joint_name, value in SIT_JOINT_TARGETS.items():
        sit[:, asset.joint_names.index(joint_name)] = value
    for joint_name, value in LOW_CROUCH_JOINT_TARGETS.items():
        low[:, asset.joint_names.index(joint_name)] = value
    for joint_name, value in HIGH_CROUCH_JOINT_TARGETS.items():
        high[:, asset.joint_names.index(joint_name)] = value

    keyframes = (
        (0.0, sit),
        (0.35, low),
        (0.70, high),
        (1.0, target),
    )
    out = torch.empty_like(target)
    for index, (start_p, start_target) in enumerate(keyframes[:-1]):
        end_p, end_target = keyframes[index + 1]
        mask = (posture >= start_p) & (posture <= end_p)
        if not torch.any(mask):
            continue
        alpha = ((posture[mask] - start_p) / (end_p - start_p)).unsqueeze(-1)
        out[mask] = (1.0 - alpha) * start_target[mask] + alpha * end_target[mask]
    out[posture <= keyframes[0][0]] = keyframes[0][1][posture <= keyframes[0][0]]
    out[posture >= keyframes[-1][0]] = keyframes[-1][1][posture >= keyframes[-1][0]]
    return out


def reset_root_to_sit_height(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    base_height: float,
    roll_range: tuple[float, float] = (-0.15, 0.15),
    pitch_range: tuple[float, float] = (-0.20, 0.20),
    yaw_range: tuple[float, float] = (-0.05, 0.05),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()

    positions = env.scene.env_origins[env_ids].clone()
    positions[:, 2] += base_height

    roll = sample_uniform(roll_range[0], roll_range[1], (len(env_ids),), device=asset.device)
    pitch = sample_uniform(pitch_range[0], pitch_range[1], (len(env_ids),), device=asset.device)
    yaw = sample_uniform(yaw_range[0], yaw_range[1], (len(env_ids),), device=asset.device)
    orientations = quat_from_euler_xyz(roll, pitch, yaw)
    velocities = torch.zeros_like(root_states[:, 7:13])

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_to_sit_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_noise: float = 0.08,
    velocity_noise: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = _target_joint_tensor(asset, env_ids, SIT_JOINT_TARGETS)
    joint_pos += sample_uniform(-position_noise, position_noise, joint_pos.shape, device=asset.device)
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids, ...]
    joint_pos = torch.clamp(joint_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1])

    joint_vel = sample_uniform(-velocity_noise, velocity_noise, joint_pos.shape, device=asset.device)
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = torch.clamp(joint_vel, min=-joint_vel_limits, max=joint_vel_limits)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_robot_to_random_posture(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    posture_range: tuple[float, float],
    sit_base_height: float,
    stand_base_height: float,
    roll_range: tuple[float, float] = (-0.04, 0.04),
    pitch_range: tuple[float, float] = (-0.04, 0.04),
    yaw_range: tuple[float, float] = (-0.35, 0.35),
    position_noise: float = 0.04,
    velocity_noise: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    posture = sample_uniform(posture_range[0], posture_range[1], (len(env_ids),), device=asset.device)

    root_states = asset.data.default_root_state[env_ids].clone()
    positions = env.scene.env_origins[env_ids].clone()
    positions[:, 2] += sit_base_height + posture * (stand_base_height - sit_base_height)

    roll = sample_uniform(roll_range[0], roll_range[1], (len(env_ids),), device=asset.device)
    pitch = sample_uniform(pitch_range[0], pitch_range[1], (len(env_ids),), device=asset.device)
    yaw = sample_uniform(yaw_range[0], yaw_range[1], (len(env_ids),), device=asset.device)
    orientations = quat_from_euler_xyz(roll, pitch, yaw)
    velocities = torch.zeros_like(root_states[:, 7:13])

    joint_pos = posture_joint_target(asset, posture, env_ids)
    joint_pos += sample_uniform(-position_noise, position_noise, joint_pos.shape, device=asset.device)
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids, ...]
    joint_pos = torch.clamp(joint_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1])

    joint_vel = sample_uniform(-velocity_noise, velocity_noise, joint_pos.shape, device=asset.device)
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = torch.clamp(joint_vel, min=-joint_vel_limits, max=joint_vel_limits)

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
