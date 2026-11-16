from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from .events import SIT_JOINT_TARGETS, posture_joint_target

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _target_joint_tensor(asset: Articulation, targets: dict[str, float]) -> torch.Tensor:
    joint_target = asset.data.default_joint_pos.clone()
    for joint_name, target in targets.items():
        joint_target[:, asset.joint_names.index(joint_name)] = target
    return joint_target


def sit_joint_pose_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_target = _target_joint_tensor(asset, SIT_JOINT_TARGETS)
    error = torch.mean(torch.square(asset.data.joint_pos - joint_target), dim=1)
    return torch.exp(-error / std**2)


def posture_joint_pose_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    posture = env.command_manager.get_command(command_name)[:, 0]
    joint_target = posture_joint_target(asset, posture)
    error = torch.mean(torch.square(asset.data.joint_pos - joint_target), dim=1)
    return torch.exp(-error / std**2)


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    height_error = torch.square(asset.data.root_pos_w[:, 2] - (env.scene.env_origins[:, 2] + target_height))
    return torch.exp(-height_error / std**2)


def posture_base_height_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    sit_height: float,
    stand_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    posture = env.command_manager.get_command(command_name)[:, 0]
    target_height = sit_height + posture * (stand_height - sit_height)
    height_error = torch.square(asset.data.root_pos_w[:, 2] - (env.scene.env_origins[:, 2] + target_height))
    return torch.exp(-height_error / std**2)


def base_stability_l2(
    env: ManagerBasedRLEnv,
    lin_weight: float = 1.0,
    ang_weight: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    lin_error = torch.sum(torch.square(asset.data.root_lin_vel_b), dim=1)
    ang_error = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
    return lin_weight * lin_error + ang_weight * ang_error


def upright_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    gravity_xy_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return torch.exp(-gravity_xy_error / std**2)


def body_angle_command_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    roll_weight: float = 1.0,
    pitch_weight: float = 1.0,
    yaw_weight: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    current = torch.stack((roll, pitch, yaw), dim=-1)
    target = env.command_manager.get_command(command_name)
    weights = torch.tensor((roll_weight, pitch_weight, yaw_weight), device=asset.device)
    error = torch.sum(weights * torch.square(wrap_to_pi(current - target)), dim=1)
    return torch.exp(-error / std**2)


def posture_view_angle_command_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    roll_weight: float = 1.0,
    pitch_weight: float = 1.0,
    yaw_weight: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    current = torch.stack((roll, pitch, yaw), dim=-1)
    command = env.command_manager.get_command(command_name)
    posture = command[:, 0]
    target = command[:, 1:4]
    weights = torch.tensor((roll_weight, pitch_weight, yaw_weight), device=asset.device)
    error = torch.sum(weights * torch.square(wrap_to_pi(current - target)), dim=1)
    stand_weight = 0.35 + 0.65 * (1.0 - posture)
    return torch.exp(-stand_weight * error / std**2)


def feet_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.norm(forces, dim=-1) > threshold
    return torch.mean(in_contact.float(), dim=1)


def feet_below_ground_penalty(
    env: ManagerBasedRLEnv,
    min_height: float,
    scale: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(-1)
    penetration = torch.clamp(min_height - foot_z, min=0.0)
    return torch.mean(torch.square(penetration / scale), dim=1)


def feet_contact_force_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_force: float,
    scale: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    excess_force = torch.clamp(forces - max_force, min=0.0)
    return torch.mean(torch.square(excess_force / scale), dim=1)


def feet_below_ground(
    env: ManagerBasedRLEnv,
    min_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(-1)
    return torch.any(foot_z < min_height, dim=1)
