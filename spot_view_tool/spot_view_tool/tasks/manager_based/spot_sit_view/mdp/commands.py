from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class BodyAngleCommand(CommandTerm):
    """Uniform roll, pitch, yaw command for the robot body frame."""

    cfg: "BodyAngleCommandCfg"

    def __init__(self, cfg: "BodyAngleCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.angle_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["error_roll"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "BodyAngleCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tRoll range: {self.cfg.ranges.roll}\n"
        msg += f"\tPitch range: {self.cfg.ranges.pitch}\n"
        msg += f"\tYaw range: {self.cfg.ranges.yaw}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self.angle_command

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.angle_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.roll)
        self.angle_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.pitch)
        self.angle_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.yaw)

    def _update_command(self):
        pass

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        roll, pitch, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
        current = torch.stack((roll, pitch, yaw), dim=-1)
        error = torch.abs(wrap_to_pi(self.angle_command - current)) / max_command_step
        self.metrics["error_roll"] += error[:, 0]
        self.metrics["error_pitch"] += error[:, 1]
        self.metrics["error_yaw"] += error[:, 2]


@configclass
class BodyAngleCommandCfg(CommandTermCfg):
    class_type: type = BodyAngleCommand

    asset_name: str = MISSING

    @configclass
    class Ranges:
        roll: tuple[float, float] = MISSING
        pitch: tuple[float, float] = MISSING
        yaw: tuple[float, float] = MISSING

    ranges: Ranges = MISSING


class PostureViewCommand(CommandTerm):
    """Posture and body-angle command.

    Command layout is ``[posture, roll, pitch, yaw]`` where posture is 0 for seated
    and 1 for standing. Body-angle commands are faded out as posture approaches
    stand so the stand command remains upright.
    """

    cfg: "PostureViewCommandCfg"

    def __init__(self, cfg: "PostureViewCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.posture_view_command = torch.zeros(self.num_envs, 4, device=self.device)
        self.metrics["posture"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_roll"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "PostureViewCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tPosture range: {self.cfg.ranges.posture}\n"
        msg += f"\tRoll range: {self.cfg.ranges.roll}\n"
        msg += f"\tPitch range: {self.cfg.ranges.pitch}\n"
        msg += f"\tYaw range: {self.cfg.ranges.yaw}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self.posture_view_command

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        posture = r.uniform_(*self.cfg.ranges.posture)
        view_scale = 1.0 - posture
        self.posture_view_command[env_ids, 0] = posture
        self.posture_view_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.roll) * view_scale
        self.posture_view_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.pitch) * view_scale
        self.posture_view_command[env_ids, 3] = r.uniform_(*self.cfg.ranges.yaw) * view_scale

    def _update_command(self):
        pass

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        roll, pitch, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
        current = torch.stack((roll, pitch, yaw), dim=-1)
        target = self.posture_view_command[:, 1:4]
        error = torch.abs(wrap_to_pi(target - current)) / max_command_step
        self.metrics["posture"][:] = self.posture_view_command[:, 0]
        self.metrics["error_roll"] += error[:, 0]
        self.metrics["error_pitch"] += error[:, 1]
        self.metrics["error_yaw"] += error[:, 2]


@configclass
class PostureViewCommandCfg(CommandTermCfg):
    class_type: type = PostureViewCommand

    asset_name: str = MISSING

    @configclass
    class Ranges:
        posture: tuple[float, float] = MISSING
        roll: tuple[float, float] = MISSING
        pitch: tuple[float, float] = MISSING
        yaw: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
