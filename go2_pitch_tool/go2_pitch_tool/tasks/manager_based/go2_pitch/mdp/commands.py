"""Command terms for the Go2 pitch task."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class BodyPitchCommand(CommandTerm):
    """Uniform body-pitch command (1-dim, in radians).

    Positive pitch means the robot noses up. The tracking error is measured on the
    projected gravity vector instead of an euler angle, so it stays well-defined and
    continuous for any body orientation the policy visits.
    """

    cfg: BodyPitchCommandCfg

    def __init__(self, cfg: BodyPitchCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.pitch_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "BodyPitchCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tPitch range: {self.cfg.ranges.pitch}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self.pitch_command

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.pitch_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.pitch)
        # a fraction of the environments is commanded to stay level, so the policy keeps a
        # clean "neutral" behaviour instead of always leaning somewhere
        if self.cfg.rel_level_envs > 0.0:
            ids = torch.as_tensor(env_ids, device=self.device)
            level = torch.rand(len(env_ids), device=self.device) < self.cfg.rel_level_envs
            self.pitch_command[ids[level], 0] = 0.0

    def _update_command(self):
        pass

    def _update_metrics(self):
        # measured pitch from the body-frame gravity vector: g_b = (sin(p), *, -cos(p))
        gravity_b = self.robot.data.projected_gravity_b
        measured_pitch = torch.atan2(gravity_b[:, 0], -gravity_b[:, 2])
        self.metrics["error_pitch"] = torch.abs(self.pitch_command[:, 0] - measured_pitch)


@configclass
class BodyPitchCommandCfg(CommandTermCfg):
    """Configuration for the body-pitch command generator."""

    class_type: type = BodyPitchCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    rel_level_envs: float = 0.2
    """Fraction of environments that are commanded to keep a level body. Defaults to 0.2."""

    @configclass
    class Ranges:
        pitch: tuple[float, float] = MISSING
        """Range for the body-pitch command (in rad). Positive is nose-up."""

    ranges: Ranges = MISSING
