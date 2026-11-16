"""Reward terms for the Go2 pitch task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_body_pitch_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of the commanded body pitch with an exponential kernel.

    For a body pitched by ``p``, the gravity vector expressed in the body frame is
    ``(sin(p), 0, -cos(p))``. Comparing against that target avoids euler-angle
    conversions and their wrapping issues, and is only weakly coupled to roll.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    target_pitch = env.command_manager.get_command(command_name)[:, 0]
    gravity_b = asset.data.projected_gravity_b
    error = (gravity_b[:, 0] - torch.sin(target_pitch)) ** 2 + (gravity_b[:, 2] + torch.cos(target_pitch)) ** 2
    return torch.exp(-error / std**2)


def body_roll_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize body roll only.

    This replaces ``flat_orientation_l2`` (which penalizes both roll and pitch and would
    directly fight the pitch command) while still keeping the robot upright sideways.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 1])
