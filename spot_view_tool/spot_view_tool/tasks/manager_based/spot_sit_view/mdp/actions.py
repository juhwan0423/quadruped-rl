from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .events import posture_joint_target

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


logger = logging.getLogger(__name__)


class PostureJointPositionAction(ActionTerm):
    """Joint position residual around the commanded posture keyframe."""

    cfg: "PostureJointPositionActionCfg"

    def __init__(self, cfg: "PostureJointPositionActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._asset: Articulation
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        logger.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        command = self._env.command_manager.get_command(self.cfg.command_name)
        posture = command[:, 0]
        joint_reference = posture_joint_target(self._asset, posture)[:, self._joint_ids]
        self._processed_actions = joint_reference + self._raw_actions * self.cfg.scale

        joint_pos_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._processed_actions = torch.clamp(
            self._processed_actions,
            min=joint_pos_limits[..., 0],
            max=joint_pos_limits[..., 1],
        )

    def apply_actions(self):
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0


@configclass
class PostureJointPositionActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = PostureJointPositionAction

    joint_names: list[str] = MISSING
    command_name: str = MISSING
    scale: float = 0.25
    preserve_order: bool = False
