"""Curriculum terms for the Go2 pitch task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pitch_command_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "body_pitch",
    start_range: float = 0.15,
    end_range: float = 0.4,
    num_steps: int = 4000,
) -> torch.Tensor:
    """Widen the body-pitch command range linearly over training.

    Walking and leaning at the same time is much harder than either alone, so the range
    starts narrow and reaches ``end_range`` after ``num_steps`` policy steps. One policy
    step is one ``env.step`` for all environments at once, so a 300-iteration run with
    ``num_steps_per_env = 24`` is 7200 policy steps regardless of ``num_envs``. Returns the
    current half-range so it shows up in the training logs.
    """
    progress = min(env.common_step_counter / num_steps, 1.0)
    half_range = start_range + (end_range - start_range) * progress
    term_cfg = env.command_manager.get_term(command_name).cfg
    term_cfg.ranges.pitch = (-half_range, half_range)
    return torch.tensor(half_range, device=env.device)
