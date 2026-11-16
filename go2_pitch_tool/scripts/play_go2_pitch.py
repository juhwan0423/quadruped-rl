"""Play a trained Go2 pitch checkpoint with Isaac Lab's RSL-RL script."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_PATH", PROJECT_ROOT.parents[1] / "IsaacLab"))
RSL_RL_SCRIPT_DIR = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))

import go2_pitch_tool.tasks  # noqa: E402,F401


if "--task" not in sys.argv:
    sys.argv.extend(["--task", "Isaac-Velocity-Flat-Go2-Pitch-Play-v0"])

runpy.run_path(str(RSL_RL_SCRIPT_DIR / "play.py"), run_name="__main__")
