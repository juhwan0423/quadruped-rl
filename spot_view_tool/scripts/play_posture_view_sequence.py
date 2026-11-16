"""Play a posture-view checkpoint with a fixed stand -> sit/view -> stand command sequence."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_PATH", PROJECT_ROOT.parents[1] / "IsaacLab"))
RSL_RL_SCRIPT_DIR = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))

from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # isort: skip  # noqa: E402


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Play Spot posture-view policy with explicit command phases.")
    parser.add_argument("--task", type=str, default="Isaac-Spot-Posture-View-Play-V0", help="Task name.")
    parser.add_argument(
        "--agent",
        type=str,
        default="rsl_rl_cfg_entry_point",
        help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument("--stand-seconds", type=float, default=0.5, help="Initial stand hold duration.")
    parser.add_argument("--to-sit-seconds", type=float, default=1.5, help="Stand-to-sit command ramp duration.")
    parser.add_argument("--sit-seconds", type=float, default=2.0, help="Seated view hold duration.")
    parser.add_argument("--to-stand-seconds", type=float, default=1.5, help="Sit-to-stand command ramp duration.")
    parser.add_argument("--final-stand-seconds", type=float, default=1.0, help="Final stand hold duration.")
    parser.add_argument("--cycle", action="store_true", help="Repeat the command sequence until the app closes.")
    parser.add_argument("--max-seconds", type=float, default=None, help="Stop after this many wall-clock policy seconds.")
    parser.add_argument("--view-roll", type=float, default=0.0, help="Seated roll command in radians.")
    parser.add_argument("--view-pitch", type=float, default=-0.12, help="Seated pitch command in radians.")
    parser.add_argument("--view-yaw", type=float, default=0.0, help="Seated yaw command in radians.")
    parser.add_argument("--pitch", type=float, default=None, help="Shortcut for --view-pitch in radians.")
    parser.add_argument("--keyboard", action="store_true", help="Enable live seated angle tuning from the viewport.")
    parser.add_argument("--angle-step-deg", type=float, default=2.0, help="Keyboard angle step in degrees.")
    parser.add_argument("--print-interval", type=float, default=0.5, help="Seconds between diagnostics.")
    parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    if args_cli.pitch is not None:
        args_cli.view_pitch = args_cli.pitch
    return args_cli, hydra_args


args_cli, hydra_args = parse_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import spot_view_tool.tasks  # noqa: E402,F401


installed_version = metadata.version("rsl-rl-lib")


class ViewAngleController:
    """Mutable seated view-angle command, optionally driven by Omniverse keyboard events."""

    def __init__(self, roll: float, pitch: float, yaw: float, step_deg: float, enable_keyboard: bool):
        self.default_angles = torch.tensor((roll, pitch, yaw), dtype=torch.float32)
        self.angles = self.default_angles.clone()
        self.step = torch.deg2rad(torch.tensor(step_deg, dtype=torch.float32)).item()
        self._input = None
        self._keyboard = None
        self._sub_keyboard = None

        if enable_keyboard:
            import carb.input
            import omni.appwindow

            self._input = carb.input.acquire_input_interface()
            self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
            print("[INFO] Live seated-angle controls:")
            print("       I/K pitch +/-   J/L yaw +/-   U/O roll +/-   R reset")

    def tensor(self, device: torch.device) -> torch.Tensor:
        return self.angles.to(device=device)

    def close(self) -> None:
        if self._input is not None and self._keyboard is not None and self._sub_keyboard is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
            self._sub_keyboard = None

    def _on_keyboard_event(self, event, *args, **kwargs):
        import carb.input

        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True

        key = event.input.name
        if key == "I":
            self.angles[1] += self.step
        elif key == "K":
            self.angles[1] -= self.step
        elif key == "J":
            self.angles[2] += self.step
        elif key == "L":
            self.angles[2] -= self.step
        elif key == "U":
            self.angles[0] += self.step
        elif key == "O":
            self.angles[0] -= self.step
        elif key == "R":
            self.angles[:] = self.default_angles
        else:
            return True

        self.angles[0] = torch.clamp(self.angles[0], -0.30, 0.30)
        self.angles[1] = torch.clamp(self.angles[1], -0.35, 0.35)
        self.angles[2] = torch.clamp(self.angles[2], -0.60, 0.60)
        print(
            "[INFO] view angles rad: "
            f"roll={self.angles[0].item():+.3f}, pitch={self.angles[1].item():+.3f}, yaw={self.angles[2].item():+.3f}"
        )
        return True


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def phase_command(t: float, device: torch.device, view_angles: torch.Tensor) -> tuple[str, torch.Tensor]:
    stand = args_cli.stand_seconds
    to_sit = args_cli.to_sit_seconds
    sit = args_cli.sit_seconds
    to_stand = args_cli.to_stand_seconds
    final_stand = args_cli.final_stand_seconds
    total = stand + to_sit + sit + to_stand + final_stand

    if args_cli.cycle and total > 0.0:
        t = t % total
    else:
        t = min(t, total)

    view = view_angles.to(device=device)
    command = torch.zeros(4, device=device)

    if t < stand:
        phase = "stand"
        posture = 1.0
        angles = torch.zeros(3, device=device)
    elif t < stand + to_sit:
        phase = "to_sit"
        alpha = smoothstep((t - stand) / max(to_sit, 1.0e-6))
        posture = 1.0 - alpha
        angles = alpha * view
    elif t < stand + to_sit + sit:
        phase = "sit_view"
        posture = 0.0
        angles = view
    elif t < stand + to_sit + sit + to_stand:
        phase = "to_stand"
        alpha = smoothstep((t - stand - to_sit - sit) / max(to_stand, 1.0e-6))
        posture = alpha
        angles = (1.0 - alpha) * view
    else:
        phase = "stand_end"
        posture = 1.0
        angles = torch.zeros(3, device=device)

    command[0] = posture
    command[1:4] = angles
    return phase, command


def set_command(env, command: torch.Tensor) -> None:
    command_tensor = env.command_manager.get_command("posture_view")
    command_tensor[:] = command.unsqueeze(0).repeat(env.num_envs, 1)
    command_term = env.command_manager.get_term("posture_view")
    command_term.time_left[:] = 1.0e6


def print_diagnostics(env, phase: str, command: torch.Tensor, step: int, t: float, foot_ids: list[int]) -> None:
    robot = env.scene["robot"]
    root = robot.data.root_pos_w[0].detach().cpu()
    roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    rpy = torch.stack((roll, pitch, yaw), dim=-1)[0].detach().cpu()
    angle_error = command[1:4].detach().cpu() - rpy
    foot_z = robot.data.body_pos_w[0, foot_ids, 2].detach().cpu() - env.scene.env_origins[0, 2].detach().cpu()
    print(
        f"[{phase:>9}] step={step:05d} t={t:6.2f}s "
        f"cmd=({command[0].item():.2f}, {command[1].item():+.2f}, {command[2].item():+.2f}, {command[3].item():+.2f}) "
        f"rpy=({rpy[0].item():+.2f}, {rpy[1].item():+.2f}, {rpy[2].item():+.2f}) "
        f"err=({angle_error[0].item():+.2f}, {angle_error[1].item():+.2f}, {angle_error[2].item():+.2f}) "
        f"root=({root[0].item():+.2f}, {root[1].item():+.2f}, {root[2].item():+.2f}) "
        f"min_foot_z={torch.min(foot_z).item():+.4f}"
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped
    print(f"[INFO] Requested num_envs={args_cli.num_envs}, actual num_envs={base_env.num_envs}")
    print("[INFO] Diagnostics below report env_0 only; commands are applied to every env.")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version))
    policy = runner.get_inference_policy(device=base_env.device)

    robot = base_env.scene["robot"]
    foot_ids = [index for index, name in enumerate(robot.body_names) if name.endswith("_foot")]
    if not foot_ids:
        raise RuntimeError(f"Could not find Spot foot bodies in: {robot.body_names}")

    dt = base_env.step_dt
    step = 0
    next_print_t = 0.0
    obs = env.get_observations()
    view_controller = ViewAngleController(
        args_cli.view_roll,
        args_cli.view_pitch,
        args_cli.view_yaw,
        args_cli.angle_step_deg,
        args_cli.keyboard,
    )

    while simulation_app.is_running():
        start_time = time.time()
        t = step * dt
        if args_cli.max_seconds is not None and t >= args_cli.max_seconds:
            break

        phase, command = phase_command(t, base_env.device, view_controller.tensor(base_env.device))
        set_command(base_env, command)
        obs = env.get_observations()

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)

        if t >= next_print_t:
            print_diagnostics(base_env, phase, command, step, t, foot_ids)
            next_print_t += args_cli.print_interval

        step += 1
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0.0:
            time.sleep(sleep_time)

    view_controller.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
