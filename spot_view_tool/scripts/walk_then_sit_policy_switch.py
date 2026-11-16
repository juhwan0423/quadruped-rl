"""Spawn one Spot and switch policies: walk -> sit -> stand -> walk.

Run from the IsaacLab checkout:

    ./isaaclab.sh -p ../spot_view_tool/scripts/walk_then_sit_policy_switch.py --device cuda:0
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAAC_WS = PROJECT_ROOT.parent
TOOL_POLICY_ROOT = ISAAC_WS / "policies"
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    default_walk_policy = TOOL_POLICY_ROOT / "spot_policy.pt"
    default_posture_policy = TOOL_POLICY_ROOT / "spot_posture_pitch_fast_policy.pt"
    default_posture_checkpoint = (
        TOOL_POLICY_ROOT / "spot_posture_pitch_fast_2026-06-23_16-47-16_fast_mvp_model_4999.pt"
    )

    parser = argparse.ArgumentParser(description="Switch one spawned Spot between Isaac walk policy and posture policy.")
    parser.add_argument("--walk-policy", type=Path, default=default_walk_policy, help="TorchScript walking policy.pt.")
    parser.add_argument(
        "--posture-policy",
        type=Path,
        default=default_posture_policy,
        help="TorchScript posture policy exported from the fast RSL-RL checkpoint.",
    )
    parser.add_argument(
        "--posture-checkpoint",
        type=Path,
        default=default_posture_checkpoint,
        help="Original fast posture RSL-RL checkpoint kept with the tool policies for provenance.",
    )
    parser.add_argument("--walk-seconds", type=float, default=4.0, help="Seconds to run the walking policy.")
    parser.add_argument("--stop-seconds", type=float, default=0.6, help="Seconds to run walking policy with zero velocity.")
    parser.add_argument(
        "--stand-seconds",
        type=float,
        default=0.2,
        help="Initial stand command duration before ramping to sit.",
    )
    parser.add_argument(
        "--to-sit-seconds",
        type=float,
        default=0.8,
        help="Posture command ramp duration from stand to sit.",
    )
    parser.add_argument(
        "--sit-seconds",
        type=float,
        default=0.5,
        help="Seated view hold duration.",
    )
    parser.add_argument(
        "--to-stand-seconds",
        type=float,
        default=0.8,
        help="Posture command ramp duration from sit back to stand.",
    )
    parser.add_argument(
        "--final-stand-seconds",
        type=float,
        default=0.5,
        help="Final stand command duration before handing back to walking.",
    )
    parser.add_argument(
        "--stand-hold-seconds",
        type=float,
        default=0.0,
        help="Extra seconds to run walking policy with zero command after posture stand.",
    )
    parser.add_argument(
        "--return-walk-seconds",
        type=float,
        default=4.0,
        help="Seconds to run walking policy again after the stand-up blend.",
    )
    parser.add_argument(
        "--walk-command",
        nargs=3,
        type=float,
        default=(0.6, 0.0, 0.0),
        metavar=("VX", "VY", "WZ"),
        help="Walking command in body frame: x velocity, y velocity, yaw rate.",
    )
    parser.add_argument(
        "--return-walk-command",
        nargs=3,
        type=float,
        default=None,
        metavar=("VX", "VY", "WZ"),
        help="Walking command after standing up. Defaults to --walk-command.",
    )
    parser.add_argument(
        "--view-command",
        nargs=3,
        type=float,
        default=(0.0, -0.36, 0.0),
        metavar=("ROLL", "PITCH", "YAW"),
        help="Seated view command in radians.",
    )
    parser.add_argument("--root-height", type=float, default=0.50, help="Initial Spot root height.")
    parser.add_argument("--action-clip", type=float, default=5.0, help="Clamp raw policy action before target scaling.")
    parser.add_argument("--policy-device", default="cpu", choices=["cpu", "cuda:0"], help="TorchScript policy device.")
    parser.add_argument("--debug-steps", type=int, default=0, help="Print detailed diagnostics for the first N steps.")
    parser.add_argument("--print-interval", type=float, default=0.5, help="Seconds between foot-height diagnostics.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional hard limit on policy steps.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()

if args_cli.walk_policy is None or not args_cli.walk_policy.expanduser().exists():
    raise FileNotFoundError(f"Walking exported policy.pt not found: {args_cli.walk_policy}")
if args_cli.posture_policy is None or not args_cli.posture_policy.expanduser().exists():
    raise FileNotFoundError(f"Posture TorchScript policy.pt not found: {args_cli.posture_policy}")
if args_cli.posture_checkpoint is not None and not args_cli.posture_checkpoint.expanduser().exists():
    raise FileNotFoundError(f"Posture checkpoint not found: {args_cli.posture_checkpoint}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_assets.robots.spot import SPOT_CFG
from spot_view_tool.tasks.manager_based.spot_sit_view.mdp.events import posture_joint_target


WALK_ACTION_SCALE = 0.20
POSTURE_ACTION_SCALE = 0.25
CONTROL_DT = 0.02
SIM_DT = 0.002
DECIMATION = int(round(CONTROL_DT / SIM_DT))


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def design_scene() -> Articulation:
    ground_cfg = sim_utils.CuboidCfg(
        size=(40.0, 40.0, 0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18)),
    )
    ground_cfg.func("/World/FlatGround", ground_cfg, translation=(0.0, 0.0, -0.025))

    light_cfg = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    spot_cfg = copy.deepcopy(SPOT_CFG).replace(prim_path="/World/Robot")
    return Articulation(spot_cfg)


def load_policy(path: Path, device: str) -> torch.jit.ScriptModule:
    policy = torch.jit.load(str(path.expanduser().resolve()), map_location=torch.device(device))
    policy.eval()
    return policy


def clamp_joint_target(robot: Articulation, target: torch.Tensor) -> torch.Tensor:
    limits = robot.data.soft_joint_pos_limits
    return torch.clamp(target, min=limits[..., 0], max=limits[..., 1])


def reset_robot(robot: Articulation, root_height: float) -> None:
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] = torch.tensor((0.0, 0.0, root_height), device=robot.device)
    root_state[:, 7:] = 0.0

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    robot.reset()


def build_walk_observation(robot: Articulation, command: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            robot.data.root_lin_vel_b,
            robot.data.root_ang_vel_b,
            robot.data.projected_gravity_b,
            command,
            robot.data.joint_pos - robot.data.default_joint_pos,
            robot.data.joint_vel - robot.data.default_joint_vel,
            last_action,
        ),
        dim=-1,
    )


def build_posture_observation(robot: Articulation, command: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            robot.data.root_lin_vel_b,
            robot.data.root_ang_vel_b,
            robot.data.projected_gravity_b,
            robot.data.joint_pos - robot.data.default_joint_pos,
            robot.data.joint_vel - robot.data.default_joint_vel,
            command,
            last_action,
        ),
        dim=-1,
    )


def phase_posture_command(t: float, view_command: torch.Tensor) -> tuple[str, torch.Tensor]:
    stand = args_cli.stand_seconds
    to_sit = args_cli.to_sit_seconds
    sit = args_cli.sit_seconds
    to_stand = args_cli.to_stand_seconds
    final_stand = args_cli.final_stand_seconds

    command = torch.zeros((1, 4), device=view_command.device, dtype=view_command.dtype)
    if t < stand:
        phase = "stand"
        posture = 1.0
        angles = torch.zeros_like(view_command)
    elif t < stand + to_sit:
        phase = "to_sit"
        alpha = smoothstep((t - stand) / max(to_sit, 1.0e-6))
        posture = 1.0 - alpha
        angles = alpha * view_command
    elif t < stand + to_sit + sit:
        phase = "sit"
        posture = 0.0
        angles = view_command
    elif t < stand + to_sit + sit + to_stand:
        phase = "to_stand"
        alpha = smoothstep((t - stand - to_sit - sit) / max(to_stand, 1.0e-6))
        posture = alpha
        angles = (1.0 - alpha) * view_command
    else:
        phase = "stand_end"
        posture = 1.0
        angles = torch.zeros_like(view_command)

    command[:, 0] = posture
    command[:, 1:4] = angles
    return phase, command


def policy_action(
    policy: torch.jit.ScriptModule,
    obs: torch.Tensor,
    action_clip: float,
    policy_device: str,
    action_device: str,
) -> torch.Tensor:
    with torch.inference_mode():
        action = policy(obs.to(policy_device))
    if isinstance(action, (tuple, list)):
        action = action[0]
    action = torch.nan_to_num(action, nan=0.0, posinf=action_clip, neginf=-action_clip)
    return torch.clamp(action.to(action_device), -action_clip, action_clip)


def apply_target_for_decimation(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    target: torch.Tensor,
    foot_ids: list[int],
) -> tuple[float, float]:
    min_foot_z = math.inf
    max_foot_force = 0.0
    for _ in range(DECIMATION):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(SIM_DT)
        if foot_ids:
            foot_z = robot.data.body_pos_w[:, foot_ids, 2]
            min_foot_z = min(min_foot_z, float(foot_z.min()))
    return min_foot_z, max_foot_force


def print_status(phase: str, step: int, sim_time: float, robot: Articulation, foot_ids: list[int]) -> None:
    root = robot.data.root_pos_w[0].detach().cpu().tolist()
    if foot_ids:
        foot_z = robot.data.body_pos_w[:, foot_ids, 2]
        min_foot_z = float(foot_z.min())
        per_foot = [float(foot_z[0, i]) for i in range(len(foot_ids))]
    else:
        min_foot_z = float("nan")
        per_foot = []
    print(
        f"[{phase:>4}] step={step:05d} t={sim_time:6.2f}s "
        f"root=({root[0]:+.2f}, {root[1]:+.2f}, {root[2]:+.2f}) "
        f"min_foot_z={min_foot_z:+.4f} foot_z={[round(v, 4) for v in per_foot]}",
        flush=True,
    )


def main() -> None:
    print("[INFO] Creating simulation context...", flush=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=SIM_DT, device=args_cli.device))
    print("[INFO] Spawning Spot scene...", flush=True)
    robot = design_scene()
    print("[INFO] Resetting simulation...", flush=True)
    sim.reset()
    print("[INFO] Resetting robot state...", flush=True)
    reset_robot(robot, args_cli.root_height)
    robot.update(SIM_DT)

    sim.set_camera_view(eye=(3.0, 3.0, 1.5), target=(0.0, 0.0, 0.35))

    print("[INFO] Loading policies...", flush=True)
    walk_policy = load_policy(args_cli.walk_policy, args_cli.policy_device)
    posture_policy = load_policy(args_cli.posture_policy, args_cli.policy_device)

    foot_ids = [index for index, name in enumerate(robot.body_names) if name.endswith("_foot")]
    print(f"[INFO] walk_policy={args_cli.walk_policy}", flush=True)
    print(f"[INFO] posture_policy={args_cli.posture_policy}", flush=True)
    print(f"[INFO] posture_checkpoint={args_cli.posture_checkpoint}", flush=True)
    print(f"[INFO] policy_device={args_cli.policy_device} sim_device={robot.device}", flush=True)
    print(f"[INFO] joints={robot.joint_names}", flush=True)
    print(f"[INFO] feet={[robot.body_names[index] for index in foot_ids]}", flush=True)
    print(
        "[INFO] Phases: walk policy -> zero-command walk policy -> posture sit/stand policy -> walk policy",
        flush=True,
    )

    walk_command = torch.tensor(args_cli.walk_command, device=robot.device, dtype=torch.float32).view(1, 3)
    return_walk_command_values = args_cli.return_walk_command if args_cli.return_walk_command is not None else args_cli.walk_command
    return_walk_command = torch.tensor(return_walk_command_values, device=robot.device, dtype=torch.float32).view(1, 3)
    zero_velocity_command = torch.zeros_like(walk_command)
    view_command = torch.tensor(args_cli.view_command, device=robot.device, dtype=torch.float32).view(1, 3)
    last_action = torch.zeros((1, robot.num_joints), device=robot.device)

    walk_phases = (
        (
            "walk",
            args_cli.walk_seconds,
            walk_policy,
            walk_command,
            robot.data.default_joint_pos,
            WALK_ACTION_SCALE,
            build_walk_observation,
        ),
        (
            "stop",
            args_cli.stop_seconds,
            walk_policy,
            zero_velocity_command,
            robot.data.default_joint_pos,
            WALK_ACTION_SCALE,
            build_walk_observation,
        ),
    )

    step = 0
    next_print_time = 0.0
    for phase_name, duration, policy, command, action_offset, action_scale, obs_builder in walk_phases:
        if duration <= 0.0:
            continue
        phase_steps = int(round(duration / CONTROL_DT))
        for _ in range(phase_steps):
            if args_cli.max_steps is not None and step >= args_cli.max_steps:
                raise SystemExit(0)
            if step < args_cli.debug_steps:
                print(f"[DEBUG] step={step} phase={phase_name}: building obs", flush=True)
            obs = obs_builder(robot, command, last_action)
            if step < args_cli.debug_steps:
                print(
                    f"[DEBUG] step={step}: obs finite={bool(torch.isfinite(obs).all())} "
                    f"obs_min={float(obs.min()):+.4f} obs_max={float(obs.max()):+.4f}",
                    flush=True,
                )
            action = policy_action(policy, obs, args_cli.action_clip, args_cli.policy_device, robot.device)
            if step < args_cli.debug_steps:
                print(
                    f"[DEBUG] step={step}: action finite={bool(torch.isfinite(action).all())} "
                    f"action_min={float(action.min()):+.4f} action_max={float(action.max()):+.4f}",
                    flush=True,
                )
            target = clamp_joint_target(robot, torch.nan_to_num(action_offset + action_scale * action))
            if step < args_cli.debug_steps:
                print(
                    f"[DEBUG] step={step}: target_min={float(target.min()):+.4f} "
                    f"target_max={float(target.max()):+.4f}; stepping sim",
                    flush=True,
                )
            last_action = action
            sim_time = step * CONTROL_DT
            apply_target_for_decimation(sim, robot, target, foot_ids)
            if step < args_cli.debug_steps:
                print(f"[DEBUG] step={step}: sim stepped", flush=True)
            if sim_time >= next_print_time:
                print_status(phase_name, step, sim_time, robot, foot_ids)
                next_print_time = sim_time + args_cli.print_interval
            step += 1

    posture_total_seconds = (
        args_cli.stand_seconds
        + args_cli.to_sit_seconds
        + args_cli.sit_seconds
        + args_cli.to_stand_seconds
        + args_cli.final_stand_seconds
    )
    if posture_total_seconds > 0.0:
        last_action.zero_()
        posture_steps = int(round(posture_total_seconds / CONTROL_DT))
        for posture_step in range(posture_steps):
            if args_cli.max_steps is not None and step >= args_cli.max_steps:
                raise SystemExit(0)
            local_time = posture_step * CONTROL_DT
            phase_name, posture_command = phase_posture_command(local_time, view_command)
            obs = build_posture_observation(robot, posture_command, last_action)
            action = policy_action(posture_policy, obs, args_cli.action_clip, args_cli.policy_device, robot.device)
            action_offset = posture_joint_target(robot, posture_command[:, 0])
            target = clamp_joint_target(robot, torch.nan_to_num(action_offset + POSTURE_ACTION_SCALE * action))
            last_action = action
            sim_time = step * CONTROL_DT
            apply_target_for_decimation(sim, robot, target, foot_ids)
            if sim_time >= next_print_time:
                print_status(phase_name, step, sim_time, robot, foot_ids)
                next_print_time = sim_time + args_cli.print_interval
            step += 1

    if args_cli.stand_hold_seconds > 0.0:
        last_action.zero_()
        hold_steps = int(round(args_cli.stand_hold_seconds / CONTROL_DT))
        for _ in range(hold_steps):
            if args_cli.max_steps is not None and step >= args_cli.max_steps:
                raise SystemExit(0)
            obs = build_walk_observation(robot, zero_velocity_command, last_action)
            action = policy_action(walk_policy, obs, args_cli.action_clip, args_cli.policy_device, robot.device)
            target = clamp_joint_target(robot, torch.nan_to_num(robot.data.default_joint_pos + WALK_ACTION_SCALE * action))
            last_action = action
            sim_time = step * CONTROL_DT
            apply_target_for_decimation(sim, robot, target, foot_ids)
            if sim_time >= next_print_time:
                print_status("hold", step, sim_time, robot, foot_ids)
                next_print_time = sim_time + args_cli.print_interval
            step += 1

    if args_cli.return_walk_seconds > 0.0:
        last_action.zero_()
        return_walk_steps = int(round(args_cli.return_walk_seconds / CONTROL_DT))
        for _ in range(return_walk_steps):
            if args_cli.max_steps is not None and step >= args_cli.max_steps:
                raise SystemExit(0)
            obs = build_walk_observation(robot, return_walk_command, last_action)
            action = policy_action(walk_policy, obs, args_cli.action_clip, args_cli.policy_device, robot.device)
            target = clamp_joint_target(robot, torch.nan_to_num(robot.data.default_joint_pos + WALK_ACTION_SCALE * action))
            last_action = action
            sim_time = step * CONTROL_DT
            apply_target_for_decimation(sim, robot, target, foot_ids)
            if sim_time >= next_print_time:
                print_status("walk2", step, sim_time, robot, foot_ids)
                next_print_time = sim_time + args_cli.print_interval
            step += 1

    if args_cli.max_steps is not None:
        raise SystemExit(0)

    while simulation_app.is_running():
        robot.write_data_to_sim()
        sim.step()
        robot.update(SIM_DT)


if __name__ == "__main__":
    main()
    simulation_app.close()
