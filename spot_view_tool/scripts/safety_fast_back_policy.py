"""Run a fast backward Spot safety walking behavior.

Run from the IsaacLab checkout:

    ./isaaclab.sh -p ../spot_view_tool/scripts/safety_fast_back_policy.py --device cuda:0
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAAC_WS = PROJECT_ROOT.parent
TOOL_POLICY_ROOT = ISAAC_WS / "policies"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast backward safety behavior using Isaac's Spot walking policy.")
    parser.add_argument(
        "--walk-policy",
        type=Path,
        default=TOOL_POLICY_ROOT / "spot_policy.pt",
        help="TorchScript Spot walking policy.",
    )
    parser.add_argument("--back-seconds", type=float, default=0.5, help="Seconds to walk backward.")
    parser.add_argument("--settle-seconds", type=float, default=0.5, help="Seconds to hold zero command afterward.")
    parser.add_argument("--back-speed", type=float, default=2.0, help="Backward speed command magnitude.")
    parser.add_argument("--strafe-speed", type=float, default=0.0, help="Optional lateral velocity command.")
    parser.add_argument("--yaw-rate", type=float, default=0.0, help="Optional yaw rate command while backing up.")
    parser.add_argument("--root-height", type=float, default=0.50, help="Initial Spot root height.")
    parser.add_argument("--policy-device", default="cpu", choices=["cpu", "cuda:0"], help="TorchScript policy device.")
    parser.add_argument("--action-clip", type=float, default=5.0, help="Clamp raw policy action before target scaling.")
    parser.add_argument("--print-interval", type=float, default=0.25, help="Seconds between diagnostics.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional hard limit on policy steps.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()

if args_cli.walk_policy is None or not args_cli.walk_policy.expanduser().exists():
    raise FileNotFoundError(f"Walking policy not found: {args_cli.walk_policy}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_assets.robots.spot import SPOT_CFG


ACTION_SCALE = 0.20
CONTROL_DT = 0.02
SIM_DT = 0.002
DECIMATION = int(round(CONTROL_DT / SIM_DT))


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


def build_observation(robot: Articulation, command: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
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


def policy_action(policy: torch.jit.ScriptModule, obs: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        action = policy(obs.to(args_cli.policy_device))
    if isinstance(action, (tuple, list)):
        action = action[0]
    action = torch.nan_to_num(action, nan=0.0, posinf=args_cli.action_clip, neginf=-args_cli.action_clip)
    return torch.clamp(action.to(obs.device), -args_cli.action_clip, args_cli.action_clip)


def clamp_joint_target(robot: Articulation, target: torch.Tensor) -> torch.Tensor:
    limits = robot.data.soft_joint_pos_limits
    return torch.clamp(target, min=limits[..., 0], max=limits[..., 1])


def apply_target_for_decimation(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    target: torch.Tensor,
    foot_ids: list[int],
) -> float:
    min_foot_z = math.inf
    for _ in range(DECIMATION):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(SIM_DT)
        if foot_ids:
            foot_z = robot.data.body_pos_w[:, foot_ids, 2]
            min_foot_z = min(min_foot_z, float(foot_z.min()))
    return min_foot_z


def print_status(
    phase: str,
    step: int,
    sim_time: float,
    robot: Articulation,
    command: torch.Tensor,
    min_foot_z: float,
) -> None:
    root = robot.data.root_pos_w[0].detach().cpu().tolist()
    root_vel = robot.data.root_lin_vel_b[0].detach().cpu().tolist()
    cmd = command[0].detach().cpu().tolist()
    print(
        f"[{phase:>6}] step={step:05d} t={sim_time:5.2f}s "
        f"cmd=({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f}) "
        f"root=({root[0]:+.2f}, {root[1]:+.2f}, {root[2]:+.2f}) "
        f"vel_b=({root_vel[0]:+.2f}, {root_vel[1]:+.2f}, {root_vel[2]:+.2f}) "
        f"min_foot_z={min_foot_z:+.4f}",
        flush=True,
    )


def run_policy_phase(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    policy: torch.jit.ScriptModule,
    command: torch.Tensor,
    phase: str,
    duration: float,
    step: int,
    next_print_time: float,
    foot_ids: list[int],
    last_action: torch.Tensor,
) -> tuple[int, float, torch.Tensor]:
    num_steps = int(round(duration / CONTROL_DT))
    for _ in range(num_steps):
        if args_cli.max_steps is not None and step >= args_cli.max_steps:
            raise SystemExit(0)
        obs = build_observation(robot, command, last_action)
        action = policy_action(policy, obs)
        target = clamp_joint_target(robot, torch.nan_to_num(robot.data.default_joint_pos + ACTION_SCALE * action))
        sim_time = step * CONTROL_DT
        min_foot_z = apply_target_for_decimation(sim, robot, target, foot_ids)
        if sim_time >= next_print_time:
            print_status(phase, step, sim_time, robot, command, min_foot_z)
            next_print_time = sim_time + args_cli.print_interval
        last_action = action
        step += 1
    return step, next_print_time, last_action


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=SIM_DT, device=args_cli.device))
    robot = design_scene()
    sim.reset()
    reset_robot(robot, args_cli.root_height)
    robot.update(SIM_DT)
    sim.set_camera_view(eye=(3.0, 3.0, 1.5), target=(0.0, 0.0, 0.35))

    policy = load_policy(args_cli.walk_policy, args_cli.policy_device)
    back_command = torch.tensor(
        (-abs(args_cli.back_speed), args_cli.strafe_speed, args_cli.yaw_rate),
        device=robot.device,
        dtype=torch.float32,
    ).view(1, 3)
    zero_command = torch.zeros_like(back_command)
    last_action = torch.zeros((1, robot.num_joints), device=robot.device)
    foot_ids = [index for index, name in enumerate(robot.body_names) if name.endswith("_foot")]

    print(f"[INFO] walk_policy={args_cli.walk_policy}", flush=True)
    print(f"[INFO] safety back command={back_command[0].detach().cpu().tolist()} duration={args_cli.back_seconds:.2f}s", flush=True)
    print(f"[INFO] policy_device={args_cli.policy_device} sim_device={robot.device}", flush=True)

    step = 0
    next_print_time = 0.0
    step, next_print_time, last_action = run_policy_phase(
        sim, robot, policy, back_command, "back", args_cli.back_seconds, step, next_print_time, foot_ids, last_action
    )
    if args_cli.settle_seconds > 0.0:
        last_action.zero_()
        step, next_print_time, last_action = run_policy_phase(
            sim,
            robot,
            policy,
            zero_command,
            "settle",
            args_cli.settle_seconds,
            step,
            next_print_time,
            foot_ids,
            last_action,
        )

    if args_cli.max_steps is not None:
        raise SystemExit(0)

    while simulation_app.is_running():
        step, next_print_time, last_action = run_policy_phase(
            sim,
            robot,
            policy,
            zero_command,
            "hold",
            CONTROL_DT,
            step,
            next_print_time,
            foot_ids,
            last_action,
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
