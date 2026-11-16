"""Run a scripted Spot sit/stand primitive without RL.

Run from the IsaacLab checkout:

    ./isaaclab.sh -p ../spot_view_tool/scripts/sit_stand_primitive.py --device cuda:0
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scripted Spot sit/stand primitive on flat ground.")
    parser.add_argument("--cycles", type=int, default=1, help="Number of sit->stand cycles.")
    parser.add_argument("--stand-hold-seconds", type=float, default=2.0)
    parser.add_argument("--sit-down-seconds", type=float, default=4.0)
    parser.add_argument("--sit-hold-seconds", type=float, default=2.0)
    parser.add_argument("--stand-up-seconds", type=float, default=6.0)
    parser.add_argument("--root-height", type=float, default=0.50, help="Initial Spot root height.")
    parser.add_argument("--print-interval", type=float, default=0.5, help="Seconds between diagnostics.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional hard limit on control steps.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_assets.robots.spot import SPOT_CFG
from spot_view_tool.tasks.manager_based.spot_sit_view.mdp.events import SIT_JOINT_TARGETS


CONTROL_DT = 0.02
SIM_DT = 0.002
DECIMATION = int(round(CONTROL_DT / SIM_DT))


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def design_scene() -> Articulation:
    ground_cfg = sim_utils.CuboidCfg(
        size=(20.0, 20.0, 0.05),
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


def joint_target_from_dict(robot: Articulation, joint_targets: dict[str, float]) -> torch.Tensor:
    target = robot.data.default_joint_pos.clone()
    unknown_names = sorted(set(joint_targets) - set(robot.joint_names))
    if unknown_names:
        raise ValueError(f"Unknown Spot joint(s): {unknown_names}")
    for joint_name, value in joint_targets.items():
        target[:, robot.joint_names.index(joint_name)] = value
    return target


def clamp_joint_target(robot: Articulation, target: torch.Tensor) -> torch.Tensor:
    limits = robot.data.soft_joint_pos_limits
    return torch.clamp(target, min=limits[..., 0], max=limits[..., 1])


def primitive_keyframes(robot: Articulation) -> dict[str, torch.Tensor]:
    low_crouch = {
        "fl_hx": 0.10,
        "fr_hx": -0.10,
        "hl_hx": 0.12,
        "hr_hx": -0.12,
        "fl_hy": 0.88,
        "fr_hy": 0.88,
        "hl_hy": 1.30,
        "hr_hy": 1.30,
        "fl_kn": -1.65,
        "fr_kn": -1.65,
        "hl_kn": -1.95,
        "hr_kn": -1.95,
    }
    high_crouch = {
        "fl_hx": 0.10,
        "fr_hx": -0.10,
        "hl_hx": 0.10,
        "hr_hx": -0.10,
        "fl_hy": 0.90,
        "fr_hy": 0.90,
        "hl_hy": 1.15,
        "hr_hy": 1.15,
        "fl_kn": -1.55,
        "fr_kn": -1.55,
        "hl_kn": -1.65,
        "hr_kn": -1.65,
    }
    return {
        "stand": robot.data.default_joint_pos.clone(),
        "high_crouch": joint_target_from_dict(robot, high_crouch),
        "low_crouch": joint_target_from_dict(robot, low_crouch),
        "sit": joint_target_from_dict(robot, SIT_JOINT_TARGETS),
    }


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


def step_target(sim: sim_utils.SimulationContext, robot: Articulation, target: torch.Tensor) -> None:
    for _ in range(DECIMATION):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(SIM_DT)


def blend_target(robot: Articulation, start: torch.Tensor, end: torch.Tensor, alpha: float) -> torch.Tensor:
    blend = smoothstep(alpha)
    return clamp_joint_target(robot, (1.0 - blend) * start + blend * end)


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
        f"[{phase:>11}] step={step:05d} t={sim_time:6.2f}s "
        f"root=({root[0]:+.2f}, {root[1]:+.2f}, {root[2]:+.2f}) "
        f"min_foot_z={min_foot_z:+.4f} foot_z={[round(v, 4) for v in per_foot]}",
        flush=True,
    )


def hold(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    target: torch.Tensor,
    duration: float,
    phase: str,
    step: int,
    next_print_time: float,
    foot_ids: list[int],
) -> tuple[int, float]:
    num_steps = int(round(duration / CONTROL_DT))
    for _ in range(num_steps):
        if args_cli.max_steps is not None and step >= args_cli.max_steps:
            raise SystemExit(0)
        sim_time = step * CONTROL_DT
        step_target(sim, robot, target)
        if sim_time >= next_print_time:
            print_status(phase, step, sim_time, robot, foot_ids)
            next_print_time = sim_time + args_cli.print_interval
        step += 1
    return step, next_print_time


def transition(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    end_target: torch.Tensor,
    duration: float,
    phase: str,
    step: int,
    next_print_time: float,
    foot_ids: list[int],
) -> tuple[int, float]:
    start_target = robot.data.joint_pos.clone()
    num_steps = max(1, int(round(duration / CONTROL_DT)))
    for index in range(num_steps):
        if args_cli.max_steps is not None and step >= args_cli.max_steps:
            raise SystemExit(0)
        alpha = (index + 1) / num_steps
        target = blend_target(robot, start_target, end_target, alpha)
        sim_time = step * CONTROL_DT
        step_target(sim, robot, target)
        if sim_time >= next_print_time:
            print_status(phase, step, sim_time, robot, foot_ids)
            next_print_time = sim_time + args_cli.print_interval
        step += 1
    return step, next_print_time


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=SIM_DT, device=args_cli.device))
    robot = design_scene()
    sim.reset()
    reset_robot(robot, args_cli.root_height)
    robot.update(SIM_DT)
    sim.set_camera_view(eye=(3.0, 3.0, 1.5), target=(0.0, 0.0, 0.35))

    keyframes = primitive_keyframes(robot)
    foot_ids = [index for index, name in enumerate(robot.body_names) if name.endswith("_foot")]
    print(f"[INFO] joints={robot.joint_names}", flush=True)
    print(f"[INFO] feet={[robot.body_names[index] for index in foot_ids]}", flush=True)
    print("[INFO] Primitive phases: stand hold -> high crouch -> low crouch -> sit -> low/high crouch -> stand", flush=True)

    step = 0
    next_print_time = 0.0
    for cycle in range(args_cli.cycles):
        print(f"[INFO] cycle={cycle + 1}/{args_cli.cycles}", flush=True)
        step, next_print_time = hold(
            sim, robot, keyframes["stand"], args_cli.stand_hold_seconds, "stand_hold", step, next_print_time, foot_ids
        )
        step, next_print_time = transition(
            sim, robot, keyframes["high_crouch"], args_cli.sit_down_seconds * 0.35, "to_high", step, next_print_time, foot_ids
        )
        step, next_print_time = transition(
            sim, robot, keyframes["low_crouch"], args_cli.sit_down_seconds * 0.30, "to_low", step, next_print_time, foot_ids
        )
        step, next_print_time = transition(
            sim, robot, keyframes["sit"], args_cli.sit_down_seconds * 0.35, "to_sit", step, next_print_time, foot_ids
        )
        step, next_print_time = hold(
            sim, robot, keyframes["sit"], args_cli.sit_hold_seconds, "sit_hold", step, next_print_time, foot_ids
        )
        step, next_print_time = transition(
            sim, robot, keyframes["low_crouch"], args_cli.stand_up_seconds * 0.35, "up_low", step, next_print_time, foot_ids
        )
        step, next_print_time = transition(
            sim, robot, keyframes["high_crouch"], args_cli.stand_up_seconds * 0.30, "up_high", step, next_print_time, foot_ids
        )
        step, next_print_time = transition(
            sim, robot, keyframes["stand"], args_cli.stand_up_seconds * 0.35, "to_stand", step, next_print_time, foot_ids
        )

    if args_cli.max_steps is not None:
        raise SystemExit(0)

    while simulation_app.is_running():
        step_target(sim, robot, keyframes["stand"])


if __name__ == "__main__":
    main()
    simulation_app.close()
