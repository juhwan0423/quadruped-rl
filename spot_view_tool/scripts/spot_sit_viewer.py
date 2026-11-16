# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Tune a seated Spot pose and viewport angle on flat terrain.

Run from the IsaacLab checkout:

    ./isaaclab.sh -p ../spot_view_tool/scripts/spot_sit_viewer.py --preset sit_low
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any

import yaml

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRESET_FILE = PROJECT_ROOT / "config" / "sit_view_presets.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spot seated-pose and view-angle tuning scene.")
    parser.add_argument("--preset-file", type=Path, default=DEFAULT_PRESET_FILE, help="YAML preset file.")
    parser.add_argument("--preset", default="sit_low", help="Preset name from the preset file.")
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit.")
    parser.add_argument("--print-joints", action="store_true", help="Print loaded Spot joint names after startup.")
    parser.add_argument("--max-steps", type=int, default=None, help="Run a finite number of simulation steps then exit.")
    parser.add_argument("--joint", action="append", default=[], metavar="NAME=RAD", help="Override one joint target.")
    parser.add_argument("--base-height", type=float, default=None, help="Override Spot root height in meters.")
    parser.add_argument("--spot-usd-path", type=Path, default=None, help="Use a local Spot USD instead of Isaac assets.")
    parser.add_argument("--lock-root", action="store_true", help="Keep the root pose fixed while tuning.")
    parser.add_argument("--view-mode", choices=["spot", "orbit"], default=None, help="Override camera mode.")
    parser.add_argument("--eye-offset", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--look-yaw-deg", type=float, default=None)
    parser.add_argument("--look-pitch-deg", type=float, default=None)
    parser.add_argument("--look-distance", type=float, default=None)
    parser.add_argument("--target-offset", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--orbit-yaw-deg", type=float, default=None)
    parser.add_argument("--orbit-pitch-deg", type=float, default=None)
    parser.add_argument("--orbit-distance", type=float, default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def load_presets(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    presets = data.get("presets", {})
    if not isinstance(presets, dict):
        raise ValueError(f"Expected 'presets' mapping in {path}")
    return presets


def parse_joint_overrides(overrides: list[str]) -> dict[str, float]:
    parsed = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Joint override must be NAME=RAD, got: {item}")
        name, raw_value = item.split("=", 1)
        parsed[name.strip()] = float(raw_value)
    return parsed


def merge_camera_config(preset_camera: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    camera = {
        "mode": "spot",
        "eye_offset": [0.35, 0.0, 0.40],
        "look_yaw_deg": 0.0,
        "look_pitch_deg": -10.0,
        "look_distance": 2.0,
        "target_offset": [0.0, 0.0, 0.35],
        "orbit_yaw_deg": 35.0,
        "orbit_pitch_deg": 18.0,
        "orbit_distance": 2.2,
    }
    camera.update(preset_camera or {})
    arg_to_key = {
        "view_mode": "mode",
        "eye_offset": "eye_offset",
        "look_yaw_deg": "look_yaw_deg",
        "look_pitch_deg": "look_pitch_deg",
        "look_distance": "look_distance",
        "target_offset": "target_offset",
        "orbit_yaw_deg": "orbit_yaw_deg",
        "orbit_pitch_deg": "orbit_pitch_deg",
        "orbit_distance": "orbit_distance",
    }
    for arg_name, key in arg_to_key.items():
        value = getattr(args, arg_name)
        if value is not None:
            camera[key] = list(value) if isinstance(value, list) else value
    return camera


def degrees_to_unit_vector(yaw_deg: float, pitch_deg: float) -> tuple[float, float, float]:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    horizontal = math.cos(pitch)
    return horizontal * math.cos(yaw), horizontal * math.sin(yaw), math.sin(pitch)


args_cli = parse_args()
presets = load_presets(args_cli.preset_file)

if args_cli.list_presets:
    print("Available presets:")
    for name in sorted(presets):
        print(f"  {name}")
    raise SystemExit(0)

if args_cli.preset not in presets:
    available = ", ".join(sorted(presets))
    raise ValueError(f"Unknown preset '{args_cli.preset}'. Available presets: {available}")

if args_cli.spot_usd_path is not None and not args_cli.spot_usd_path.expanduser().exists():
    raise FileNotFoundError(f"Local Spot USD does not exist: {args_cli.spot_usd_path}")

preset = presets[args_cli.preset]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_assets.robots.spot import SPOT_CFG


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

    sim_utils.create_prim("/World/SpotOrigin", "Xform", translation=(0.0, 0.0, 0.0))
    spot_cfg = copy.deepcopy(SPOT_CFG).replace(prim_path="/World/SpotOrigin/Robot")
    if args_cli.spot_usd_path is not None:
        spot_cfg.spawn.usd_path = str(args_cli.spot_usd_path.expanduser().resolve())
    return Articulation(spot_cfg)


def build_joint_target(robot: Articulation, joint_targets: dict[str, float]) -> torch.Tensor:
    target = robot.data.default_joint_pos.clone()
    unknown_names = sorted(set(joint_targets) - set(robot.joint_names))
    if unknown_names:
        available = ", ".join(robot.joint_names)
        raise ValueError(f"Unknown Spot joint(s): {unknown_names}. Available joints: {available}")

    for name, value in joint_targets.items():
        target[:, robot.joint_names.index(name)] = value
    return target


def reset_robot(robot: Articulation, joint_target: torch.Tensor, base_height: float) -> torch.Tensor:
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] = torch.tensor((0.0, 0.0, base_height), device=robot.device)
    root_state[:, 7:] = 0.0

    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_target, joint_vel)
    robot.set_joint_position_target(joint_target)
    robot.reset()
    return root_state


def update_viewport_camera(sim: sim_utils.SimulationContext, robot: Articulation, camera: dict[str, Any]) -> None:
    base_pos = robot.data.root_pos_w[0].detach().cpu().tolist()

    if camera["mode"] == "orbit":
        target_offset = camera["target_offset"]
        target = [base_pos[i] + target_offset[i] for i in range(3)]
        direction = degrees_to_unit_vector(camera["orbit_yaw_deg"], camera["orbit_pitch_deg"])
        distance = camera["orbit_distance"]
        eye = [target[i] + direction[i] * distance for i in range(3)]
    else:
        eye_offset = camera["eye_offset"]
        eye = [base_pos[i] + eye_offset[i] for i in range(3)]
        direction = degrees_to_unit_vector(camera["look_yaw_deg"], camera["look_pitch_deg"])
        distance = camera["look_distance"]
        target = [eye[i] + direction[i] * distance for i in range(3)]

    sim.set_camera_view(eye=eye, target=target)


def main() -> None:
    joint_targets = dict(preset.get("joints", {}))
    joint_targets.update(parse_joint_overrides(args_cli.joint))
    base_height = float(args_cli.base_height if args_cli.base_height is not None else preset.get("base_height", 0.34))
    camera = merge_camera_config(preset.get("camera", {}), args_cli)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    spot = design_scene()
    sim.reset()

    joint_target = build_joint_target(spot, joint_targets)
    locked_root_state = reset_robot(spot, joint_target, base_height)
    update_viewport_camera(sim, spot, camera)

    if args_cli.print_joints:
        print("[INFO]: Spot joints:")
        for joint_name in spot.joint_names:
            print(f"  {joint_name}")

    print(f"[INFO]: Preset: {args_cli.preset}")
    print(f"[INFO]: Base height: {base_height:.3f} m")
    print(f"[INFO]: Camera: {camera}")
    print("[INFO]: Setup complete.")

    sim_dt = sim.get_physics_dt()
    count = 0
    while simulation_app.is_running():
        if args_cli.max_steps is not None and count >= args_cli.max_steps:
            break

        if args_cli.lock_root:
            spot.write_root_pose_to_sim(locked_root_state[:, :7])
            spot.write_root_velocity_to_sim(locked_root_state[:, 7:])

        spot.set_joint_position_target(joint_target)
        spot.write_data_to_sim()
        sim.step()
        spot.update(sim_dt)
        update_viewport_camera(sim, spot, camera)
        count += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
