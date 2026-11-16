# Spot View Tool

Small IsaacLab starter project for tuning a seated Boston Dynamics Spot pose and the viewport angle on flat terrain.

## Run

From the IsaacLab checkout:

```bash
cd $ISAACLAB_PATH
./isaaclab.sh -p $REPO/spot_view_tool/scripts/spot_sit_viewer.py --preset sit_low
```

Useful variants:

```bash
./isaaclab.sh -p $REPO/spot_view_tool/scripts/spot_sit_viewer.py --list-presets
./isaaclab.sh -p $REPO/spot_view_tool/scripts/spot_sit_viewer.py --preset sit_orbit_check
./isaaclab.sh -p $REPO/spot_view_tool/scripts/spot_sit_viewer.py --preset sit_low --print-joints
./isaaclab.sh -p $REPO/spot_view_tool/scripts/spot_sit_viewer.py --preset sit_low --look-pitch-deg -18 --look-yaw-deg 5
./isaaclab.sh -p $REPO/spot_view_tool/scripts/spot_sit_viewer.py --preset sit_low --joint hl_kn=-2.35 --joint hr_kn=-2.35
./isaaclab.sh -p $REPO/spot_view_tool/scripts/spot_sit_viewer.py --preset sit_low --spot-usd-path /path/to/spot.usd
```

## Current Scope

- Spawns one Spot from `isaaclab_assets.robots.spot.SPOT_CFG`.
- Uses a flat `GroundPlaneCfg`.
- Applies named joint presets from `config/sit_view_presets.yaml`.
- Uses Isaac's Spot asset by default. In offline environments, pass `--spot-usd-path` to a local `spot.usd`.
- Supports two viewport modes:
  - `spot`: camera eye follows a body-relative offset and looks along yaw/pitch.
  - `orbit`: camera orbits around the seated Spot for checking the pose.

This is intentionally script-first. Once the pose and angle controls feel right, the next step is to add an actual IsaacLab `CameraCfg` sensor and save RGB/depth frames from the tuned angle.

## Train MVP Sit Policy

The external ManagerBased RL task is registered without modifying IsaacLab source:

```text
Isaac-Spot-Sit-View-V0
Isaac-Spot-Sit-View-Play-V0
```

Sanity train:

```bash
cd $ISAACLAB_PATH
# activate your Isaac Lab Python environment
./isaaclab.sh -p $REPO/spot_view_tool/scripts/train_sit_view.py --headless --device cpu --num_envs 2 --max_iterations 1
```

Longer run:

```bash
./isaaclab.sh -p $REPO/spot_view_tool/scripts/train_sit_view.py --headless --device cuda:0 --num_envs 512 --max_iterations 300
```

Checkpoints are stored separately from the walking policy:

```text
IsaacLab/logs/rsl_rl/spot_sit_view/<run>/model_*.pt
```

Play a checkpoint:

```bash
./isaaclab.sh -p $REPO/spot_view_tool/scripts/play_sit_view.py --device cuda:0 --num_envs 16 \
  --checkpoint logs/rsl_rl/spot_sit_view/<run>/model_0.pt
```

Switch one spawned Spot through walk -> sit -> stand -> walk:

```bash
./isaaclab.sh -p $REPO/spot_view_tool/scripts/walk_then_sit_policy_switch.py --device cuda:0
```

This script uses the local tool policy bundle by default:

```text
policies/spot_policy.pt
policies/spot_env.yaml
policies/spot_posture_pitch_fast_2026-06-23_16-47-16_fast_mvp_model_4999.pt
policies/spot_posture_pitch_fast_policy.pt
```

The final file is a TorchScript inference export of the fast posture checkpoint, so the runtime can switch directly between Isaac's walking TorchScript policy and the posture TorchScript policy. The default sit/stand timing is:

```bash
--stand-seconds 0.2 \
--to-sit-seconds 0.8 \
--sit-seconds 0.5 \
--to-stand-seconds 0.8 \
--final-stand-seconds 0.5
```

Run with the defaults:

```bash
cd $ISAACLAB_PATH
# activate your Isaac Lab Python environment
./isaaclab.sh -p $REPO/spot_view_tool/scripts/walk_then_sit_policy_switch.py --device cuda:0
```

Run the first safety behavior, a fast 0.5s backward walk using the same Isaac Spot walking policy:

```bash
cd $ISAACLAB_PATH
# activate your Isaac Lab Python environment
./isaaclab.sh -p $REPO/spot_view_tool/scripts/safety_fast_back_policy.py --device cuda:0
```

Useful variants:

```bash
./isaaclab.sh -p $REPO/spot_view_tool/scripts/safety_fast_back_policy.py --device cuda:0 --back-speed 1.5
./isaaclab.sh -p $REPO/spot_view_tool/scripts/safety_fast_back_policy.py --device cuda:0 --back-seconds 0.5 --settle-seconds 0.8
```

Run scripted sit/stand without RL:

```bash
./isaaclab.sh -p $REPO/spot_view_tool/scripts/sit_stand_primitive.py --device cuda:0
```

The MVP reward terms are:

- sit joint pose tracking
- base height tracking
- upright body orientation
- feet contact
- base stability, action-rate, joint-velocity, and joint-torque penalties

## Train Posture/View Policy

This variant learns sit, view-angle adjustment, and stand-up as one command-conditioned policy:

```text
Isaac-Spot-Posture-View-V0
Isaac-Spot-Posture-View-Play-V0
```

The command is `[posture, roll, pitch, yaw]`, but the current training setup uses pitch only:


- `posture=0`: seated view pose
- `posture=1`: standing pose
- `roll=0`, `yaw=0`
- `pitch`: sampled from `-0.45` to `0.15` rad while seated
- intermediate posture values follow the sit/crouch/stand primitive keyframe chain

Sanity train:

```bash
cd $ISAACLAB_PATH
# activate your Isaac Lab Python environment
./isaaclab.sh -p $REPO/spot_view_tool/scripts/train_posture_view.py --headless --device cpu --num_envs 2 --max_iterations 1
```

Longer run:

```bash
./isaaclab.sh -p $REPO/spot_view_tool/scripts/train_posture_view.py --headless --device cuda:0 --num_envs 512 --max_iterations 5000 --run_name pitch_mvp
```

Checkpoints are stored separately:

```text
IsaacLab/logs/rsl_rl/spot_posture_pitch/<run>/model_*.pt
```

Play a checkpoint:

```bash
./isaaclab.sh -p $REPO/spot_view_tool/scripts/play_posture_view.py --device cuda:0 --num_envs 16 \
  --checkpoint logs/rsl_rl/spot_posture_pitch/<run>/model_0.pt
```

The default RSL-RL play script samples random commands. For a readable quick stand -> sit/view -> stand check, use:

```bash
./isaaclab.sh -p $REPO/spot_view_tool/scripts/play_posture_view_sequence.py \
  --device cuda:0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/spot_posture_pitch/<run>/model_4999.pt \
  --pitch -0.36 \
  --real-time
```

By default this sequence stands for 0.5s, sits down over 1.5s, holds the seated pitch for 2s, then stands up over 1.5s.
Override the timing if needed:

```bash
./isaaclab.sh -p $REPO/spot_view_tool/scripts/play_posture_view_sequence.py \
  --device cuda:0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/spot_posture_pitch/<run>/model_4999.pt \
  --pitch -0.36 \
  --to-sit-seconds 1.0 \
  --sit-seconds 2.0 \
  --to-stand-seconds 1.0 \
  --real-time
```

## Train Fast Posture Policy

This variant keeps the pitch-only command setup but trains on faster command changes:

```text
Isaac-Spot-Posture-Fast-V0
Isaac-Spot-Posture-Fast-Play-V0
```

Train:

```bash
cd $ISAACLAB_PATH
# activate your Isaac Lab Python environment
./isaaclab.sh -p $REPO/spot_view_tool/scripts/train_posture_fast.py \
  --headless \
  --device cuda:0 \
  --num_envs 4096 \
  --max_iterations 5000 \
  --run_name fast_mvp
```

Checkpoints are stored separately:

```text
IsaacLab/logs/rsl_rl/spot_posture_pitch_fast/<run>/model_*.pt
```

Evaluate with a fast stand -> sit -> stand sequence:

```bash
./isaaclab.sh -p $REPO/spot_view_tool/scripts/play_posture_view_sequence.py \
  --device cuda:0 \
  --num_envs 1 \
  --task Isaac-Spot-Posture-Fast-Play-V0 \
  --checkpoint logs/rsl_rl/spot_posture_pitch_fast/2026-06-23_16-47-16_fast_mvp/model_4999.pt \
  --pitch -0.36 \
  --stand-seconds 0.2 \
  --to-sit-seconds 0.8 \
  --sit-seconds 0.5 \
  --to-stand-seconds 0.8 \
  --final-stand-seconds 0.5 \
  --real-time
```
