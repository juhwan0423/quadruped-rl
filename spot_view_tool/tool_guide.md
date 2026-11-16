# Spot Tool Guide

This guide describes the files needed to move the completed Spot tools into another IsaacLab/Isaac Sim map and how to run them.

## Completed Tools

- `scripts/walk_then_sit_policy_switch.py`
  - Runs Isaac's Spot walking policy.
  - Stops, switches to the fast posture policy, performs stand -> sit/view -> stand.
  - Returns to the walking policy.
- `scripts/safety_fast_back_policy.py`
  - Runs Isaac's Spot walking policy with a fast backward command.
  - Default behavior is 0.5s backward at `vx=-2.0`, then zero-command hold.

## Required Runtime Files

For the completed tools, copy these directories together:

```text
spot_view_tool/
policies/
```

Minimum useful contents:

```text
spot_view_tool/
  pyproject.toml
  scripts/
    walk_then_sit_policy_switch.py
    safety_fast_back_policy.py
  spot_view_tool/
    __init__.py
    tasks/
      __init__.py
      manager_based/
        __init__.py
        spot_sit_view/
          __init__.py
          mdp/
            __init__.py
            events.py

policies/
  spot_policy.pt
  spot_posture_pitch_fast_policy.pt
```

Recommended full policy bundle:

```text
policies/
  spot_policy.pt
  spot_env.yaml
  spot_posture_pitch_fast_2026-06-23_16-47-16_fast_mvp_model_4999.pt
  spot_posture_pitch_fast_policy.pt
```

Notes:

- `spot_policy.pt` is Isaac's TorchScript Spot locomotion policy.
- `spot_env.yaml` is not required by the current standalone tool scripts, but keep it with the bundle as policy provenance/config.
- `spot_posture_pitch_fast_policy.pt` is the TorchScript inference export used at runtime.
- `spot_posture_pitch_fast_..._model_4999.pt` is the original RSL-RL checkpoint. It is not loaded by the completed runtime script, but keep it so the export can be regenerated later.
- Do not copy `__pycache__/` directories.

## Environment Requirements

The receiving workspace needs:

- IsaacLab available and runnable with `isaaclab.sh`.
- Isaac Sim / IsaacLab Python environment activated.
- `isaaclab_assets.robots.spot.SPOT_CFG` available.
- Isaac's Spot USD asset available through the normal Isaac asset root, or cached locally by IsaacLab.

The scripts assume this layout by default:

```text
ISAAC_WS/
  IsaacLab/
  spot_view_tool/
  policies/
```

If the policy files live somewhere else, pass explicit paths:

```bash
--walk-policy /path/to/spot_policy.pt
--posture-policy /path/to/spot_posture_pitch_fast_policy.pt
```

## Run In A Fresh Workspace

From the IsaacLab checkout:

```bash
cd /path/to/ISAAC_WS/IsaacLab
# activate your Isaac Lab Python environment
```

Walk -> sit/view -> stand -> walk:

```bash
./isaaclab.sh -p $REPO/spot_view_tool/scripts/walk_then_sit_policy_switch.py --device cuda:0
```

Fast backward safety behavior:

```bash
./isaaclab.sh -p $REPO/spot_view_tool/scripts/safety_fast_back_policy.py --device cuda:0
```

Headless smoke checks:

```bash
./isaaclab.sh -p $REPO/spot_view_tool/scripts/walk_then_sit_policy_switch.py \
  --headless \
  --device cuda:0 \
  --max-steps 10

./isaaclab.sh -p $REPO/spot_view_tool/scripts/safety_fast_back_policy.py \
  --headless \
  --device cuda:0 \
  --max-steps 10
```

## Current Default Commands

Walking command:

```text
vx=+0.6, vy=0.0, wz=0.0
```

Safety backward command:

```text
vx=-2.0, vy=0.0, wz=0.0, duration=0.5s
```

Posture sit/view sequence:

```text
stand=0.2s
to_sit=0.8s
sit=0.5s
to_stand=0.8s
final_stand=0.5s
view_command=(roll=0.0, pitch=-0.36, yaw=0.0)
```

## Putting The Tools Into Another Map

The two completed scripts currently create their own simple flat ground and spawn one Spot in `design_scene()`.

To use a different map:

1. Keep policy loading and the control loop unchanged.
2. Replace the ground creation inside `design_scene()` with the map/stage setup you want.
3. Spawn the Spot articulation at the map-specific prim path.
4. Keep the returned object as an `isaaclab.assets.Articulation`.
5. Make sure the Spot joint names remain:

```text
fl_hx fr_hx hl_hx hr_hx
fl_hy fr_hy hl_hy hr_hy
fl_kn fr_kn hl_kn hr_kn
```

The walking and posture policies depend on those joint names/order and on the Spot morphology from `SPOT_CFG`.

## Common Integration Hooks

Useful arguments:

```bash
--root-height 0.50
--policy-device cpu
--policy-device cuda:0
--walk-command VX VY WZ
--return-walk-command VX VY WZ
--view-command ROLL PITCH YAW
--back-speed 2.0
--back-seconds 0.5
--settle-seconds 0.5
```

If the robot falls after a behavior completes, keep a zero-command policy hold running. The safety tool already does this after the backward phase.

## Regenerating The Fast Posture TorchScript

The runtime uses:

```text
policies/spot_posture_pitch_fast_policy.pt
```

This was exported from:

```text
policies/spot_posture_pitch_fast_2026-06-23_16-47-16_fast_mvp_model_4999.pt
```

If the exported file is missing, regenerate it by loading the checkpoint's `actor_state_dict` into a 49 -> 256 -> 256 -> 12 ELU MLP and tracing it with an input tensor of shape `(1, 49)`.

## Files Not Needed For Runtime

These are useful for training or pose tuning, but not required for the completed runtime tools:

```text
scripts/train_*.py
scripts/play_posture_*.py
scripts/play_sit_view.py
scripts/spot_sit_viewer.py
scripts/sit_stand_primitive.py
config/sit_view_presets.yaml
spot_view_tool/tasks/manager_based/spot_sit_view/env_cfg.py
spot_view_tool/tasks/manager_based/spot_sit_view/agents/
spot_view_tool/tasks/manager_based/spot_sit_view/mdp/actions.py
spot_view_tool/tasks/manager_based/spot_sit_view/mdp/commands.py
spot_view_tool/tasks/manager_based/spot_sit_view/mdp/rewards.py
```

Keep the full project if you also want to retrain, evaluate checkpoints through RSL-RL, or tune seated camera poses.
