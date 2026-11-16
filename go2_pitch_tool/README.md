# Go2 Pitch Tool

Unitree Go2 flat-terrain locomotion with an extra **body-pitch command**, registered as an
external IsaacLab task. Nothing under `IsaacLab/` is modified — the task inherits
`Isaac-Velocity-Flat-Unitree-Go2-v0` and the training loop is IsaacLab's own RSL-RL script,
invoked through a thin wrapper in `scripts/`.

## Registered tasks

```text
Isaac-Velocity-Flat-Go2-Pitch-v0
Isaac-Velocity-Flat-Go2-Pitch-Play-v0
```

## Run

```bash
cd $ISAACLAB_PATH
# activate your Isaac Lab Python environment

# sanity check (a few seconds)
./isaaclab.sh -p $REPO/go2_pitch_tool/scripts/train_go2_pitch.py \
    --headless --device cpu --num_envs 2 --max_iterations 1

# real run
./isaaclab.sh -p $REPO/go2_pitch_tool/scripts/train_go2_pitch.py \
    --headless --device cuda:0 --num_envs 4096 --max_iterations 1000

# play a checkpoint
./isaaclab.sh -p $REPO/go2_pitch_tool/scripts/play_go2_pitch.py \
    --device cuda:0 --num_envs 16 \
    --checkpoint logs/rsl_rl/go2_pitch/<run>/model_<n>.pt
```

Checkpoints land in `IsaacLab/logs/rsl_rl/go2_pitch/<run>/`, separate from the stock
`unitree_go2_flat` runs.

## What changed vs. Go2 Flat v0

**Command** — `base_velocity` (vx, vy, wz) is untouched; a separate 1-dim `body_pitch`
command is added. Keeping it as its own term means the velocity term's heading control and
standing-environment logic are reused as-is. It resamples every 3–5 s (velocity resamples
every 10 s) so the policy has to learn pitch *transitions*, and 20 % of resamples are forced
to 0 rad so a clean level posture stays in the behaviour set.

**Observation** — 48 → 49 dims. `pitch_command` sits directly after `velocity_commands`:

```text
0  base_lin_vel        (3)
3  base_ang_vel        (3)
6  projected_gravity   (3)
9  velocity_commands   (3)
12 pitch_command       (1)
13 joint_pos          (12)
25 joint_vel          (12)
37 actions            (12)
```

**Reward** — the stock flat task pins the body level with `flat_orientation_l2 = -2.5`,
which fights the pitch command directly, so it is zeroed and replaced by:

- `track_body_pitch_exp` (+1.0) — exponential kernel on the body-frame gravity vector.
  For pitch `p` the target is `(sin p, ·, -cos p)`, which avoids euler wrapping and is only
  weakly coupled to roll.
- `body_roll_l2` (-2.5) — keeps the robot upright sideways without touching pitch.
- `dof_pos_limits` raised from 0 to -0.1, so the commanded pitch is not reached by bottoming
  out the knees.

**Curriculum** — `pitch_levels` ramps the command range from ±0.15 rad to ±0.4 rad over the
first 4000 policy steps (≈ 170 iterations at `num_steps_per_env = 24`). Walking and leaning
at once is much harder than either alone; the current half-range is logged as
`Curriculum/pitch_levels`. The PLAY config disables the ramp and runs the full ±0.4 rad.

`base_velocity.ranges.ang_vel_z` is narrowed to ±0.8 rad/s to keep the early signal clean.

## Things to watch while tuning

- `Metrics/body_pitch/error_pitch` is the honest measure of pitch tracking — the reward term
  saturates, this one does not.
- If gait quality collapses when the curriculum widens, slow it down (raise `num_steps`) or
  lower `end_range` before touching PPO hyperparameters.
- `track_lin_vel_xy_exp` measures velocity in the **body** frame, so a pitched body tracks a
  slightly tilted forward axis. Harmless at ±0.4 rad; if you go much beyond that, switch the
  term to a gravity-aligned horizontal frame.
- `ang_vel_xy_l2` (-0.05, inherited) penalizes pitch *rate*, which mildly damps transitions.
  Lower it if pitch responses feel sluggish.
