# RocketRL

RocketRL is a pure reinforcement-learning rocket-landing project built with Stable-Baselines3 PPO, Box2D, and a Pythia-70M feature backbone with LoRA adapters. It uses no teacher controller, imitation data, expert actions, or action labels.

The policy has four discrete actions: coast, main engine plus left steering, main engine, and main engine plus right steering. The architecture remains Pythia-70M + LoRA throughout training.

## Current model

`rocket_pythia_phase2_best.zip` is the protected evaluation-selected checkpoint used by playback. It is only replaced when a candidate improves the fixed full-randomization held-out suite, does not regress lateral or combined platform-departure failures, and retains at least 95% success on the compact suite. Playback uses the same learned policy in a canonical orientation, with a narrow initial-state safety gate for low-altitude outward-moving starts.

The current checkpoint was selected from a 65,536-step, full-challenge PPO continuation with no teacher or architecture change. On the independent 100-seed audit (`970000` through `970099`), it improves landing success from 55% to 59% versus the preserved pre-training checkpoint. Off-pad contacts remain at 7, lateral exits remain at 8, and unsettled platform contacts fall from 15 to 8. Mean reward rises from +25.09 to +36.51.

The environment samples one continuous uniform distribution—there are no easy, medium, or hard modes. At full challenge it covers:

- Horizontal position: -8 to +8 m
- Altitude: 4 to 28 m
- Horizontal velocity: -3 to +3 m/s
- Vertical velocity: -4 to +0.75 m/s
- Angle: -0.30 to +0.30 rad
- Angular velocity: -0.20 to +0.20 rad/s

Physically unrecoverable vertical starts are rejected and resampled.

## Install and validate

Use the project virtual environment:

```powershell
.\.venv\Scripts\python.exe run_tests.py
```

The test suite covers gravity, thrust, steering authority, braking, collision and touchdown classification, deterministic seeding, observation bounds, reward direction, and Pythia/LoRA gradient flow.

## Evaluate

Run the reproducible 64-seed full-challenge deployment benchmark:

```powershell
.\.venv\Scripts\python.exe rocket_eval.py --checkpoint rocket_pythia_phase2_best.zip
```

Evaluate the deployment policy on the independent 128-seed audit:

```powershell
.\.venv\Scripts\python.exe rocket_eval.py `
  --checkpoint rocket_pythia_phase2_best.zip `
  --episodes 128 `
  --seed-start 910000 `
  --canonical-initial-side positive
```

Map performance across the continuous difficulty range:

```powershell
.\.venv\Scripts\python.exe analyze_challenge_frontier.py --episodes 32
```

Evaluation reports success rate, mean/max reward, action use, policy entropy, failure categories, and success by altitude band. A positive episode reward alone is not counted as a landing.

The evaluator uses the same gated symmetry inference as playback by default. Add `--raw-policy` for an uncorrected checkpoint comparison.

## Train

Start a guarded adaptive PPO stage from the protected checkpoint:

```powershell
.\.venv\Scripts\python.exe train_phase2.py `
  --checkpoint rocket_pythia_phase2_best.zip `
  --steps 131072 `
  --adaptive-curriculum `
  --challenge-start 0.925 `
  --challenge-end 1.0 `
  --curriculum-target 0.35 `
  --learning-rate 0.00001 `
  --n-epochs 4 `
  --clip-range 0.04 `
  --logit-scale 1
```

The adaptive curriculum changes only the breadth of randomized initial conditions, based on recent PPO episode success. It does not provide desired actions. Exact checkpoints are evaluated every 8,192 steps; rejected runs cannot overwrite the protected best model.

For left/right-balanced pure-RL training, add `--antithetic-starts`. Each continuously randomized start is followed by its mirrored state, preventing finite PPO rollouts from learning a directional bias without adding teacher actions or labels.

Run `--smoke` for a short pipeline check. Use `--fresh` only to initialize the same existing Pythia-70M + LoRA architecture from scratch.

TensorBoard logs are written under `logs/`, and per-run evaluation history/checkpoints are written under `runs/phase2/randomized/`.

## Play

```powershell
.\.venv\Scripts\python.exe play_rocket.py
```

Useful options:

```powershell
.\.venv\Scripts\python.exe play_rocket.py --episodes 20 --seed 900000 --device auto
```

Playback loads `rocket_pythia_phase2_best.zip` by default, runs the full randomized distribution, and displays trajectory, pose, velocities, fuel, contacts, status, and failure reason at the environment's 25 Hz control rate. Symmetry-consistent inference is enabled by default; pass `--raw-policy` only to compare the uncorrected checkpoint behavior. No second model, teacher, or action labels are used.

## Environment and reward

Box2D integrates at 50 Hz while each policy command is held for two physics frames (25 Hz control). Success requires both legs on the pad with bounded impact speed, tilt, and angular velocity, followed by a stable contact period. Wrong-pad or invalid ground contact terminates instead of remaining stuck and accumulating reward.

The reward combines small time/fuel costs with potential-based shaping for projected pad error, lateral speed, attitude, rotation, safe descent, and braking distance. Terminal landing, crash, timeout, and out-of-bounds outcomes dominate shaping. All policy updates come exclusively from PPO rewards.
