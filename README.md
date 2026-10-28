# RocketRL 3D

RocketRL trains a Pythia-70M based PPO agent to land a rigid rocket on a circular pad in MuJoCo. The rocket moves and rotates in three dimensions. The policy controls the main engine and two independent steering axes. It learns from environment rewards; no teacher controller or action labels are used.

The project was migrated from a 2D Box2D environment. The 3D policy applies one shared Pythia backbone to separate X and Y views of each observation. It has independent steering outputs and a shared engine decision. `checkpoints/baseline.zip` is the transferred 3D starting point. `checkpoints/improved.zip` is the single promoted PPO checkpoint. Training replaces that file only after a better result on fixed evaluation seeds and adequate compact-task retention. Git LFS tracks the checkpoints and the local Pythia weights.

Training compares full-range checkpoints on 200 development seeds `1000000–1000199`, a focused 96-seed low-height offset set, and smaller near and compact retention checks.

## Verified result

Deterministic full-challenge comparison on seeds `5000000–5000399`:

| Checkpoint | All landings | 6–10 m landings | Off-pad failures |
| --- | ---: | ---: | ---: |
| Previous improved (`e23688f`) | 338/400 (84.5%) | 73/97 (75.3%) | 42 |
| Interim improved (`3aae3b8`) | 345/400 (86.25%) | 73/97 (75.3%) | 34 |
| Current improved | **370/400 (92.5%)** | **82/97 (84.5%)** | **16** |

On untouched seeds `6000000–6000399`, the current improved checkpoint landed **373/400 (93.25%)**, including **89/97 (91.8%)** starts at 6–10 m, with 15 off-pad failures. It was selected at training step 131,072 on separate development seeds, scoring 191/200 full challenge and 89/96 on the focused low-height offset benchmark. The original 3D baseline scored 56/200 on historical seeds `4000000–4000199`.

## Environment

The MuJoCo scene has a free moving rocket body, four contact feet, a ground plane, and a circular 3 m landing pad. At full challenge, starts vary continuously over X and Y positions up to ±4 m, heights 6–24 m, horizontal drift up to ±1.2 m/s per axis, descent up to 2.8 m/s, and initial tilt and rotation. Vertically and laterally unrecoverable starts are resampled using a simple kinematic feasibility bound. There are no discrete easy/medium/hard modes.

A landing requires at least three feet on the pad, horizontal speed below 0.8 m/s, vertical speed below 0.8 m/s, tilt below 0.20 rad, angular speed below 0.4 rad/s, and 12 consecutive stable physics steps. Contact can settle for at most 150 physics steps before becoming a failure. Hull contact, hard impact, off-pad contact, boundary exits, and unsettled contact are failures.

The 18-value state includes position and velocity expressed in the rocket's yaw frame, vertical position and velocity, attitude, angular velocity, foot contacts, and fuel. Eight states are stacked. The action space is `MultiDiscrete([2, 3, 3])`: engine off/on and independent steering in both horizontal axes.

## Run

Use the project virtual environment. Install dependencies listed in `requirements-3d.txt` if building a new environment.

```powershell
.\.venv\Scripts\python.exe run_tests.py
.\.venv\Scripts\python.exe rocket3d_eval.py --checkpoint checkpoints/baseline.zip --episodes 200 --seed-start 4000000
.\.venv\Scripts\python.exe rocket3d_eval.py --checkpoint checkpoints/improved.zip --episodes 400 --seed-start 6000000
.\.venv\Scripts\python.exe rocket3d_eval.py --checkpoint checkpoints/improved.zip --episodes 200 --seed-start 6100000 --height-ceiling 10 --min-start-radius 2
.\.venv\Scripts\python.exe play_rocket.py --checkpoint checkpoints/improved.zip
```

The interactive viewer shows the rocket, engine flame, and yellow-bordered landing target in daylight. MuJoCo's control panels are hidden to leave more room for the flight. For terminal only playback, add `--headless`. To reproduce an episode, add `--seed N`.

Continue PPO training from the current improved checkpoint with more low, offset starts:

```powershell
.\.venv\Scripts\python.exe train3d.py --steps 131072 --eval-interval 16384 --challenge-start 1.0 --full-workers 2 --low-workers 3 --mid-workers 3 --low-height-ceiling 10 --low-start-radius 2.0 --low-target-radius 3.0 --mid-height-floor 6 --mid-height-ceiling 10 --mid-start-radius 3.0 --mid-target-radius 4.2 --train-head-only --prioritize-low-height --minimum-full-success 0.88 --learning-rate 0.000007 --entropy-coef 0.001 --gamma 0.99 --gae-lambda 0.95 --seed 20261204
```

The transferred action logits were calibrated once with `--engine-logit-scale 0.05 --steering-logit-scale 0.5` so PPO could sample early thrust and steering. The improved checkpoint already contains that calibration; continuation uses the default scales. Training settings are stored in the single metrics log.

Training gradually widens a continuous randomized distribution based on recent episode success. Worker counts are configurable; the latest run used two full-range workers and six 6–10 m offset workers. The full-range evaluation distribution stays unchanged. The reward includes terminal landing success, motion shaping, and a clearance cost when the projected horizontal position still misses the pad. No target actions or teacher controller are used. `runs/3d_metrics.jsonl` stores evaluation history. The evaluator reports success by start height and distance.

For targeted training on lateral control, `--focus-axis y --challenge-start 0.4` draws continuously randomized starts with larger Y than X error. This provides no target actions; the same full challenge evaluator determines whether the resulting checkpoint helps.

## Source control

The Git repository tracks source, tests, baseline and improved checkpoints, and the Pythia weights with Git LFS. The local virtual environment, logs, and metrics are ignored. No remote is configured.

The previous 2D source is preserved in `legacy2d/` for historical reference. Active 3D commands are at the repository root.
