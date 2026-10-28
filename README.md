# RocketRL 3D

RocketRL trains a Pythia-70M based PPO agent to land a rigid rocket on a circular pad in MuJoCo. The rocket moves and rotates in three dimensions. The policy controls the main engine and two independent steering axes. It learns from environment rewards; no teacher controller or action labels are used.

The project was migrated from a 2D Box2D environment. The 3D policy applies one shared Pythia backbone to separate X and Y views of each observation. It has independent steering outputs and a shared engine decision. `checkpoints/baseline.zip` is the transferred 3D starting point. `checkpoints/improved.zip` is the single promoted PPO checkpoint. Training replaces that file only after a better result on fixed evaluation seeds and adequate compact-task retention. Git LFS tracks the checkpoints and the local Pythia weights.

Training compares full-range checkpoints on 200 development seeds `1000000–1000199`, with smaller near and compact retention checks. The 85% goal remains an acceptance target until a separate 200-seed full challenge evaluation verifies it. A preliminary independent 200-seed run scored 150/200 (75%), so the goal has not yet been met.

## Environment

The MuJoCo scene has a free moving rocket body, four contact feet, a ground plane, and a circular 3 m landing pad. At full challenge, starts vary continuously over X and Y positions up to ±4 m, heights 6–24 m, horizontal drift up to ±1.2 m/s per axis, descent up to 2.8 m/s, and initial tilt and rotation. Vertically and laterally unrecoverable starts are resampled using a simple kinematic feasibility bound. There are no discrete easy/medium/hard modes.

A landing requires at least three feet on the pad, horizontal speed below 0.8 m/s, vertical speed below 0.8 m/s, tilt below 0.20 rad, angular speed below 0.4 rad/s, and 12 consecutive stable physics steps. Contact can settle for at most 150 physics steps before becoming a failure. Hull contact, hard impact, off-pad contact, boundary exits, and unsettled contact are failures.

The 18-value state includes position and velocity expressed in the rocket's yaw frame, vertical position and velocity, attitude, angular velocity, foot contacts, and fuel. Eight states are stacked. The action space is `MultiDiscrete([2, 3, 3])`: engine off/on and independent steering in both horizontal axes.

## Run

Use the project virtual environment. Install dependencies listed in `requirements-3d.txt` if building a new environment.

```powershell
.\.venv\Scripts\python.exe run_tests.py
.\.venv\Scripts\python.exe rocket3d_eval.py --checkpoint checkpoints/baseline.zip --episodes 200 --seed-start 2000000
.\.venv\Scripts\python.exe rocket3d_eval.py --checkpoint checkpoints/improved.zip --episodes 200 --seed-start 2000000
.\.venv\Scripts\python.exe play_rocket.py --checkpoint checkpoints/improved.zip
```

For terminal only playback, add `--headless`. To reproduce an episode, add `--seed N`.

Continue PPO training from the current improved checkpoint:

```powershell
.\.venv\Scripts\python.exe train3d.py --steps 262144 --eval-interval 16384 --seed 20261028
```

For hard starts, `--engine-logit-scale 0.05 --steering-logit-scale 0.5` softens the transferred action logits before PPO rollouts so early engine and steering actions can be sampled. The deterministic action choice is unchanged at initialization. These settings affect the current training run and are stored in the single metrics log.

Training gradually widens a continuous randomized distribution based on recent episode success. Two workers emphasize 6–10 m starts and ramp their minimum offset from 1 to 2.5 m. Two workers emphasize 10–16 m starts and ramp minimum offset from 2 to 3.5 m. The full-range evaluation distribution stays unchanged. The reward includes terminal landing success, motion shaping, and a clearance cost when the projected horizontal position still misses the pad. No target actions or teacher controller are used. `runs/3d_metrics.jsonl` stores evaluation history. The evaluator reports success by start height and distance. The acceptance goal is at least 85% landings over 200 previously unused full challenge seeds; the score is not claimed until that run passes.

For targeted training on lateral control, `--focus-axis y --challenge-start 0.4` draws continuously randomized starts with larger Y than X error. This provides no target actions; the same full challenge evaluator determines whether the resulting checkpoint helps.

## Source control

The Git repository tracks source, tests, baseline and improved checkpoints, and the Pythia weights with Git LFS. The local virtual environment, logs, and metrics are ignored. No remote is configured.

The previous 2D source is preserved in `legacy2d/` for historical reference. Active 3D commands are at the repository root.
