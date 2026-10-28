# RocketRL 3D

RocketRL trains a Pythia-70M based PPO agent to land a rigid rocket on a circular pad in MuJoCo. The rocket moves and rotates in three dimensions. The policy controls the main engine and two independent steering axes. It learns from environment rewards; no teacher controller or action labels are used.

The project is being migrated from a 2D Box2D environment. The 3D policy retains the Pythia backbone, but its input and action heads have been adapted to the 3D observation and control spaces. `checkpoints/baseline.zip` is the transferred 3D starting point. `checkpoints/improved.zip` is the single promoted PPO checkpoint. Training replaces that file only after a better result on fixed evaluation seeds and adequate compact-task retention. Git LFS tracks the checkpoints and the local Pythia weights.

## Environment

The MuJoCo scene has a free moving rocket body, four contact feet, a ground plane, and a circular 3 m landing pad. At full challenge, starts vary continuously over X and Y positions up to ±4 m, heights 6–24 m, horizontal drift up to ±1.2 m/s per axis, descent up to 2.8 m/s, and initial tilt and rotation. Vertically and laterally unrecoverable starts are resampled using a simple kinematic feasibility bound. There are no discrete easy/medium/hard modes.

A landing requires at least three feet on the pad, horizontal speed below 0.8 m/s, vertical speed below 0.8 m/s, tilt below 0.20 rad, angular speed below 0.4 rad/s, and 12 consecutive stable physics steps. Hull contact, hard impact, off-pad contact, boundary exits, and unsettled contact are failures.

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

Training gradually widens a single continuous randomized distribution based on recent episode success. `runs/3d_metrics.jsonl` stores evaluation history. The acceptance goal is at least 85% landings over 200 previously unused full challenge seeds; the score is not claimed until that run passes.

## Source control

The Git repository tracks source, tests, baseline and improved checkpoints, and the Pythia weights with Git LFS. The local virtual environment, logs, and metrics are ignored. No remote is configured.
