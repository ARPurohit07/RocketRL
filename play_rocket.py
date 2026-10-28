"""Watch the learned 3D rocket land in MuJoCo's interactive viewer."""

import argparse
import time
from pathlib import Path

from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import PPO

from rocket3d_env import Rocket3DEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/improved.zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"No trained 3D checkpoint: {args.checkpoint}")
    model = PPO.load(args.checkpoint, device=args.device)
    env = FrameStackObservation(Rocket3DEnv(challenge=1.0), 8)
    viewer = None
    if not args.headless:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.unwrapped.physics.model,
                                              env.unwrapped.physics.data)
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -25
        viewer.cam.distance = 24
        viewer.cam.lookat[:] = (0, 0, 6)
    try:
        for episode in range(args.episodes):
            if viewer is not None and not viewer.is_running():
                break
            seed = None if args.seed is None else args.seed + episode
            obs, _ = env.reset(seed=seed)
            total_reward = 0.0
            if viewer is not None:
                viewer.sync()
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if viewer is not None:
                    x, y, z = info["position"]
                    viewer.cam.lookat[:] = (0.25 * x, 0.25 * y, max(3.0, 0.5 * z))
                    viewer.cam.distance = max(18.0, 1.6 * z)
                    viewer.sync()
                    time.sleep(env.unwrapped.control_repeat *
                               env.unwrapped.physics.DT)
                    if not viewer.is_running():
                        return
                if terminated or truncated:
                    print(
                        f"Episode {episode + 1}: {info['status']} "
                        f"reward={total_reward:+.2f} reason={info['reason']} "
                        f"xyz={[round(v, 2) for v in info['position']]}",
                        flush=True,
                    )
                    if viewer is not None:
                        time.sleep(0.7)
                    break
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
