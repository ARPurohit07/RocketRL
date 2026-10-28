import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from pythia_controller import PythiaController

model = PPO.load("pythia_residual", device="cuda")

eval_env = Monitor(FrameStackObservation(gym.make("LunarLander-v3"), stack_size=8))

mean, std = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=False)

print(f"Mean reward: {mean:.2f}")
print(f"Standard deviation: {std:.2f}")
eval_env.close()

env = FrameStackObservation(
    gym.make("LunarLander-v3", render_mode="human"), stack_size=8
)

for episode in range(3):
    obs, info = env.reset()
    total_reward = 0

    while True:
        action, _ = model.predict(obs, deterministic=False)

        obs, reward, terminated, truncated, info = env.step(int(action))

        total_reward += reward

        if terminated or truncated:
            break

    print(f"Episode {episode + 1}: {total_reward:.2f}")

env.close()
