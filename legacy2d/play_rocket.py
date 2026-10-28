"""Interactive playback using the exact training environment and checkpoint."""
import argparse
import numpy as np
import pygame
import torch
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import PPO
from rocket_env import RocketLandingEnv
from rocket_eval import initial_canonical_mask, policy_probabilities

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="rocket_pythia_phase2_best.zip")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--device", default="auto")
parser.add_argument(
    "--raw-policy", action="store_true",
    help="disable symmetry-consistent inference for comparison",
)
args = parser.parse_args()
WIDTH, HEIGHT, SCALE, GROUND_Y = 1100, 720, 18, 650
pygame.init(); screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pythia-70M | Rocket Landing"); clock = pygame.time.Clock()
font = pygame.font.Font(None, 25)
env = FrameStackObservation(RocketLandingEnv(), 8)
PLAYBACK_MODEL = args.checkpoint
model = PPO.load(PLAYBACK_MODEL, device=args.device)
if model.num_timesteps < 2048:
    raise RuntimeError(
        f"Loaded checkpoint has only {model.num_timesteps} timesteps. "
        "Run sustained training first; the smoke checkpoint is not a landing policy."
    )
trajectory = []

def screen_pos(point):
    return (int(WIDTH / 2 + point[0] * SCALE), int(GROUND_Y - point[1] * SCALE))

def render(action, episode, total_reward, info):
    screen.fill((12, 17, 30)); pygame.draw.rect(screen, (70, 75, 85), (0, GROUND_Y, WIDTH, HEIGHT-GROUND_Y))
    pad_left = screen_pos((-4, 0))[0]; pygame.draw.rect(screen, (50, 210, 120), (pad_left, GROUND_Y-5, 8*SCALE, 5))
    rocket = env.unwrapped.physics.rocket; trajectory.append((rocket.position.x, rocket.position.y))
    if len(trajectory) > 1: pygame.draw.lines(screen, (100, 170, 255), False, [screen_pos(p) for p in trajectory[-300:]], 2)
    for fixture in rocket.fixtures:
        vertices = [screen_pos(rocket.GetWorldPoint(v)) for v in fixture.shape.vertices]
        pygame.draw.polygon(screen, (210,215,225) if fixture.userData == "hull" else (255,185,75), vertices)
    # All powered actions fire the main engine; actions 1 and 3 also gimbal
    # thrust/produce steering torque.
    if action in (1, 2, 3):
        flame = [screen_pos(rocket.GetWorldPoint(v)) for v in [(0,-1.05),(-.2,-1.2),(0,-2.3),(.2,-1.2)]]
        pygame.draw.polygon(screen, (255,110,30), flame)
    if action in (1, 3):
        side = -1 if action == 1 else 1
        flame = [screen_pos(rocket.GetWorldPoint(v)) for v in [(side*.9,-.7),(side*1.1,-.7),(side*1.7,-.7)]]
        pygame.draw.polygon(screen, (255,150,40), flame)
    p = env.unwrapped.physics
    start = info.get("start", {})
    policy_mode = "raw" if args.raw_policy else "gated symmetry"
    lines = [f"Episode {episode}  randomized  Policy {policy_mode}  Action {action}  Reward {total_reward:.1f}",
             f"Start x={start.get('x', 0):.2f} y={start.get('y', 0):.2f} vx={start.get('vx', 0):.2f} vy={start.get('vy', 0):.2f} angle={start.get('angle', 0):.2f}",
             f"x={rocket.position.x:.2f} m  y={rocket.position.y:.2f} m  angle={rocket.angle:.2f} rad",
             f"vx={rocket.linearVelocity.x:.2f}  vy={rocket.linearVelocity.y:.2f}  omega={rocket.angularVelocity:.2f}",
             f"Fuel={p.fuel:.2f} kg  Legs L/R={p.contacts.left}/{p.contacts.right}",
             f"Landing error={info.get('landing_error', abs(rocket.position.x)):.2f} m",
             f"Status={info.get('status', p.status)}  Reason={info.get('crash_reason') or '-'}"]
    for i, line in enumerate(lines): screen.blit(font.render(line, True, (235,235,235)), (18, 18+i*28))
    pygame.display.flip()

running = True
control_hz = env.unwrapped.physics.FPS / env.unwrapped.control_repeat
for episode in range(1, args.episodes + 1):
    if not running: break
    episode_seed = None if args.seed is None else args.seed + episode - 1
    obs, reset_info = env.reset(seed=episode_seed); trajectory.clear(); total_reward = 0.0
    # Mirror an eligible negative-X episode into the policy's stronger learned
    # orientation and map steering actions back.  The initial-state gate avoids
    # unsafe transfer in low-altitude, outward-moving emergencies.  Holding the
    # choice for the full episode avoids discontinuities near the centerline.
    mirror_episode = bool(
        not args.raw_policy and initial_canonical_mask(obs[None, ...])[0]
    )
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
        if not running: break
        if args.raw_policy:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            with torch.no_grad():
                probabilities = policy_probabilities(
                    model, obs[None, ...],
                    canonical_mask=np.array([mirror_episode]),
                )
            action = int(np.argmax(probabilities[0]))
        obs, reward, terminated, truncated, info = env.step(action); total_reward += reward
        render(action, episode, total_reward, info); clock.tick(control_hz)
        if terminated or truncated:
            print(f"Episode {episode}: {info['status']} reward={total_reward:.2f} reason={info.get('crash_reason')}")
            pygame.time.wait(700); break
env.close(); pygame.quit()
