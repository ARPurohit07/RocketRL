"""Gymnasium interface for full 3D rocket landing."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rocket3d_physics import Rocket3DPhysics


class Rocket3DEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self, challenge=1.0, control_repeat=2, gamma=0.99,
        focus_axis=None, height_floor=None, height_ceiling=None,
        min_start_radius=0.0,
    ):
        super().__init__()
        if control_repeat < 1:
            raise ValueError("control_repeat must be positive")
        self.physics = Rocket3DPhysics(
            challenge=challenge, focus_axis=focus_axis,
            height_floor=height_floor, height_ceiling=height_ceiling,
            min_start_radius=min_start_radius,
        )
        self.control_repeat = int(control_repeat)
        self.gamma = float(gamma)
        self.action_space = spaces.MultiDiscrete([2, 3, 3])
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(18,), dtype=np.float32)
        self.previous_potential = 0.0

    def set_challenge(self, challenge):
        self.physics.set_challenge(challenge)

    def _observation(self):
        return np.clip(self.physics.observe(), -1.0, 1.0).astype(np.float32)

    def _potential(self):
        p = self.physics
        position, velocity = p.position, p.velocity
        altitude = max(0.0, position[2] - 1.4)
        time_to_impact = (velocity[2] + np.sqrt(max(
            0.0, velocity[2] ** 2 + 2 * 9.81 * altitude
        ))) / 9.81
        projected_xy = position[:2] + velocity[:2] * min(time_to_impact, 2.5)
        rotation = p.rotation_matrix
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        yaw_frame = np.array([[c, s], [-s, c]])
        projected_local = yaw_frame @ projected_xy
        velocity_local = yaw_frame @ velocity[:2]
        desired_up_xy = -np.clip(
            0.065 * projected_local + 0.035 * velocity_local, -0.28, 0.28
        ) * min(1.0, altitude / 5.0)
        actual_up_xy = yaw_frame @ rotation[:2, 2]
        braking = max(0.0, -velocity[2]) ** 2 / 20.0
        violation = max(0.0, braking + 0.4 - altitude)
        return -float(
            5.0 * np.linalg.norm(projected_xy)
            + 2.0 * np.linalg.norm(position[:2])
            + 1.0 * np.linalg.norm(velocity[:2])
            + 8.0 * np.linalg.norm(actual_up_xy - desired_up_xy)
            + 0.4 * np.linalg.norm(p.angular_velocity)
            + 1.0 * violation
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        episode_seed = int(self.np_random.integers(0, 2**32 - 1))
        self.physics.reset(seed=episode_seed, start=(options or {}).get("start"))
        self.previous_potential = self._potential()
        return self._observation(), self._info()

    def _info(self):
        p = self.physics
        feet, hull = p.contacts()
        return {
            "status": p.status, "is_success": p.status == "landed",
            "reason": p.reason, "start": p.start,
            "position": p.position.tolist(), "velocity": p.velocity.tolist(),
            "quaternion": p.data.qpos[3:7].tolist(),
            "angular_velocity": p.angular_velocity.tolist(),
            "fuel": p.fuel, "feet_contact": feet.tolist(),
            "hull_contact": hull, "steps": p.steps,
        }

    def step(self, action):
        action = np.asarray(action, dtype=int)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        p = self.physics
        frames = 0
        for _ in range(self.control_repeat):
            p.step(action)
            frames += 1
            if p.status != "flying":
                break
        position, velocity = p.position, p.velocity
        altitude = max(0.0, position[2] - 1.4)
        descent_excess = max(0.0, -velocity[2] - (0.5 + 0.4 * np.sqrt(altitude)))
        upward_excess = max(0.0, velocity[2] - 1.0)
        reward = -0.02 * frames - 0.02 * descent_excess**2
        reward -= 0.015 * upward_excess**2
        # Keep the target visible to PPO throughout the flight. A pure
        # potential difference telescopes over an episode and provides little
        # incentive to correct large lateral errors before the first contact.
        reward -= 0.01 * frames * float(np.dot(position[:2], position[:2]))
        # A rocket that is still moving away from the pad needs altitude to
        # brake and translate. Penalize using that margin before touchdown.
        lead_position = position[:2] + 0.7 * velocity[:2]
        off_pad_distance = max(0.0, np.linalg.norm(lead_position) - 1.8)
        required_clearance = 0.5 + 1.5 * off_pad_distance
        altitude_shortfall = max(0.0, required_clearance - altitude)
        reward -= 0.35 * frames * altitude_shortfall**2
        reward -= 0.03 * np.linalg.norm(velocity[:2]) if altitude < 4.0 else 0.0
        if p.status == "landed":
            reward += 150.0
        elif p.status == "crashed":
            reward -= 100.0
        elif p.status == "out_of_bounds":
            reward -= 200.0
        elif p.status == "timeout":
            reward -= 40.0
        next_potential = 0.0 if p.status == "landed" else self._potential()
        reward += self.gamma * next_potential - self.previous_potential
        self.previous_potential = next_potential
        terminated = p.status in ("landed", "crashed", "out_of_bounds")
        truncated = p.status == "timeout"
        return self._observation(), float(reward), terminated, truncated, self._info()

    def close(self):
        self.physics = None
