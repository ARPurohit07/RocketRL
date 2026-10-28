import gymnasium as gym
import numpy as np

from gymnasium import spaces
from rocket_physics import RocketPhysics


class RocketLandingEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self, gamma=0.99, challenge=1.0, control_repeat=2,
        antithetic_starts=False,
    ):
        super().__init__()

        self.gamma = gamma
        if int(control_repeat) < 1:
            raise ValueError("control_repeat must be at least 1")
        self.control_repeat = int(control_repeat)
        self.antithetic_starts = bool(antithetic_starts)
        self._pending_antithetic_start = None
        self.physics = RocketPhysics(challenge=challenge)
        self.max_fuel = 12.0

        # =====================================
        # ACTION SPACE
        # =====================================
        # 0 = No thrust
        # 1 = Main engine + left steering engine
        # 2 = Main engine
        # 3 = Main engine + right steering engine

        self.action_space = spaces.Discrete(4)

        # =====================================
        # OBSERVATION SPACE
        # =====================================
        # x, y, vx, vy, angle, omega,
        # left_leg, right_leg, fuel

        # All continuous state is bounded and normalized.  This is important
        # for PPO and also makes the saved policy/configuration unambiguous.
        low = np.array([-1, -1, -1, -1, -1, -1, 0, 0, 0], dtype=np.float32)
        high = np.ones(9, dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.previous_potential = 0.0

    def set_challenge(self, challenge):
        self.physics.set_challenge(challenge)

    # =========================================
    # GET CURRENT OBSERVATION
    # =========================================

    def _observation(self):

        state = self.physics.observe()

        fuel_fraction = np.array(
            [np.clip(self.physics.fuel / 12.0, 0.0, 1.0)], dtype=np.float32
        )

        return np.clip(np.concatenate([state, fuel_fraction]), -1.0, 1.0).astype(np.float32)

    # =========================================
    # POTENTIAL-BASED REWARD SHAPING
    # =========================================

    def _potential(self, state):

        x, y, vx, vy, angle, omega = [float(v) for v in state[:6]]

        # Convert normalized observations back to physical units.  Guidance
        # must aim at where the rocket is heading, not only where it is now.
        x_m = self.physics.X_OBSERVATION_SCALE * x
        altitude_m = max(0.0, self.physics.ALTITUDE_OBSERVATION_SCALE * y)
        vx_mps = 10.0 * vx
        vy_mps = 10.0 * vy
        omega_rps = 5.0 * omega

        fall_time = (
            vy_mps + np.sqrt(max(0.0, vy_mps**2 + 2.0 * 9.81 * altitude_m))
        ) / 9.81
        fall_time = float(np.clip(fall_time, 0.0, 3.0))
        projected_x = x_m + vx_mps * fall_time

        # Positive angle points main thrust toward negative world X.  The
        # desired tilt therefore has the same sign as projected pad error.
        steering_strength = float(np.clip(altitude_m / 5.0, 0.0, 1.0))
        desired_angle = steering_strength * float(
            np.clip(0.07 * projected_x + 0.04 * vx_mps, -0.28, 0.28)
        )

        safe_descent = 0.6 + 0.45 * np.sqrt(altitude_m)
        descent_error = max(0.0, -vy_mps - safe_descent)
        braking_distance = max(0.0, -vy_mps) ** 2 / (2.0 * 10.0)
        braking_violation = max(0.0, braking_distance + 0.4 - altitude_m)
        penalty = (
            1.80 * abs(projected_x)
            + 0.24 * abs(x_m)
            + 0.54 * abs(vx_mps)
            + 4.0 * abs(angle - desired_angle)
            + 0.30 * abs(omega_rps)
            + 0.60 * descent_error
            + 1.20 * braking_violation
        )

        return -float(penalty)

    # =========================================
    # RESET ENVIRONMENT
    # =========================================

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        episode_seed = int(self.np_random.integers(0, 2**32 - 1))
        options = dict(options or {})
        explicit_start = any(
            key in options for key in (
                "initial_position", "initial_velocity", "initial_angle",
                "initial_angular_velocity",
            )
        )
        antithetic_start = False
        if (
            self.antithetic_starts
            and not explicit_start
            and self._pending_antithetic_start is not None
        ):
            options = self._pending_antithetic_start
            self._pending_antithetic_start = None
            explicit_start = True
            antithetic_start = True
        if explicit_start:
            self.physics.reset(
                seed=episode_seed,
                initial_position=options.get("initial_position"),
                initial_velocity=options.get("initial_velocity"),
                initial_angle=options.get("initial_angle"),
                initial_angular_velocity=options.get("initial_angular_velocity"),
            )
            if not self.physics.is_recoverable_start():
                raise ValueError("Requested initial state is not vertically recoverable")
        else:
            # Rejection sampling keeps the requested ranges while excluding
            # states whose vertical braking distance cannot fit before touchdown.
            for attempt in range(1000):
                self.physics.reset(seed=episode_seed + attempt)
                if self.physics.is_recoverable_start():
                    break
            else:
                raise RuntimeError("Could not sample a recoverable initial condition")

        observation = self._observation()

        if self.antithetic_starts and not explicit_start:
            start = self.physics.start_state
            self._pending_antithetic_start = {
                "initial_position": (-start["x"], start["y"]),
                "initial_velocity": (-start["vx"], start["vy"]),
                "initial_angle": -start["angle"],
                "initial_angular_velocity": -start["angular_velocity"],
            }

        self.previous_potential = self._potential(observation)

        info = {
            "status": "flying",
            "is_success": False,
            "fuel": self.physics.fuel,
            "steps": 0,
            "randomized": True,
            "antithetic_start": antithetic_start,
            "start": dict(self.physics.start_state),
        }

        return observation, info

    # =========================================
    # EXECUTE ACTION AND CALCULATE REWARD
    # =========================================

    def step(self, action):

        action = int(action)

        previous_fuel = self.physics.fuel

        # Hold each discrete command for a short, fixed control interval.  A
        # 25 Hz controller gives steering enough duration for PPO to observe
        # its consequence while Box2D still integrates at 50 Hz.
        physics_frames = 0
        status = self.physics.status
        for _ in range(self.control_repeat):
            _, status = self.physics.step(action)
            physics_frames += 1
            if status != "flying":
                break

        observation = self._observation()

        body = self.physics.rocket

        x = float(body.position.x)
        y = float(body.position.y)

        vx = float(body.linearVelocity.x)
        vy = float(body.linearVelocity.y)

        angle = float(body.angle)
        omega = float(body.angularVelocity)

        # =====================================
        # EPISODE TERMINATION
        # =====================================

        terminated = status in ("landed", "crashed", "out_of_bounds")

        truncated = status == "timeout"

        fuel_used = max(0.0, previous_fuel - self.physics.fuel)
        altitude = max(0.0, y - 1.55)
        safe_descent_speed = 0.5 + 0.35 * altitude
        descent_penalty = 0.03 * max(0.0, -vy - safe_descent_speed) ** 2
        # Upward motion is useful only briefly while braking.  Penalize
        # sustained ascent so a policy cannot maximize reward by escaping.
        upward_penalty = 0.02 * max(0.0, vy) ** 2
        if y > 10.0 and vy > 0.0:
            upward_penalty += 0.08 * (y - 10.0) * vy
        # Do not punish useful movement toward the pad.  Far from center only
        # outward velocity is costly; near the pad all lateral speed is risky.
        outward_speed = max(0.0, np.sign(x) * vx)
        controlled_speed = abs(vx) if abs(x) < 2.0 else outward_speed
        horizontal_penalty = (0.012 + 0.035 / (1.0 + altitude)) * controlled_speed
        fuel_penalty = 0.05 * fuel_used
        projected_fall_time = (
            vy + np.sqrt(max(0.0, vy**2 + 2.0 * 9.81 * altitude))
        ) / 9.81
        projected_x = x + vx * float(np.clip(projected_fall_time, 0.0, 3.0))
        steering_strength = float(np.clip(altitude / 5.0, 0.0, 1.0))
        desired_angle = steering_strength * float(
            np.clip(0.07 * projected_x + 0.04 * vx, -0.28, 0.28)
        )
        reward = physics_frames * (
            -0.02 - descent_penalty - upward_penalty - horizontal_penalty
        ) - fuel_penalty

        if status == "landed":

            reward += 150.0

        elif status == "crashed":

            reward -= 100.0

        elif status == "out_of_bounds":

            # Escaping upward or sideways must be decisively worse than a
            # recoverable but imperfect approach.
            reward -= 200.0

        elif status == "timeout":

            reward -= 20.0

        # =====================================
        # 7. POTENTIAL-BASED SHAPING
        # =====================================

        # A valid landing has zero remaining error.  Failed terminal states
        # retain their state potential, so crashing early cannot erase pad,
        # speed, or attitude error and game the shaping reward.
        if status == "landed":
            next_potential = 0.0
        else:
            next_potential = self._potential(observation)

        shaping_reward = self.gamma * next_potential - self.previous_potential

        self.previous_potential = next_potential

        reward += shaping_reward

        # =====================================
        # 8. EPISODE INFORMATION
        # =====================================

        info = {
            "status": status,
            "is_success": status == "landed",
            "fuel": float(self.physics.fuel),
            "fuel_used": float(fuel_used),
            "steps": int(self.physics.steps),
            "control_repeat": self.control_repeat,
            "physics_frames": physics_frames,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "angle": angle,
            "angular_velocity": omega,
            "altitude": altitude,
            "safe_descent_speed": safe_descent_speed,
            "descent_penalty": float(descent_penalty),
            "upward_penalty": float(upward_penalty),
            "horizontal_penalty": float(horizontal_penalty),
            "fuel_penalty": float(fuel_penalty),
            "projected_x": float(projected_x),
            "desired_angle": float(desired_angle),
            "shaping_reward": float(shaping_reward),
            "crash_reason": self.physics.crash_reason,
            "left_contact": bool(self.physics.contacts.left),
            "right_contact": bool(self.physics.contacts.right),
            "randomized": True,
            "start": dict(self.physics.start_state),
        }

        if status in ("landed", "crashed"):
            info["touchdown_vx"] = abs(vx)
            info["touchdown_vy"] = abs(vy)
            info["landing_error"] = abs(x)
            info["landing_time"] = float(self.physics.steps * self.physics.DT)

        return (observation, float(reward), terminated, truncated, info)
