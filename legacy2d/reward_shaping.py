import gymnasium as gym


class LandingRewardWrapper(gym.Wrapper):

    def __init__(self, env, gamma=0.99):
        super().__init__(env)
        self.gamma = gamma
        self.previous_potential = 0.0

    def potential(self, state):
        x, y, vx, vy, angle, omega, left, right = state

        # Normalized altitude: 0 = near ground, 1 = high
        altitude = max(0.0, min(float(y) / 1.2, 1.0))
        near_ground = 1.0 - altitude

        # High altitude: prioritize horizontal control
        position_weight = 2.0 + 3.0 * altitude
        drift_weight = 1.5 + 2.5 * altitude

        # Low altitude: prioritize stability
        angle_weight = 1.5 + 2.5 * near_ground
        rotation_weight = 0.6 + 1.4 * near_ground

        # Accept faster descent at higher altitudes
        safe_descent = 0.12 + 0.35 * altitude

        excessive_descent = max(0.0, -float(vy) - safe_descent)

        descent_weight = 4.0 * near_ground

        penalty = (
            position_weight * abs(x)
            + drift_weight * abs(vx)
            + angle_weight * abs(angle)
            + rotation_weight * abs(omega)
            + descent_weight * excessive_descent
        )

        return -float(penalty)

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.previous_potential = self.potential(state)
        return state, info

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)

        # True terminal states have zero potential.
        # Time-limit truncation is treated as nonterminal.
        next_potential = 0.0 if terminated else self.potential(state)

        shaping_reward = self.gamma * next_potential - self.previous_potential

        self.previous_potential = next_potential

        info["original_reward"] = float(reward)
        info["shaping_reward"] = float(shaping_reward)

        return (state, float(reward + shaping_reward), terminated, truncated, info)
