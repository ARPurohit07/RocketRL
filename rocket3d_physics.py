"""MuJoCo rigid-body rocket with a circular landing pad and four feet."""

from dataclasses import dataclass

import mujoco
import numpy as np


SCENE = """
<mujoco model="rocket3d">
  <option timestep="0.02" gravity="0 0 -9.81" integrator="RK4"/>
  <default><geom friction="1.2 0.02 0.002" solref="0.015 1"/></default>
  <worldbody>
    <geom name="ground" type="plane" size="40 40 0.1" rgba="0.18 0.20 0.23 1"/>
    <geom name="pad" type="cylinder" pos="0 0 -0.018" size="3.0 0.012"
          contype="0" conaffinity="0" rgba="0.12 0.7 0.28 1"/>
    <body name="rocket" pos="0 0 10">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="42.5" diaginertia="8 8 2.5"/>
      <geom name="hull" type="cylinder" size="0.38 0.90" mass="0"
            rgba="0.78 0.82 0.88 1"/>
      <geom name="foot_fl" type="sphere" pos="0.70 0.70 -1.32" size="0.12" mass="0"/>
      <geom name="foot_fr" type="sphere" pos="0.70 -0.70 -1.32" size="0.12" mass="0"/>
      <geom name="foot_bl" type="sphere" pos="-0.70 0.70 -1.32" size="0.12" mass="0"/>
      <geom name="foot_br" type="sphere" pos="-0.70 -0.70 -1.32" size="0.12" mass="0"/>
    </body>
  </worldbody>
</mujoco>
"""


@dataclass(frozen=True)
class StartRange:
    xy: float
    height: tuple[float, float]
    drift: float
    descent: tuple[float, float]
    tilt: float
    rotation: float


class Rocket3DPhysics:
    FPS = 50
    DT = 1.0 / FPS
    MAIN_THRUST = 850.0
    GIMBAL_RATIO = 0.14
    STEERING_TORQUE = 45.0
    MAIN_FUEL_RATE = 0.75
    SIDE_FUEL_RATE = 0.12
    PAD_RADIUS = 3.0
    FUEL_CAPACITY = 12.0
    MAX_STEPS = 900
    SETTLE_STEPS = 12
    MAX_CONTACT_STEPS = 150
    CORE = StartRange(1.0, (7.0, 11.0), 0.25, (-1.0, -0.3), 0.04, 0.025)
    FULL = StartRange(4.0, (6.0, 24.0), 1.2, (-2.8, 0.5), 0.18, 0.10)

    def __init__(
        self, challenge=1.0, focus_axis=None, height_floor=None,
        height_ceiling=None, min_start_radius=0.0,
    ):
        self.model = mujoco.MjModel.from_xml_string(SCENE)
        self.data = mujoco.MjData(self.model)
        self.body_id = self.model.body("rocket").id
        self.ground_id = self.model.geom("ground").id
        self.hull_id = self.model.geom("hull").id
        self.foot_ids = tuple(self.model.geom(name).id for name in (
            "foot_fl", "foot_fr", "foot_bl", "foot_br"
        ))
        if focus_axis not in (None, "y"):
            raise ValueError("focus_axis must be None or 'y'")
        self.focus_axis = focus_axis
        if height_ceiling is not None and height_ceiling <= self.CORE.height[0]:
            raise ValueError("height_ceiling must exceed the minimum start height")
        if height_floor is not None and height_floor >= self.FULL.height[1]:
            raise ValueError("height_floor must be below the maximum start height")
        self.height_floor = height_floor
        self.height_ceiling = height_ceiling
        self.set_min_start_radius(min_start_radius)
        self.set_challenge(challenge)
        self.reset(seed=0)

    def set_min_start_radius(self, radius):
        if radius < 0:
            raise ValueError("min_start_radius must be nonnegative")
        self.min_start_radius = float(radius)

    def set_challenge(self, challenge):
        self.challenge = float(np.clip(challenge, 0.0, 1.0))

    def _range(self):
        c = self.challenge
        a, b = self.CORE, self.FULL
        mix = lambda first, second: first + c * (second - first)
        return StartRange(
            mix(a.xy, b.xy),
            tuple(mix(x, y) for x, y in zip(a.height, b.height)),
            mix(a.drift, b.drift),
            tuple(mix(x, y) for x, y in zip(a.descent, b.descent)),
            mix(a.tilt, b.tilt), mix(a.rotation, b.rotation),
        )

    def reset(self, seed=None, start=None):
        rng = np.random.default_rng(seed)
        spec = self._range()
        if start is None:
            if self.min_start_radius >= np.sqrt(2.0) * spec.xy:
                raise ValueError("min_start_radius exceeds this challenge's start range")
            max_height = (
                spec.height[1] if self.height_ceiling is None
                else min(spec.height[1], self.height_ceiling)
            )
            min_height = (
                spec.height[0] if self.height_floor is None
                else max(spec.height[0], self.height_floor)
            )
            if min_height >= max_height:
                raise ValueError("Height sampling range is empty")
            for _ in range(1000):
                position = np.array([
                    rng.uniform(-min(spec.xy, 1.0), min(spec.xy, 1.0))
                    if self.focus_axis == "y" else rng.uniform(-spec.xy, spec.xy),
                    rng.uniform(-spec.xy, spec.xy),
                    rng.uniform(min_height, max_height),
                ])
                if np.linalg.norm(position[:2]) < self.min_start_radius:
                    continue
                velocity = np.array([
                    rng.uniform(-min(spec.drift, 0.3), min(spec.drift, 0.3))
                    if self.focus_axis == "y" else rng.uniform(-spec.drift, spec.drift),
                    rng.uniform(-spec.drift, spec.drift),
                    rng.uniform(*spec.descent),
                ])
                angles = np.array([
                    rng.uniform(-spec.tilt, spec.tilt),
                    rng.uniform(-spec.tilt, spec.tilt),
                    rng.uniform(-np.pi, np.pi),
                ])
                omega = rng.uniform(-spec.rotation, spec.rotation, size=3)
                if self.focus_axis == "y":
                    yaw = angles[2]
                    yaw_rotation = np.array([
                        [np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)],
                    ])
                    position[:2] = yaw_rotation @ position[:2]
                    velocity[:2] = yaw_rotation @ velocity[:2]
                clearance = position[2] - 1.5
                brake_distance = max(0.0, -velocity[2]) ** 2 / (2 * 9.0)
                fall_time = (
                    velocity[2] + np.sqrt(velocity[2] ** 2 + 19.62 * clearance)
                ) / 9.81
                projected_xy = position[:2] + velocity[:2] * fall_time
                lateral_error = max(0.0, np.linalg.norm(projected_xy) - 2.2)
                lateral_capacity = 2.0 * (1.4 * fall_time) ** 2
                if (
                    clearance > 1.3 * brake_distance + 0.5
                    and lateral_error < lateral_capacity + 0.5
                ):
                    break
            else:
                raise RuntimeError("Could not sample a vertically recoverable start")
        else:
            position = np.asarray(start["position"], dtype=float)
            velocity = np.asarray(start["velocity"], dtype=float)
            angles = np.asarray(start["angles"], dtype=float)
            omega = np.asarray(start["omega"], dtype=float)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = position
        quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_euler2Quat(quaternion, angles, "xyz")
        self.data.qpos[3:7] = quaternion
        self.data.qvel[:3] = velocity
        self.data.qvel[3:6] = omega
        mujoco.mj_forward(self.model, self.data)
        self.fuel = self.FUEL_CAPACITY
        self.steps = 0
        self.stable_steps = 0
        self.contact_steps = 0
        self.status = "flying"
        self.reason = None
        self.start = {
            "position": position.tolist(), "velocity": velocity.tolist(),
            "angles": angles.tolist(), "omega": omega.tolist(),
        }
        return self.observe()

    @property
    def position(self):
        return self.data.qpos[:3].copy()

    @property
    def velocity(self):
        return self.data.qvel[:3].copy()

    @property
    def rotation_matrix(self):
        return self.data.xmat[self.body_id].reshape(3, 3).copy()

    @property
    def angular_velocity(self):
        return self.rotation_matrix @ self.data.qvel[3:6]

    def contacts(self):
        touching = set()
        for contact in self.data.contact[:self.data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.ground_id in pair:
                touching.update(pair - {self.ground_id})
        return np.array([id_ in touching for id_ in self.foot_ids]), self.hull_id in touching

    def observe(self):
        feet, _ = self.contacts()
        rotation = self.rotation_matrix
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        yaw_frame = np.array([[c, s], [-s, c]])
        local_position = yaw_frame @ self.position[:2]
        local_velocity = yaw_frame @ self.velocity[:2]
        local_up = yaw_frame @ rotation[:2, 2]
        local_omega = yaw_frame @ self.angular_velocity[:2]
        return np.concatenate((
            [local_position[0] / 12.0, local_position[1] / 12.0,
             self.position[2] / 26.0],
            [local_velocity[0] / 8.0, local_velocity[1] / 8.0,
             self.velocity[2] / 8.0],
            [local_up[0], local_up[1], rotation[2, 2], yaw / np.pi],
            [local_omega[0] / 3.0, local_omega[1] / 3.0,
             self.angular_velocity[2] / 3.0],
            feet.astype(float),
            [self.fuel / self.FUEL_CAPACITY],
        )).astype(np.float32)

    def step(self, action):
        if self.status != "flying":
            return self.observe(), self.status
        engine, steer_x, steer_y = (int(v) for v in action)
        if engine not in (0, 1) or steer_x not in (0, 1, 2) or steer_y not in (0, 1, 2):
            raise ValueError("Expected [engine 0/1, steer_x 0/1/2, steer_y 0/1/2]")
        before = self.velocity
        before_omega = self.angular_velocity
        active = engine and self.fuel > 0.0
        self.data.xfrc_applied[self.body_id] = 0
        if active:
            sx, sy = steer_x - 1, steer_y - 1
            local_force = np.array([self.GIMBAL_RATIO * sx, self.GIMBAL_RATIO * sy, 1.0])
            local_force *= self.MAIN_THRUST / np.linalg.norm(local_force)
            local_torque = np.array([-sy, sx, 0.0]) * self.STEERING_TORQUE
            rotation = self.rotation_matrix
            self.data.xfrc_applied[self.body_id, :3] = rotation @ local_force
            self.data.xfrc_applied[self.body_id, 3:6] = rotation @ local_torque
            self.fuel = max(0.0, self.fuel - self.DT * (
                self.MAIN_FUEL_RATE + self.SIDE_FUEL_RATE * (abs(sx) + abs(sy))
            ))
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        position, velocity = self.position, self.velocity
        tilt = np.arccos(np.clip(self.rotation_matrix[2, 2], -1.0, 1.0))
        feet, hull = self.contacts()
        touching = bool(feet.any())
        foot_positions = self.data.geom_xpos[list(self.foot_ids)]
        on_pad = np.linalg.norm(foot_positions[:, :2], axis=1) < self.PAD_RADIUS
        if hull:
            self.status, self.reason = "crashed", "hull_contact"
        elif touching and (
            before[2] < -5.0 or np.linalg.norm(before[:2]) > 6.0
            or tilt > 0.65 or np.linalg.norm(before_omega[:2]) > 2.0
        ):
            self.status, self.reason = "crashed", "hard_impact"
        elif touching and np.any(feet & ~on_pad):
            self.status, self.reason = "crashed", "off_pad"
        else:
            stable = (
                feet.sum() >= 3 and np.all(on_pad[feet])
                and np.linalg.norm(velocity[:2]) < 0.8 and abs(velocity[2]) < 0.8
                and tilt < 0.20 and np.linalg.norm(self.angular_velocity) < 0.4
            )
            self.stable_steps = self.stable_steps + 1 if stable else 0
            self.contact_steps = self.contact_steps + 1 if touching else 0
            if self.stable_steps >= self.SETTLE_STEPS:
                self.status = "landed"
            elif self.contact_steps >= self.MAX_CONTACT_STEPS:
                self.status, self.reason = "crashed", "unsettled_contact"
            elif np.any(np.abs(position[:2]) > 12.0):
                self.status, self.reason = "out_of_bounds", "lateral_boundary"
            elif position[2] > 30.0:
                self.status, self.reason = "out_of_bounds", "altitude_boundary"
            elif self.steps >= self.MAX_STEPS:
                self.status, self.reason = "timeout", "time_limit"
        return self.observe(), self.status
