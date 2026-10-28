import numpy as np
from Box2D import b2World, b2ContactListener
from dataclasses import dataclass


class GroundContact(b2ContactListener):

    def __init__(self):
        super().__init__()
        self._counts = {"left": 0, "right": 0}
        self.hull_hit = False

    def _update(self, contact, touching):
        a = contact.fixtureA.userData
        b = contact.fixtureB.userData

        if "ground" not in (a, b):
            return

        part = b if a == "ground" else a

        if part in self._counts:
            self._counts[part] = max(0, self._counts[part] + (1 if touching else -1))
        elif part == "hull" and touching:
            self.hull_hit = True

    @property
    def left(self):
        return self._counts["left"] > 0

    @property
    def right(self):
        return self._counts["right"] > 0

    def BeginContact(self, contact):
        self._update(contact, True)

    def EndContact(self, contact):
        self._update(contact, False)


class RocketPhysics:

    FPS = 50
    DT = 1.0 / FPS

    # The fixture mass is about 42.5 kg.  Thus T/W=2.04 and the maximum
    # upward acceleration is T/m-|g| ~= 10.6 m/s^2: enough to brake a
    # 6 m/s descent in about 0.57 s over 1.7 m, but not enough to hover
    # indefinitely when fuel is exhausted.
    MAIN_THRUST = 850.0
    GIMBAL_RATIO = 0.18
    STEERING_TORQUE = 45.0

    MAIN_FUEL_RATE = 0.75
    SIDE_FUEL_RATE = 0.18

    MAX_STEPS = 1200
    LANDING_CONFIRMATION_STEPS = 15
    PAD_HALF_WIDTH = 4.0
    X_OBSERVATION_SCALE = 15.0
    ALTITUDE_OBSERVATION_SCALE = 30.0
    HORIZONTAL_BOUND = 15.0
    ALTITUDE_BOUND = 32.0

    @dataclass(frozen=True)
    class StartSpec:
        x: tuple[float, float]
        altitude: tuple[float, float]
        vx: tuple[float, float]
        vy: tuple[float, float]
        angle: tuple[float, float]
        omega: tuple[float, float]

    # One continuous distribution is used for every episode: there are no
    # easy/medium/hard modes.  Uniform sampling gives the edge of every range
    # meaningful coverage instead of concentrating almost all starts near the
    # pad.  It spans low emergency starts, long high-altitude descents, lateral
    # drift, initial tilt, and initial rotation.  Vertical rejection sampling
    # removes draws that are physically impossible to brake.
    # The low end of the continuous schedule is compact but still randomized.
    # All position, height, velocity and attitude ranges then widen smoothly,
    # so there are no discrete difficulty modes or hand-selected episodes.
    INITIAL_START_SPEC = StartSpec(
        (-2.0, 2.0), (6.0, 10.0), (-0.3, 0.3), (-0.75, 0.10),
        (-0.03, 0.03), (-0.02, 0.02)
    )
    RANDOM_START_SPEC = StartSpec(
        (-8.0, 8.0), (4.0, 28.0), (-3.0, 3.0), (-4.0, 0.75),
        (-0.30, 0.30), (-0.20, 0.20)
    )

    def __init__(self, gravity=-9.81, challenge=1.0):
        self.gravity = gravity
        self.set_challenge(challenge)
        self.reset()

    def set_challenge(self, challenge):
        self.challenge = float(np.clip(challenge, 0.0, 1.0))

    @staticmethod
    def _blend_pair(core, broad, amount):
        return tuple(float(a + amount * (b - a)) for a, b in zip(core, broad))

    def _active_start_distribution(self):
        c, b, amount = self.INITIAL_START_SPEC, self.RANDOM_START_SPEC, self.challenge
        spec = self.StartSpec(*(
            self._blend_pair(getattr(c, field), getattr(b, field), amount)
            for field in ("x", "altitude", "vx", "vy", "angle", "omega")
        ))
        return spec

    def reset(self, seed=None, initial_position=None, initial_velocity=None,
              initial_angle=None, initial_angular_velocity=None,
              randomized=True):
        rng = np.random.default_rng(seed)

        if not randomized:
            raise ValueError("Only randomized starts are supported")
        spec = self._active_start_distribution()

        def sample(bounds):
            return float(rng.uniform(bounds[0], bounds[1]))

        if initial_position is None:
            initial_position = (sample(spec.x), sample(spec.altitude))
        if initial_velocity is None:
            initial_velocity = (sample(spec.vx), sample(spec.vy))
        if initial_angle is None:
            initial_angle = sample(spec.angle)
        if initial_angular_velocity is None:
            initial_angular_velocity = sample(spec.omega)

        self.world = b2World(gravity=(0, self.gravity), doSleep=True)

        self.contacts = GroundContact()
        self.world.contactListener = self.contacts
        ground = self.world.CreateStaticBody(position=(0, -0.5))

        fixture = ground.CreatePolygonFixture(box=(60, 0.5), friction=0.8)
        fixture.userData = "ground"

        self.rocket = self.world.CreateDynamicBody(
            position=initial_position,
            angle=float(initial_angle),
            linearDamping=0.02,
            angularDamping=0.05,
        )

        hull = self.rocket.CreatePolygonFixture(
            box=(0.5, 1.05), density=20.0, friction=0.4
        )
        hull.userData = "hull"

        left = self.rocket.CreatePolygonFixture(
            box=(0.22, 0.07, (-0.85, -1.45), 0), density=2.0, friction=0.9
        )
        left.userData = "left"

        right = self.rocket.CreatePolygonFixture(
            box=(0.22, 0.07, (0.85, -1.45), 0), density=2.0, friction=0.9
        )
        right.userData = "right"

        self.rocket.linearVelocity = initial_velocity
        self.rocket.angularVelocity = float(initial_angular_velocity)

        self.fuel = 12.0
        self.steps = 0
        self.stable_steps = 0
        self.ground_contact_steps = 0
        self.off_pad_contact_steps = 0
        self.status = "flying"
        self.crash_reason = None

        self.start_state = {
            "x": float(initial_position[0]), "y": float(initial_position[1]),
            "vx": float(initial_velocity[0]), "vy": float(initial_velocity[1]),
            "angle": float(initial_angle), "angular_velocity": float(initial_angular_velocity),
            "randomized": True, "challenge": self.challenge,
        }

        return self.observe()

    def is_recoverable_start(self):
        """Conservative vertical braking feasibility check at reset."""
        mass = float(self.rocket.mass)
        clearance = float(self.rocket.position.y - 1.55)
        upward_accel = self.MAIN_THRUST * max(0.0, np.cos(self.rocket.angle)) / mass + self.gravity
        if upward_accel <= 0:
            return False
        downward_speed = max(0.0, -float(self.rocket.linearVelocity.y))
        braking_distance = downward_speed**2 / (2.0 * upward_accel)
        return clearance > 1.25 * braking_distance + 0.25

    def _fire(self, action):

        if self.fuel <= 0.0:
            return

        # Actions 1 and 3 combine a gimbaled main-engine command with RCS
        # torque.  Their immediate lateral acceleration and eventual tilted
        # thrust point in the same direction, giving the policy a learnable,
        # physically coherent control response.
        if action in (1, 2, 3):
            steer = 1 if action == 1 else -1 if action == 3 else 0
            local_x = -self.GIMBAL_RATIO * steer
            norm = float(np.sqrt(1.0 + local_x**2))
            direction = self.rocket.GetWorldVector((local_x / norm, 1.0 / norm))

            self.rocket.ApplyForceToCenter(
                (direction.x * self.MAIN_THRUST, direction.y * self.MAIN_THRUST), True
            )

            self.fuel = max(0.0, self.fuel - self.MAIN_FUEL_RATE * self.DT)

        if action in (1, 3):
            sign = 1 if action == 1 else -1
            self.rocket.ApplyTorque(sign * self.STEERING_TORQUE, True)

            self.fuel = max(0.0, self.fuel - self.SIDE_FUEL_RATE * self.DT)

    def observe(self):
        body = self.rocket

        return np.array(
            [
                body.position.x / self.X_OBSERVATION_SCALE,
                (body.position.y - 1.5) / self.ALTITUDE_OBSERVATION_SCALE,
                body.linearVelocity.x / 10.0,
                body.linearVelocity.y / 10.0,
                body.angle,
                body.angularVelocity / 5.0,
                float(self.contacts.left),
                float(self.contacts.right),
            ],
            dtype=np.float32,
        )

    def step(self, action):

        if self.status != "flying":
            return self.observe(), self.status

        if action not in (0, 1, 2, 3):
            raise ValueError("Invalid action")

        body = self.rocket

        # Record velocities immediately before physics update; these are the
        # impact velocities, not post-contact solver velocities.
        vx_before = float(body.linearVelocity.x)
        vy_before = float(body.linearVelocity.y)
        omega_before = float(body.angularVelocity)

        was_touching = self.contacts.left or self.contacts.right

        self._fire(action)

        self.world.Step(self.DT, 8, 3)
        self.world.ClearForces()

        self.steps += 1

        x = float(body.position.x)
        y = float(body.position.y)
        angle = float(body.angle)
        vx = float(body.linearVelocity.x)
        vy = float(body.linearVelocity.y)
        omega = float(body.angularVelocity)

        left = self.contacts.left
        right = self.contacts.right
        touching = left or right

        # 1. Hull hitting the ground is a crash
        if self.contacts.hull_hit:
            self.status = "crashed"
            self.crash_reason = "Hull struck ground"

        # 2. Detect genuinely dangerous impacts
        elif touching and not was_touching:

            hard_impact = (
                vy_before < -5.0
                or abs(vx_before) > 7.0
                or abs(omega_before) > 2.0
                or abs(angle) > 0.70
            )

            if hard_impact:
                self.status = "crashed"

                self.crash_reason = (
                    f"Pre-impact VX={vx_before:.2f}, "
                    f"VY={vy_before:.2f}, "
                    f"Omega={omega_before:.2f}, "
                    f"Angle={angle:.2f} | "
                    f"Excessive horizontal speed={abs(vx_before) > 7.0}, "
                    f"Excessive descent={vy_before < -5.0}, "
                    f"Excessive rotation={abs(omega_before) > 2.0}, "
                    f"Excessive tilt={abs(angle) > 0.70}"
                )

        # 3. Give the rocket time to settle
        if self.status == "flying":

            left_pos = body.GetWorldPoint((-0.85, -1.45))
            right_pos = body.GetWorldPoint((0.85, -1.45))

            both_on_pad = (
                left
                and right
                and abs(left_pos.x) <= self.PAD_HALF_WIDTH
                and abs(right_pos.x) <= self.PAD_HALF_WIDTH
            )

            stable = (
                abs(vx) < 0.8
                and abs(vy) < 0.8
                and abs(angle) < 0.22
                and abs(omega) < 0.35
            )

            if both_on_pad and stable:
                self.stable_steps += 1
            else:
                self.stable_steps = 0

            # Successful landing after sustained stable two-leg pad contact.
            if self.stable_steps >= self.LANDING_CONFIRMATION_STEPS:
                self.status = "landed"

            # Prevent a rocket wedged away from the pad from remaining
            # "flying" and collecting shaping reward.  Valid two-leg pad
            # contact must remain alive long enough to satisfy the 15-frame
            # landing confirmation above.
            if touching:
                self.ground_contact_steps += 1
            else:
                self.ground_contact_steps = 0

            off_pad_contact = (
                (left and abs(left_pos.x) > self.PAD_HALF_WIDTH)
                or (right and abs(right_pos.x) > self.PAD_HALF_WIDTH)
            )
            if off_pad_contact:
                self.off_pad_contact_steps += 1
            else:
                self.off_pad_contact_steps = 0

            if self.status == "flying" and self.off_pad_contact_steps >= 8:
                self.status = "crashed"
                self.crash_reason = "Ground contact outside landing pad"
            elif (
                self.status == "flying"
                and self.ground_contact_steps >= 50
                and not both_on_pad
            ):
                self.status = "crashed"
                self.crash_reason = "Ground contact without a valid two-leg landing"

        # 4. Handle remaining termination conditions
        if self.status == "flying":

            # Leaving the controllable flight envelope is a failure.  A
            # rocket climbing to 50 m is not a successful hover/landing.
            if abs(x) > self.HORIZONTAL_BOUND:
                self.status = "out_of_bounds"
                self.crash_reason = "Exceeded lateral flight boundary"

            elif y > self.ALTITUDE_BOUND:
                self.status = "out_of_bounds"
                self.crash_reason = "Exceeded altitude flight boundary"

            elif self.steps >= self.MAX_STEPS:
                self.status = "timeout"
                self.crash_reason = "Episode time limit reached"

        return self.observe(), self.status
