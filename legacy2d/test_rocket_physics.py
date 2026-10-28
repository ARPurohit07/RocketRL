import numpy as np

from rocket_physics import RocketPhysics


def make_physics(pos=(0.0, 10.0), vel=(0.0, 0.0), angle=0.0, omega=0.0):
    p = RocketPhysics()
    p.reset(
        seed=123, initial_position=pos, initial_velocity=vel,
        initial_angle=angle, initial_angular_velocity=omega,
    )
    return p


def run(p, action, n=1):
    for _ in range(n):
        p.step(action)
    return p


def test_no_thrust_falls_under_gravity():
    p = make_physics()
    y0 = p.rocket.position.y
    run(p, 0, 10)
    assert p.rocket.position.y < y0
    assert p.rocket.linearVelocity.y < -1.5


def test_main_thrust_acceleration_is_upward():
    p = make_physics(vel=(0, 0))
    run(p, 2)
    assert p.rocket.linearVelocity.y > 0.1
    assert p.fuel < 12.0 and p.fuel >= 0.0


def test_side_thrusters_have_opposite_torque():
    left = make_physics()
    right = make_physics()
    run(left, 1, 5)
    run(right, 3, 5)
    assert left.rocket.angularVelocity > 0.0
    assert right.rocket.angularVelocity < 0.0


def test_steering_commands_have_immediate_consistent_lateral_authority():
    left = make_physics()
    right = make_physics()
    run(left, 1, 5)
    run(right, 3, 5)
    assert left.rocket.linearVelocity.x < -0.2
    assert right.rocket.linearVelocity.x > 0.2


def test_steering_actions_also_brake_descent():
    coast = make_physics(vel=(0, -2.0))
    steer = make_physics(vel=(0, -2.0))
    run(coast, 0, 5)
    run(steer, 1, 5)
    assert steer.rocket.linearVelocity.y > coast.rocket.linearVelocity.y


def test_main_engine_can_brake_descent():
    p = make_physics(pos=(0, 4.0), vel=(0, -6.0))
    run(p, 2, 15)
    assert p.rocket.linearVelocity.y > -6.0


def test_safe_touchdown_lands():
    p = make_physics(pos=(0, 1.9), vel=(0, -0.4))
    for _ in range(40):
        _, status = p.step(0)
        if status != "flying":
            break
    assert p.status == "landed", (p.status, p.crash_reason)


def test_fast_touchdown_crashes():
    p = make_physics(pos=(0, 2.2), vel=(0, -8.0))
    for _ in range(10):
        p.step(0)
        if p.status != "flying":
            break
    assert p.status == "crashed"
    assert "descent" in (p.crash_reason or "").lower()


def test_hull_contact_crashes():
    p = make_physics(pos=(0, 1.0), vel=(0, 0))
    p.step(0)
    assert p.status == "crashed"
    assert "Hull" in (p.crash_reason or "")


def test_ground_contact_outside_pad_terminates():
    p = make_physics(pos=(6.0, 2.0), vel=(0, -0.2))
    for _ in range(30):
        p.step(0)
        if p.status != "flying":
            break
    assert p.status == "crashed"
    assert "outside landing pad" in (p.crash_reason or "").lower()


def test_out_of_bounds_and_reproducibility():
    a = make_physics()
    b = make_physics()
    for action in [0, 2, 0, 1, 3] * 3:
        a.step(action)
        b.step(action)
    np.testing.assert_allclose(a.observe(), b.observe(), atol=1e-6)
    p = make_physics(pos=(36, 10), vel=(0, 0))
    p.step(0)
    assert p.status == "out_of_bounds"


def test_randomized_ranges_and_recoverability():
    spec = RocketPhysics.RANDOM_START_SPEC
    for seed in range(20):
        p = RocketPhysics(); p.reset(seed=seed)
        s = p.start_state
        assert spec.x[0] <= s["x"] <= spec.x[1]
        assert spec.altitude[0] <= s["y"] <= spec.altitude[1]
        assert spec.vx[0] <= s["vx"] <= spec.vx[1]
        assert spec.vy[0] <= s["vy"] <= spec.vy[1]
        assert spec.angle[0] <= s["angle"] <= spec.angle[1]
        assert spec.omega[0] <= s["angular_velocity"] <= spec.omega[1]
        assert p.rocket.position.y - 1.55 > 0
        assert p.is_recoverable_start()


def test_randomized_seed_reproducibility_and_variation():
    a = RocketPhysics(); b = RocketPhysics(); c = RocketPhysics()
    a.reset(seed=44); b.reset(seed=44); c.reset(seed=45)
    assert a.start_state == b.start_state
    assert a.start_state != c.start_state
