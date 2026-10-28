import numpy as np

from rocket3d_env import Rocket3DEnv
from rocket3d_physics import Rocket3DPhysics


def test_3d_seed_reproducibility_and_observation_contract():
    a, b = Rocket3DEnv(), Rocket3DEnv()
    obs_a, info_a = a.reset(seed=127)
    obs_b, info_b = b.reset(seed=127)
    np.testing.assert_allclose(obs_a, obs_b)
    assert info_a["start"] == info_b["start"]
    assert obs_a.shape == (18,)
    assert a.observation_space.contains(obs_a)
    assert a.action_space.contains(np.array([1, 0, 2]))
    a.close(); b.close()


def test_thrust_changes_vertical_motion():
    start = {
        "position": [0.0, 0.0, 10.0],
        "velocity": [0.0, 0.0, 0.0],
        "angles": [0.0, 0.0, 0.0],
        "omega": [0.0, 0.0, 0.0],
    }
    coast, powered = Rocket3DPhysics(), Rocket3DPhysics()
    coast.reset(start=start)
    powered.reset(start=start)
    for _ in range(20):
        coast.step([0, 1, 1])
        powered.step([1, 1, 1])
    assert powered.position[2] > coast.position[2] + 0.7
    assert powered.fuel < coast.fuel


def test_steering_controls_both_horizontal_axes():
    start = {
        "position": [0.0, 0.0, 10.0],
        "velocity": [0.0, 0.0, 0.0],
        "angles": [0.0, 0.0, 0.0],
        "omega": [0.0, 0.0, 0.0],
    }
    x, y = Rocket3DPhysics(), Rocket3DPhysics()
    x.reset(start=start); y.reset(start=start)
    x.step([1, 2, 1]); y.step([1, 1, 2])
    assert x.velocity[0] > 0.0 and abs(x.velocity[1]) < 1e-6
    assert y.velocity[1] > 0.0 and abs(y.velocity[0]) < 1e-6


def test_observation_tracks_rocket_yaw_frame():
    a, b = Rocket3DPhysics(), Rocket3DPhysics()
    common = {"omega": [0.0, 0.0, 0.0]}
    a.reset(start={**common, "position": [2.0, 0.0, 10.0],
                   "velocity": [1.0, 0.0, -1.0],
                   "angles": [0.0, 0.0, 0.0]})
    b.reset(start={**common, "position": [0.0, 2.0, 10.0],
                   "velocity": [0.0, 1.0, -1.0],
                   "angles": [0.0, 0.0, np.pi / 2]})
    np.testing.assert_allclose(a.observe()[:9], b.observe()[:9], atol=1e-6)
