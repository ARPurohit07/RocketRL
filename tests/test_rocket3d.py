import numpy as np
import torch

from pythia_controller import PythiaDualAxis
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


def test_3d_landing_requires_pad_contact():
    start = {
        "position": [0.0, 0.0, 1.5],
        "velocity": [0.0, 0.0, -0.2],
        "angles": [0.0, 0.0, 0.0],
        "omega": [0.0, 0.0, 0.0],
    }
    on_pad, off_pad = Rocket3DPhysics(), Rocket3DPhysics()
    on_pad.reset(start=start)
    off_pad.reset(start={**start, "position": [4.0, 0.0, 1.5]})
    for _ in range(120):
        on_pad.step([0, 1, 1])
        off_pad.step([0, 1, 1])
        if on_pad.status != "flying" and off_pad.status != "flying":
            break
    assert on_pad.status == "landed"
    assert off_pad.status == "crashed"
    assert off_pad.reason == "off_pad"


def test_y_focus_keeps_x_recoverable_and_varies_both_y_directions():
    physics = Rocket3DPhysics(challenge=1.0, focus_axis="y")
    lateral = []
    for seed in range(64):
        observation = physics.reset(seed=seed)
        local_x, local_y = observation[:2] * 12.0
        assert abs(local_x) <= 1.01
        lateral.append(local_y)
    assert min(lateral) < -2.0
    assert max(lateral) > 2.0


def test_low_height_training_sampling_does_not_narrow_default_starts():
    low = Rocket3DPhysics(challenge=1.0, height_ceiling=10.0)
    full = Rocket3DPhysics(challenge=1.0)
    low_heights = []
    full_heights = []
    for seed in range(64):
        low.reset(seed=seed)
        full.reset(seed=seed)
        low_heights.append(low.position[2])
        full_heights.append(full.position[2])
    assert all(6.0 <= height <= 10.0 for height in low_heights)
    assert max(full_heights) > 20.0


def test_far_start_training_sampling_does_not_narrow_default_starts():
    far = Rocket3DPhysics(challenge=1.0, min_start_radius=3.5)
    full = Rocket3DPhysics(challenge=1.0)
    far_radii = []
    full_radii = []
    for seed in range(64):
        far.reset(seed=seed)
        full.reset(seed=seed)
        far_radii.append(np.linalg.norm(far.position[:2]))
        full_radii.append(np.linalg.norm(full.position[:2]))
    assert min(far_radii) >= 3.5
    assert min(full_radii) < 2.0


def test_middle_height_offset_training_sampling():
    physics = Rocket3DPhysics(
        challenge=1.0, height_floor=10.0, height_ceiling=16.0,
        min_start_radius=3.0,
    )
    for seed in range(64):
        physics.reset(seed=seed)
        assert 10.0 <= physics.position[2] <= 16.0
        assert np.linalg.norm(physics.position[:2]) >= 3.0


def test_dual_axis_views_separate_lateral_state():
    state = torch.zeros((1, 8, 18))
    state[..., 0] = 0.4
    state[..., 1] = -0.2
    state[..., 3] = 0.5
    state[..., 4] = -0.3
    state[..., 6] = -0.1
    state[..., 7] = 0.2
    x = PythiaDualAxis._axis_view(state, 0)
    y = PythiaDualAxis._axis_view(state, 1)
    assert x.shape == y.shape == (1, 8, 9)
    np.testing.assert_allclose(x[0, -1, [0, 2, 4]].numpy(), [0.32, 0.4, 0.1])
    np.testing.assert_allclose(y[0, -1, [0, 2, 4]].numpy(), [-0.16, -0.24, -0.2])
