import numpy as np
import torch

from gymnasium.wrappers import FrameStackObservation
from rocket_env import RocketLandingEnv
from pythia_controller import PythiaController
from rocket_eval import (
    initial_canonical_mask,
    mirror_observations,
    unmirror_action_probabilities,
)


def test_env_contract_and_determinism():
    a = FrameStackObservation(RocketLandingEnv(), 8)
    b = FrameStackObservation(RocketLandingEnv(), 8)
    oa, _ = a.reset(seed=7)
    ob, _ = b.reset(seed=7)
    np.testing.assert_allclose(oa, ob)
    assert oa.shape == (8, 9)
    assert a.observation_space.contains(oa)
    for action in [0, 2, 0, 1]:
        oa, _, ta, tra, ia = a.step(action)
        assert a.observation_space.contains(oa)
        assert np.isfinite(oa).all()
        assert "status" in ia
        if ta or tra:
            break
    a.close(); b.close()


def test_randomized_env_keeps_pythia_contract():
    env = FrameStackObservation(RocketLandingEnv(), 8)
    obs, info = env.reset(seed=1234)
    assert obs.shape == (8, 9)
    assert env.observation_space.contains(obs)
    assert info["randomized"] is True
    assert info["start"]["y"] > 1.55
    env.close()


def test_antithetic_randomization_pairs_each_start_with_its_mirror():
    env = RocketLandingEnv(antithetic_starts=True)
    _, first = env.reset(seed=812)
    _, mirrored = env.reset()
    a, b = first["start"], mirrored["start"]
    assert mirrored["antithetic_start"] is True
    np.testing.assert_allclose(
        [b["x"], b["vx"], b["angle"], b["angular_velocity"]],
        [-a["x"], -a["vx"], -a["angle"], -a["angular_velocity"]],
    )
    np.testing.assert_allclose([b["y"], b["vy"]], [a["y"], a["vy"]])
    assert env.physics.is_recoverable_start()
    env.close()


def test_symmetry_transform_round_trip_and_action_mapping():
    observations = np.arange(2 * 8 * 9, dtype=np.float32).reshape(2, 8, 9)
    observations[..., 6:8] %= 2
    np.testing.assert_allclose(
        mirror_observations(mirror_observations(observations)), observations
    )
    probabilities = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    np.testing.assert_allclose(
        unmirror_action_probabilities(probabilities), [[0.1, 0.4, 0.3, 0.2]]
    )


def test_initial_canonical_mask_rejects_unrecoverable_symmetry_transfer():
    env = FrameStackObservation(RocketLandingEnv(), 8)
    safe, _ = env.reset(seed=1, options={
        "initial_position": (-4.0, 18.0),
        "initial_velocity": (0.0, -1.0),
        "initial_angle": 0.1,
        "initial_angular_velocity": 0.0,
    })
    low, _ = env.reset(seed=2, options={
        "initial_position": (-4.0, 8.0),
        "initial_velocity": (-1.0, -1.0),
        "initial_angle": 0.1,
        "initial_angular_velocity": 0.0,
    })
    outward_tilt, _ = env.reset(seed=3, options={
        "initial_position": (-4.0, 18.0),
        "initial_velocity": (0.0, -1.0),
        "initial_angle": 0.25,
        "initial_angular_velocity": 0.0,
    })
    assert initial_canonical_mask(safe[None, ...]).tolist() == [True]
    assert initial_canonical_mask(low[None, ...]).tolist() == [False]
    assert initial_canonical_mask(
        outward_tilt[None, ...], max_outward_tilt=0.22
    ).tolist() == [False]
    env.close()


def test_high_altitude_observation_does_not_clip():
    env = RocketLandingEnv()
    obs, _ = env.reset(seed=5, options={
        "initial_position": (0.0, 28.0),
        "initial_velocity": (0.0, -1.0),
        "initial_angle": 0.0,
        "initial_angular_velocity": 0.0,
    })
    assert 0.8 < obs[1] < 1.0
    env.close()


def test_reward_prefers_corrective_lateral_progress():
    options = {
        "initial_position": (6.0, 12.0),
        "initial_velocity": (0.0, -1.0),
        "initial_angle": 0.0,
        "initial_angular_velocity": 0.0,
    }
    corrective = RocketLandingEnv(control_repeat=2)
    wrong_way = RocketLandingEnv(control_repeat=2)
    corrective.reset(seed=1, options=options)
    wrong_way.reset(seed=1, options=options)
    _, corrective_reward, *_ = corrective.step(1)
    _, wrong_way_reward, *_ = wrong_way.step(3)
    assert corrective_reward > wrong_way_reward + 0.25
    corrective.close(); wrong_way.close()


def test_explicit_recoverable_start_for_stratified_evaluation():
    env = FrameStackObservation(RocketLandingEnv(), 8)
    obs, info = env.reset(seed=99, options={
        "initial_position": (6.0, 24.0),
        "initial_velocity": (2.0, -2.0),
        "initial_angle": 0.2,
        "initial_angular_velocity": -0.1,
    })
    assert env.observation_space.contains(obs)
    assert info["start"]["x"] == 6.0
    assert info["start"]["y"] == 24.0
    assert info["start"]["vx"] == 2.0
    env.close()


def test_pythia_lora_and_residual_gradients():
    env = FrameStackObservation(RocketLandingEnv(), 8)
    extractor = PythiaController(env.observation_space)
    x, _ = env.reset(seed=3)
    out = extractor(torch.as_tensor(x[None]))
    loss = out.square().mean()
    loss.backward()
    lora_grads = [p.grad for n, p in extractor.named_parameters() if "lora_" in n and p.requires_grad]
    assert lora_grads and any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in lora_grads)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in extractor.residual.parameters())
    env.close()
