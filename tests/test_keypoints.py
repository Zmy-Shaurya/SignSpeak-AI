import numpy as np

from keypoints import FEATURES, SEQ_LEN, normalize_keypoints, resample


def fake_keypoints(seed=0):
    kp = np.random.default_rng(seed).uniform(0.3, 0.7, FEATURES).astype(np.float32)
    kp[132 + 63:] = 0  # right hand missing
    return kp


def test_normalization_ignores_position_and_distance():
    kp = fake_keypoints()
    moved = kp.copy()
    pose, hand = moved[:132].reshape(33, 4), moved[132:132 + 63].reshape(21, 3)
    for part in (pose, hand):
        part[:, :2] = (part[:, :2] - 0.5) * 1.5 + [0.55, 0.45]  # zoom 1.5x and shift
        part[:, 2] *= 1.5
    assert np.allclose(normalize_keypoints(kp), normalize_keypoints(moved), atol=1e-5)


def test_missing_parts_stay_zero():
    assert not normalize_keypoints(fake_keypoints())[132 + 63:].any()
    assert not normalize_keypoints(np.zeros(FEATURES, np.float32)).any()


def test_normalized_shoulders_are_centred_one_unit_apart():
    out = normalize_keypoints(fake_keypoints()).reshape(-1)[:132].reshape(33, 4)
    left, right = out[11, :2], out[12, :2]
    assert np.allclose((left + right) / 2, 0, atol=1e-6)
    assert np.isclose(np.linalg.norm(left - right), 1, atol=1e-5)


def test_resample_keeps_ends_and_length():
    seq = np.arange(90)[:, None] * np.ones(FEATURES)
    out = resample(seq)
    assert out.shape == (SEQ_LEN, FEATURES)
    assert out[0, 0] == 0 and out[-1, 0] == 89
    assert resample(seq[:10]).shape == (SEQ_LEN, FEATURES)
