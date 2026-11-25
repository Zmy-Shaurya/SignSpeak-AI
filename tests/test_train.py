import numpy as np
import torch

from keypoints import FEATURES, SEQ_LEN
from train import SignModel, augment, class_weights


def test_model_output_shape():
    model = SignModel(5).eval()
    assert model(torch.zeros(3, SEQ_LEN, FEATURES)).shape == (3, 5)


def test_augment_keeps_shape_and_missing_hands():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, SEQ_LEN, FEATURES)).astype(np.float32)
    X[..., 195:] = 0  # right hand missing
    out = augment(X, rng)
    assert out.shape == X.shape and out.dtype == np.float32
    assert not out[..., 195:].any()


def test_class_weights():
    y = np.array([0] * 10 + [1] * 30)
    assert np.allclose(class_weights(y, ["a", "b"]), [1, 1])
    w = class_weights(y, ["a", "b"], "balanced")
    assert w[0] == 3 * w[1] and np.isclose(w.mean(), 1)
    w = class_weights(y, ["a", "b"], "none", {"a": 2})
    assert w[0] == 2 * w[1] and np.isclose(w.mean(), 1)
