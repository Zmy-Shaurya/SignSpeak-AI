"""Shared keypoint layout for recording, training and inference.

Per frame, a (258,) float32 vector: pose 33x(x, y, z, visibility), then left hand 21x(x, y, z),
then right hand 21x(x, y, z). Missing parts are zero-filled so slots never shift.
Face mesh (468 points) is skipped: handshape and position carry most of the sign.
"""
import numpy as np

FEATURES = 33 * 4 + 21 * 3 * 2  # 258
SEQ_LEN = 30  # frames per sign sample
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


def resample(frames, n=SEQ_LEN):
    """(any_len, 258) -> (n, 258) by picking evenly spaced frames, so every clip covers start to end.

    ponytail: picks frames instead of interpolating; short clips repeat frames. Fine for 1-4 s clips.
    """
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return frames[idx]


def extract_keypoints(results, frame):
    """MediaPipe Holistic results -> raw (258,) vector. `frame` is the unmirrored image that was processed.

    MediaPipe gives x as a fraction of width and y of height; x and z are rescaled by width/height so all
    axes share one unit (image height). Otherwise a 4:3 webcam and a 9:16 phone give different shapes.
    """
    aspect = frame.shape[1] / frame.shape[0]
    pose = (
        np.array([[p.x * aspect, p.y, p.z * aspect, p.visibility] for p in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(33 * 4)
    )
    left = (
        np.array([[p.x * aspect, p.y, p.z * aspect] for p in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(21 * 3)
    )
    right = (
        np.array([[p.x * aspect, p.y, p.z * aspect] for p in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(21 * 3)
    )
    return np.concatenate([pose, left, right]).astype(np.float32)


def normalize_keypoints(kp):
    """Make a raw (258,) vector independent of where you stand and how far from the camera.

    x, y become relative to the shoulder midpoint; x, y, z are divided by shoulder width.
    Missing hands stay all-zero. With no pose there is no reference, so the frame becomes all-zero.
    """
    pose = kp[:132].reshape(33, 4).copy()
    hands = kp[132:].reshape(42, 3).copy()

    if not pose.any():
        return np.zeros(FEATURES, dtype=np.float32)

    l, r = pose[LEFT_SHOULDER, :2], pose[RIGHT_SHOULDER, :2]
    center = (l + r) / 2
    scale = max(np.linalg.norm(l - r), 1e-6)

    pose[:, :2] -= center
    pose[:, :3] /= scale  # visibility (column 3) left as-is

    present = hands.reshape(2, 21, 3).any(axis=(1, 2)).repeat(21)
    hands[present, :2] -= center
    hands[present] /= scale

    return np.concatenate([pose.flatten(), hands.flatten()]).astype(np.float32)


if __name__ == "__main__":
    # Self-check: the same pose shifted and zoomed must normalize to the same vector.
    rng = np.random.default_rng(0)
    kp = rng.uniform(0.3, 0.7, FEATURES).astype(np.float32)
    kp[132 + 63:] = 0  # right hand missing

    moved = kp.copy()
    pose, hands = moved[:132].reshape(33, 4), moved[132:132 + 63].reshape(21, 3)
    pose[:, :2] = (pose[:, :2] - 0.5) * 1.5 + [0.55, 0.45]  # zoom in 1.5x and shift
    pose[:, 2] *= 1.5
    hands[:, :2] = (hands[:, :2] - 0.5) * 1.5 + [0.55, 0.45]
    hands[:, 2] *= 1.5

    a, b = normalize_keypoints(kp), normalize_keypoints(moved)
    assert np.allclose(a, b, atol=1e-5), "normalization is not position/scale invariant"
    assert not a[132 + 63:].any(), "missing hand must stay zero"
    assert not normalize_keypoints(np.zeros(FEATURES, np.float32)).any(), "no pose must give zeros"
    seq = np.arange(90)[:, None] * np.ones(FEATURES)
    r = resample(seq)
    assert r.shape == (SEQ_LEN, FEATURES) and r[0, 0] == 0 and r[-1, 0] == 89, "resample must keep first and last frame"
    assert resample(seq[:10]).shape == (SEQ_LEN, FEATURES), "short clips must stretch to SEQ_LEN"
    print("keypoints.py self-check passed")
