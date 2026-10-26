"""Recognition engine: frames in, sign/word/sentence state out. No camera, window or web code in here.

Same logic as live_predict.py (rolling window -> resample to 30 -> LSTM -> hold N frames -> pause ends sentence),
packaged as a class so the web server (server.py) can drive it and settings can change while it runs.
Self-check (needs models/sign_model.pt, no camera): python engine.py
"""
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch

from keypoints import extract_keypoints, normalize_keypoints, resample
from train import SignModel

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils


def load_model(path="models/sign_model.pt"):
    checkpoint = torch.load(path)
    model = SignModel(len(checkpoint["labels"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint["labels"]


class Recognizer:
    """Settings (threshold, hold, pause, seconds) are plain attributes: change them any time."""

    def __init__(self, model, labels, on_sentence, seconds=3.0, threshold=0.95, hold=5, pause=2.0):
        self.model, self.labels, self.on_sentence = model, labels, on_sentence
        self.seconds, self.threshold, self.hold, self.pause = seconds, threshold, hold, pause
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.words = []
        self.reset()

    def reset(self):
        """Forget the window (e.g. after switching camera); accepted words are kept."""
        self.window = deque()  # (timestamp, normalized keypoints)
        self.streak_label, self.streak, self.idle_since = None, 0, None

    def close(self):
        self.holistic.close()

    def process(self, frame):
        """Draw landmarks onto `frame` (unmirrored, in place) and return the current prediction state."""
        results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # unmirrored, as in training
        now = time.time()
        self.window.append((now, normalize_keypoints(extract_keypoints(results, frame))))
        while now - self.window[0][0] > self.seconds:
            self.window.popleft()
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        if now - self.window[0][0] <= self.seconds * 0.9:  # window still (re)filling: nothing to recognize
            self.idle_since = None  # so that is not a pause
            return {"warming": True, "label": None, "confidence": 0.0, "probs": {}}

        seq = resample(np.stack([kp for _, kp in self.window]))
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.tensor(seq)[None]), 1)[0]
        conf, idx = probs.max(0)
        label = self.labels[idx] if conf >= self.threshold and self.labels[idx] != "nothing" else None
        self.streak = self.streak + 1 if label and label == self.streak_label else 1
        self.streak_label = label
        if label and self.streak >= self.hold:
            self.words.append(label)
            self.window.clear()  # cooldown: refill with fresh frames, so this sign can't fire again
            self.streak_label, self.streak = None, 0
        self.idle_since = None if label else (self.idle_since or now)
        if self.words and self.idle_since and now - self.idle_since >= self.pause:  # signer paused: sentence done
            self.on_sentence(self.words.copy())
            self.words.clear()
            self.idle_since = None
        return {
            "warming": False,
            "label": label,
            "confidence": float(conf),
            "probs": {l: float(p) for l, p in zip(self.labels, probs)},
        }


if __name__ == "__main__":
    model, labels = load_model()
    sentences = []
    rec = Recognizer(model, labels, sentences.append, seconds=0.5)
    blank = np.zeros((480, 640, 3), np.uint8)
    state = rec.process(blank.copy())
    assert state["warming"] and state["label"] is None
    end = time.time() + 1.0
    while time.time() < end:  # keep feeding frames until the 0.5 s window is full
        state = rec.process(blank.copy())
    assert not state["warming"] and set(state["probs"]) == set(labels), state
    assert abs(sum(state["probs"].values()) - 1) < 1e-4
    rec.close()
    print("engine.py self-check passed")
