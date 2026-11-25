"""Recognition engine: frames in, sign/word/sentence state out. No camera, window or web code in here.

Rolling window -> resample to 30 -> LSTM -> hold N frames -> pause ends sentence. Used by both front ends:
server.py (web UI) and live_predict.py (OpenCV window). Settings can change while it runs.
Self-check (needs models/sign_model.pt, no camera): python engine.py. Unit tests: pytest
"""
import hashlib
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from keypoints import LEFT_SHOULDER, RIGHT_SHOULDER, extract_keypoints, normalize_keypoints, resample
from train import SignModel

MODEL_PATH = "models/sign_model.pt"
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils


def load_model(path=MODEL_PATH):
    checkpoint = torch.load(path)
    model = SignModel(len(checkpoint["labels"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint["labels"]


def model_info(path=MODEL_PATH):
    """Version of the checkpoint on disk: content hash (changes on every retrain) and when it was saved."""
    p = Path(path)
    return {
        "version": hashlib.sha256(p.read_bytes()).hexdigest()[:12],
        "trained": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
    }


def tracking_quality(results):
    """What Holistic sees this frame. The model needs both shoulders (normalization reference) and the hands."""
    pose = results.pose_landmarks
    body = bool(pose) and all(pose.landmark[i].visibility > 0.5 for i in (LEFT_SHOULDER, RIGHT_SHOULDER))
    return {"body": body, "left_hand": results.left_hand_landmarks is not None,
            "right_hand": results.right_hand_landmarks is not None}


class Recognizer:
    """Settings (threshold, hold, pause, seconds, stride) are plain attributes: change them any time.

    stride: run Holistic + the model on every Nth frame only (2 = every other frame), the costly part.
    Skipped frames reuse the last landmarks and prediction, so video stays smooth while CPU use drops.
    `hold` counts processed frames, so with stride 2 a sign must hold twice as many camera frames.
    `words` is shared with other threads (the web API edits it), so it is only touched under `_lock`.
    """

    def __init__(self, model, labels, on_sentence, seconds=3.0, threshold=0.95, hold=5, pause=2.0, stride=2):
        self.model, self.labels, self.on_sentence = model, labels, on_sentence
        self.seconds, self.threshold, self.hold, self.pause, self.stride = seconds, threshold, hold, pause, stride
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.words = []
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        """Forget the window (e.g. after switching camera); accepted words are kept."""
        self.window = deque()  # (timestamp, normalized keypoints)
        self.streak_label, self.streak, self.idle_since = None, 0, None
        self._frame_count, self._last_results, self._last_state, self.inferred = 0, None, None, False

    def close(self):
        self.holistic.close()

    # --- word list, safe to call from any thread ----------------------------------------------------------------
    def get_words(self):
        with self._lock:
            return list(self.words)

    def set_words(self, words):
        """Replace the words (a hand edit). Restarts the pause timer so the sentence isn't sent mid-edit."""
        with self._lock:
            self.words[:] = words
        self.idle_since = None

    def finish(self):
        """End the sentence now instead of waiting for a pause. Returns False if there were no words."""
        with self._lock:
            words, self.words[:] = list(self.words), []
        self.idle_since = None
        if words:
            self.on_sentence(words)
        return bool(words)

    # --- recognition -------------------------------------------------------------------------------------------
    def process(self, frame):
        """Draw landmarks onto `frame` (unmirrored, in place) and return the current prediction state."""
        self._frame_count += 1
        skip = self._last_state is not None and (self._frame_count - 1) % max(1, self.stride) != 0
        self.inferred = not skip  # read by the server so latency metrics only count real inference
        if skip:  # reuse the previous frame's landmarks and prediction
            self._draw(frame, self._last_results)
            return self._last_state

        results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # unmirrored, as in training
        now = time.time()
        self.window.append((now, normalize_keypoints(extract_keypoints(results, frame))))
        while now - self.window[0][0] > self.seconds:
            self.window.popleft()
        self._draw(frame, results)

        if now - self.window[0][0] <= self.seconds * 0.9:  # window still (re)filling: nothing to recognize
            self.idle_since = None  # so that is not a pause
            state = {"warming": True, "label": None, "confidence": 0.0, "probs": {}, "streak": 0, "idle": 0.0}
        else:
            seq = resample(np.stack([kp for _, kp in self.window]))
            with torch.no_grad():
                probs = torch.softmax(self.model(torch.tensor(seq)[None]), 1)[0].numpy()
            state = self.update(probs, now)
        state["quality"] = tracking_quality(results)
        self._last_results, self._last_state = results, state
        return state

    @staticmethod
    def _draw(frame, results):
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    def update(self, probs, now):
        """Decide from one frame's class probabilities: show, accept (after `hold` wins), or end the sentence.

        Pure bookkeeping, no MediaPipe or model, so tests can drive it with made-up probabilities and times.
        """
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = self.labels[idx] if conf >= self.threshold and self.labels[idx] != "nothing" else None
        self.streak = self.streak + 1 if label and label == self.streak_label else 1
        self.streak_label = label
        if label and self.streak >= self.hold:
            with self._lock:
                self.words.append(label)
            self.window.clear()  # cooldown: refill with fresh frames, so this sign can't fire again
            self.streak_label, self.streak = None, 0
        self.idle_since = None if label else (self.idle_since or now)
        if self.idle_since and now - self.idle_since >= self.pause and self.get_words():  # signer paused
            self.finish()
        return {
            "warming": False,
            "label": label,
            "confidence": conf,
            "streak": self.streak if label else 0,  # frames the sign has held so far, for the hold-progress bar
            "idle": now - self.idle_since if self.idle_since else 0.0,  # seconds of pause so far
            "probs": {l: float(p) for l, p in zip(self.labels, probs)},
        }


if __name__ == "__main__":
    model, labels = load_model()
    sentences = []
    rec = Recognizer(model, labels, sentences.append, seconds=1.5)
    blank = np.zeros((480, 640, 3), np.uint8)
    state = rec.process(blank.copy())
    assert state["warming"] and state["label"] is None
    assert state["quality"] == {"body": False, "left_hand": False, "right_hand": False}, state["quality"]
    assert rec.process(blank.copy()) is state, "stride 2: the second frame must reuse the first one's state"
    end = time.time() + 10
    while state["warming"] and time.time() < end:  # keep feeding frames until the 1.5 s window is full
        state = rec.process(blank.copy())
    assert not state["warming"] and set(state["probs"]) == set(labels), state
    assert abs(sum(state["probs"].values()) - 1) < 1e-4
    rec.close()
    print("engine.py self-check passed")
