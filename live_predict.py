"""Live sign prediction from the webcam using models/sign_model.pt.

Usage: python live_predict.py [--camera 0] [--seconds 3] [--threshold 0.7] [--hold 5]
Keeps the last --seconds of frames (match record_videos.py), resamples to 30 like training, predicts every frame.
A sign is accepted once it wins --hold frames in a row; then the window is cleared so the same movement
can't be counted twice (that refill time is the cooldown). Accepted words build up at the bottom.
After --pause seconds of a full window showing nothing, the words become a sentence (Gemini if GEMINI_API_KEY is
set, else plain) and are spoken aloud on a background thread. c = clear words, q = quit.
"""
import argparse
import os
import queue
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch

from keypoints import extract_keypoints, normalize_keypoints, resample
from sentence import MODEL, make_speaker, to_sentence
from train import SignModel

parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--seconds", type=float, default=3.0, help="window length; same as clip length used for recording")
parser.add_argument("--threshold", type=float, default=0.95, help="minimum confidence to show a sign")
parser.add_argument("--hold", type=int, default=5, help="frames in a row a sign must win before it is accepted")
parser.add_argument("--width", type=int, default=1280, help="window width in pixels; drag or maximize to resize")
parser.add_argument("--pause", type=float, default=2.0, help="seconds of no signing that end a sentence")
parser.add_argument("--model", default=MODEL, help="Gemini model used to phrase the sentence")
args = parser.parse_args()

checkpoint = torch.load("models/sign_model.pt")
labels = checkpoint["labels"]
model = SignModel(len(labels))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
window = deque()  # (timestamp, normalized keypoints)
streak_label, streak = None, 0
words = []
idle_since = None  # when the full window started showing nothing; the pause timer
spoken = {"text": ""}  # last sentence, written by the speech thread, drawn by the video loop
sentences = queue.Queue()


def speech_worker():
    speak = make_speaker()  # created on this thread (COM rule)
    while True:
        batch = sentences.get()
        spoken["text"] = "..."
        spoken["text"] = to_sentence(batch, args.model)
        print("Sentence:", spoken["text"])
        speak(spoken["text"])


threading.Thread(target=speech_worker, daemon=True).start()
print("Sentences:", "Gemini " + args.model if os.environ.get("GEMINI_API_KEY") else "plain (set GEMINI_API_KEY for Gemini)")

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera {args.camera}.")
# Resizable window, opened larger. Only the window is scaled, not the frame (live_vcam.py sends the frame as-is).
cv2.namedWindow("SignSpeak AI - Live", cv2.WINDOW_NORMAL)
cv2.resizeWindow("SignSpeak AI - Live", args.width, int(args.width * cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    print(f"Signs: {', '.join(labels)}. Press q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # unmirrored, as in training
        now = time.time()
        window.append((now, normalize_keypoints(extract_keypoints(results, frame))))
        while now - window[0][0] > args.seconds:
            window.popleft()

        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        frame = cv2.flip(frame, 1)

        if now - window[0][0] > args.seconds * 0.9:  # wait until the window is (nearly) full
            seq = resample(np.stack([kp for _, kp in window]))
            with torch.no_grad():
                probs = torch.softmax(model(torch.tensor(seq)[None]), 1)[0]
            conf, idx = probs.max(0)
            label = labels[idx] if conf >= args.threshold and labels[idx] != "nothing" else None
            streak = streak + 1 if label and label == streak_label else 1
            streak_label = label
            if label and streak >= args.hold:
                words.append(label)
                window.clear()  # cooldown: refill with fresh frames, so this sign can't fire again
                streak_label, streak = None, 0
            idle_since = None if label else (idle_since or now)
            if words and idle_since and now - idle_since >= args.pause:  # signer paused: sentence done
                sentences.put(words.copy())
                words.clear()
                idle_since = None
            cv2.putText(frame, label or "...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
            for i, (l, p) in enumerate(zip(labels, probs)):
                cv2.putText(frame, f"{l}: {p:.0%}", (20, 90 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            idle_since = None  # can't recognize anything while the window refills, so that's not a pause
            cv2.putText(frame, "warming up...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.putText(frame, spoken["text"], (20, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, " ".join(words[-6:]), (20, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        cv2.imshow("SignSpeak AI - Live", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            words.clear()

cap.release()
cv2.destroyAllWindows()
