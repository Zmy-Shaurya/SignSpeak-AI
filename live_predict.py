"""Live sign prediction in an OpenCV window (the web UI is server.py). Recognition logic lives in engine.py.

Usage: python live_predict.py [--camera 0] [--seconds 3] [--threshold 0.95] [--hold 5] [--pause 2] [--stride 2]
Keeps the last --seconds of frames (match record_videos.py), resamples to 30 like training, predicts every
--stride frames. A sign is accepted once it wins --hold predictions in a row; then the window is cleared so the
same movement can't be counted twice (that refill time is the cooldown). Accepted words build up at the bottom.
After --pause seconds of a full window showing nothing, the words become a sentence (Gemini if GEMINI_API_KEY is
set, else plain) and are spoken aloud on a background thread. c = clear words, q = quit.
"""
import argparse
import os
import queue
import threading

import cv2

from engine import Recognizer, load_model
from sentence import MODEL, make_speaker, to_sentence

parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--seconds", type=float, default=3.0, help="window length; same as clip length used for recording")
parser.add_argument("--threshold", type=float, default=0.95, help="minimum confidence to show a sign")
parser.add_argument("--hold", type=int, default=5, help="predictions in a row a sign must win before it is accepted")
parser.add_argument("--pause", type=float, default=2.0, help="seconds of no signing that end a sentence")
parser.add_argument("--stride", type=int, default=2, help="analyse every Nth frame (1 = every frame)")
parser.add_argument("--width", type=int, default=1280, help="window width in pixels; drag or maximize to resize")
parser.add_argument("--model", default=MODEL, help="Gemini model used to phrase the sentence")
args = parser.parse_args()

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


model, labels = load_model()
rec = Recognizer(model, labels, sentences.put, seconds=args.seconds, threshold=args.threshold,
                 hold=args.hold, pause=args.pause, stride=args.stride)
threading.Thread(target=speech_worker, daemon=True).start()
print("Sentences:", "Gemini " + args.model if os.environ.get("GEMINI_API_KEY") else "plain (set GEMINI_API_KEY for Gemini)")

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera {args.camera}.")
# Resizable window, opened larger. Only the window is scaled, not the frame (live_vcam.py sends the frame as-is).
cv2.namedWindow("SignSpeak AI - Live", cv2.WINDOW_NORMAL)
cv2.resizeWindow("SignSpeak AI - Live", args.width, int(args.width * cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

print(f"Signs: {', '.join(labels)}. Press q to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break

    state = rec.process(frame)  # unmirrored, as in training; draws the landmarks onto frame
    frame = cv2.flip(frame, 1)

    if state["warming"]:
        cv2.putText(frame, "warming up...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        cv2.putText(frame, state["label"] or "...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
        for i, (l, p) in enumerate(state["probs"].items()):
            cv2.putText(frame, f"{l}: {p:.0%}", (20, 90 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if not state["quality"]["body"]:
        cv2.putText(frame, "shoulders not in frame", (20, frame.shape[0] - 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.putText(frame, spoken["text"], (20, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, " ".join(rec.get_words()[-6:]), (20, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
    cv2.imshow("SignSpeak AI - Live", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c"):
        rec.set_words([])

rec.close()
cap.release()
cv2.destroyAllWindows()
