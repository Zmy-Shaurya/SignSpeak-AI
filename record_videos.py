"""Record sign clips from the webcam into data/videos/<sign>/.

Usage: python record_videos.py hello [--camera 2] [--seconds 3]
SPACE starts a 1 s countdown (get your hands ready), then records --seconds automatically. q quits.
"""
import argparse
import time
from pathlib import Path

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("sign", help="label, used as the folder name, e.g. hello")
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--width", type=int, default=720, help="window width in pixels; drag or maximize to resize")
parser.add_argument("--seconds", type=float, default=3.0, help="clip length; long enough for your slowest sign")
args = parser.parse_args()

out_dir = Path("data/videos") / args.sign
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera {args.camera}.")
cv2.namedWindow("SignSpeak AI - Recorder", cv2.WINDOW_NORMAL)  # resizable; only the window is scaled, not the saved clip
cv2.resizeWindow("SignSpeak AI - Recorder", args.width, int(args.width * cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # only affects playback speed; extraction resamples anyway

countdown_end = record_end = None
writer = None
saved = len(list(out_dir.glob("*.mp4")))
print(f"Recording '{args.sign}' ({saved} clips so far). SPACE = record, q = quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break
    now = time.time()

    if countdown_end and now >= countdown_end:
        countdown_end = None
        record_end = now + args.seconds
        path = out_dir / f"{args.sign}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    if writer:
        writer.write(frame)  # raw, unmirrored: Holistic's left/right labels depend on it
        if now >= record_end:
            writer.release()
            writer = None
            saved += 1
            print(f"Saved {path} ({saved} total)")

    display = cv2.flip(frame, 1)
    if countdown_end:
        msg, color = f"Get ready... {countdown_end - now:.1f}", (0, 255, 255)
    elif writer:
        msg, color = f"RECORDING {record_end - now:.1f}", (0, 0, 255)
    else:
        msg, color = f"'{args.sign}': {saved} clips. SPACE = record, q = quit", (0, 255, 0)
    cv2.putText(display, msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("SignSpeak AI - Recorder", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord(" ") and not countdown_end and not writer:
        countdown_end = now + 1

if writer:  # quit mid-recording: drop the partial clip
    writer.release()
    path.unlink()
cap.release()
cv2.destroyAllWindows()
