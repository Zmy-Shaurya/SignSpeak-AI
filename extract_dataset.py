"""Turn every clip in data/videos/<sign>/ into data/keypoints/<sign>/<clip>.npy, shape (SEQ_LEN, 258).

Works the same for webcam recordings, phone videos and dataset videos. Already-extracted clips
are skipped, so just re-run after adding videos. Delete data/keypoints/ to re-extract everything.
Saved keypoints are RAW; normalize when loading for training.
"""
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from keypoints import extract_keypoints, resample

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_SECONDS = 6  # about 2x the live window (live_predict --seconds 3)

videos = sorted(p for p in Path("data/videos").glob("*/*") if p.suffix.lower() in VIDEO_EXTS)
print(f"Found {len(videos)} videos.")

for video in videos:
    out = Path("data/keypoints") / video.parent.name / f"{video.stem}.npy"
    if out.exists():
        continue

    frames = []
    cap = cv2.VideoCapture(str(video))
    seconds = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    if seconds > MAX_SECONDS:
        # Squeezing a long clip into SEQ_LEN frames leaves only a few frames of the actual sign.
        print(f"SKIP {video}: {seconds:.1f} s long (max {MAX_SECONDS}); trim it to just the sign")
        cap.release()
        continue
    # Fresh Holistic per clip so tracking from one clip never leaks into the next.
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(extract_keypoints(results, frame))
    cap.release()

    if not frames:
        print(f"SKIP {video}: could not read any frames")
        continue
    seq = resample(np.stack(frames))
    with_pose = int(seq[:, :132].any(axis=1).sum())
    if with_pose < len(seq) // 2:
        print(f"SKIP {video}: body found in only {with_pose}/{len(seq)} frames (shoulders must be visible)")
        continue

    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, seq)
    print(f"OK   {video} -> {out} ({len(frames)} frames)")
