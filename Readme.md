# SignSpeak AI

Real-time **American Sign Language (ASL)** recognition from an ordinary webcam. SignSpeak AI watches you sign, tracks your body and hands with MediaPipe, and classifies the sign with a small LSTM neural network, live on a laptop CPU.

> Semester mini project. Goal: webcam → sign recognition → text → speech.
> Status: recognition works end to end; sentence assembly, text-to-speech and a web UI are next (see [Roadmap](#roadmap)).

---

## Contents

- [How it works](#how-it-works)
- [Current vocabulary and results](#current-vocabulary-and-results)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start: run the live demo](#quick-start-run-the-live-demo)
- [Full pipeline: train your own model](#full-pipeline-train-your-own-model)
- [Adding data from other people or datasets](#adding-data-from-other-people-or-datasets)
- [Project structure](#project-structure)
- [Design decisions](#design-decisions)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## How it works

```
 webcam / video
      │
      ▼
 MediaPipe Holistic ──► 258 numbers per frame
 (body + both hands)     • 33 body points × (x, y, z, visibility)
                         • 21 left-hand points × (x, y, z)
                         • 21 right-hand points × (x, y, z)
      │
      ▼
 Normalize ──► relative to the shoulders, scaled by shoulder width
               (works no matter where you stand or how far from the camera)
      │
      ▼
 Resample ──► 30 evenly spaced frames per sign
      │
      ▼
 LSTM classifier ──► one probability per sign ──► "hello" (94%)
```

- **Keypoints, not pixels.** The model never sees the image, only body and hand coordinates. That makes it fast, lighting-independent, and trainable on a few hundred clips.
- **A sequence model.** Most signs are movements, not poses, so an LSTM reads the 30 frames in order.
- **A `nothing` class.** The model is trained on idle movement *and* on 50 unrelated ASL signs labelled `nothing`, so it learns to stay quiet instead of forcing every gesture into a known word.

## Current vocabulary and results

| Class | Meaning |
|---|---|
| `hello`, `help`, `no`, `please`, `sorry`, `thank_you`, `yes` | ASL signs |
| `nothing` | no sign / unknown sign (idle movement plus 50 unrelated ASL signs) |

Trained on about 356 clips: the author's own recordings, a second signer, and about 30 clips per sign from the [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) dataset (42 different signers, everyday webcams).

**Accuracy on held-out test clips: about 85%** (mean over 3 random splits; individual splits range from 75% to 90%). The most common confusions are between visually similar signs (please ↔ sorry, yes ↔ no) and the classes with the fewest clips.

## Requirements

- **Python 3.10 or 3.11.** Avoid 3.12+, because `mediapipe==0.10.14` doesn't support it.
- A **webcam** for recording and the live demo.
- Windows, macOS or Linux. Developed and tested on Windows 11; no GPU needed.

## Installation

```bash
git clone https://github.com/Zmy-Shaurya/SignSpeak-AI.git
cd SignSpeak-AI

# create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# PyTorch (CPU build, much smaller than the default GPU build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# everything else
pip install -r requirements.txt
```

Check it works:

```bash
python keypoints.py
```

This should print `keypoints.py self-check passed`.

> **Note:** `mediapipe` is pinned to `0.10.14` on purpose, because newer releases removed the `mp.solutions` API this project uses. MediaPipe installs `opencv-contrib-python` itself. Don't also install `opencv-python`, because the two conflict.

## Quick start: run the live demo

A trained model is included at `models/sign_model.pt`.

```bash
python live_predict.py --camera 0
```

- Stand **waist-up in frame, with both shoulders visible**.
- Wait for "warming up…" to disappear (about 3 seconds), then sign.
- The big text shows the recognized sign; `...` means nothing or not confident. The small text shows every class's probability.
- Press **`q`** to quit.

| Option | Default | Meaning |
|---|---|---|
| `--camera` | `0` | Webcam index. Try `1` or `2` if `0` isn't your camera. |
| `--seconds` | `3.0` | Length of the sliding window. Keep it equal to the recording length. |
| `--threshold` | `0.7` | Minimum confidence before a sign is shown. |

The prediction looks at the last 3 seconds as a whole, so a sign appears shortly after you finish it and stays for up to 3 seconds.

## Full pipeline: train your own model

Four commands, each run from the project root.

### 1. Record clips

```bash
python record_videos.py hello --camera 0
```

- Press **SPACE**. You get a 2-second countdown, then it records 3 seconds automatically, so your hands stay free.
- Do one sign per clip: hands down, sign, hands down.
- Record **20–30 clips per sign**, varying speed, position and distance slightly.
- Record a `nothing` class too: standing still, scratching your face, fidgeting.
- Press **`q`** when done. Clips are saved to `data/videos/<sign>/`.

The folder name **is** the label. To add a new sign, just record into a new name (use `snake_case`, for example `thank_you`).

### 2. Extract keypoints

```bash
python extract_dataset.py
```

Every clip in `data/videos/<sign>/` becomes `data/keypoints/<sign>/<clip>.npy`, a `(30, 258)` array.

- **Incremental:** already-extracted clips are skipped, so just re-run after adding videos.
- **Quality checks:** clips are skipped (with a reason printed) if your shoulders aren't visible in most frames or the clip is longer than 6 seconds.
- **Re-extracting:** to redo everything, delete `data/keypoints/` and run again.

### 3. Train

```bash
python train.py
```

This prints the training/test accuracy every 10 epochs and a confusion matrix at the end, then saves `models/sign_model.pt`. 20% of each sign is held out for testing. It takes about 1–2 minutes on a laptop CPU.

### 4. Run it live

```bash
python live_predict.py --camera 0
```

## Adding data from other people or datasets

`extract_dataset.py` handles any video (`.mp4 .mov .avi .mkv .webm`) the same way as webcam recordings: put clips into `data/videos/<sign>/` and run steps 2–3.

**Videos from friends (phone)**
- Prop the phone up and film from the front, **waist up, both shoulders in frame**.
- One sign per video, about 3 seconds.
- Send it via Google Drive, or WhatsApp **as a document**, so it isn't compressed.
- Prefix filenames with the person's name (`alex_hello_1.mp4`) to avoid collisions.

**Watch out for mirrored video.** Front cameras often save mirrored footage, which swaps left and right hands. Check a frame for backwards text, or check whether a right-handed signer shows up as left-hand only. Flip such clips before extracting. Left-handed signers can be flipped the same way so that every clip uses the same dominant hand.

**Public datasets.** This project used [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) (Microsoft Research). Its license allows **non-commercial research use only** and forbids redistributing the data or anything derived from it. That's why ASL Citizen videos and keypoints are **not** in this repository (`aslc_*` files are gitignored). To use it, download it yourself from Microsoft and copy the clips for your signs into `data/videos/<sign>/`.

## Project structure

```
SignSpeak-AI/
├── record_videos.py      1. record webcam clips      -> data/videos/<sign>/
├── extract_dataset.py    2. clips -> keypoints       -> data/keypoints/<sign>/
├── train.py              3. train the LSTM           -> models/sign_model.pt
├── live_predict.py       4. live webcam recognition
├── keypoints.py          shared: keypoint layout, normalization, resampling (+ self-check)
├── models/
│   └── sign_model.pt     trained model (weights + label names)
├── data/
│   ├── videos/           raw clips (gitignored: faces, size)
│   └── keypoints/        extracted (30, 258) arrays
├── learning/             Phase 1 exercises (webcam, hand landmarks, Holistic demo)
└── requirements.txt
```

The `learning/` scripts are standalone exercises. Run step 4 as a module from the project root: `python -m learning.step4_holistic_keypoints`.

## Design decisions

| Decision | Why |
|---|---|
| **MediaPipe Holistic** (body + hands) instead of hands only | Where a sign happens (chin, chest) matters, and Holistic labels hands left/right consistently. |
| **Face mesh left out** | It would add 1,404 numbers per frame for little gain at this vocabulary size. |
| **Process the unmirrored frame** and mirror only for display | Holistic's left/right labels assume a normal camera image; feeding it a mirrored one swaps the hands. |
| **x and z rescaled by width/height** | Keypoints from 4:3 webcams and 9:16 phones end up in the same units. |
| **Save raw keypoints, normalize at load time** | Normalization can change without re-extracting every video. |
| **Keep raw videos** | Extraction can be redone (for example to add face points) without re-recording. |
| **PyTorch** instead of TensorFlow | TensorFlow conflicts with MediaPipe's protobuf version. |
| **Data augmentation and dropout** | Random speed, rotation, scale and shift every epoch. It raised the mean test accuracy from 77% to 85% on the same splits. |

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open camera 0` | Try `--camera 1` or `--camera 2`, and close other apps using the webcam. |
| `module 'mediapipe' has no attribute 'solutions'` | Wrong MediaPipe version: `pip install mediapipe==0.10.14`. |
| `cv2` errors or a blank window | Uninstall `opencv-python` and keep only `opencv-contrib-python`. |
| A sign is never recognized | Check that your shoulders are in frame, record more clips of yourself doing that sign, and retrain. |
| Random movements trigger a sign | Record more varied `nothing` clips, or raise `--threshold`. |
| Many `SKIP` lines during extraction | Shoulders are out of frame, or clips are over 6 seconds; re-record or trim them. |

## Limitations

- **Isolated words only.** It recognizes one sign at a time, not continuous signing or ASL grammar.
- **Small vocabulary** (7 signs plus `nothing`), with about 30–50 clips each.
- **No facial expressions.** Signs that differ only by facial grammar can't be distinguished.
- **Latency:** predictions use a 3-second window, so results trail the sign by up to about 3 seconds.
- **Test accuracy varies ±10% between random splits** because the test set is small (about 70 clips).

## Roadmap

- [x] Webcam capture and MediaPipe Holistic keypoints
- [x] Normalization, recording tool, dataset extraction
- [x] LSTM training with augmentation; live recognition
- [x] Public dataset integration (ASL Citizen) and an unknown-sign class
- [ ] Debounce/cooldown so each sign is accepted once
- [ ] Sentence assembly (a pause ends the sentence)
- [ ] Text-to-speech (offline, `pyttsx3`)
- [ ] Web UI (Streamlit / Gradio) with live captions and confidence
- [ ] Larger vocabulary and more signers


## License

The **code** in this repository is released under the [MIT License](LICENSE).

Third-party data keeps its own license: ASL Citizen is **not** included and is licensed by Microsoft Research for non-commercial research use only. If you train on it, you're bound by its terms.

## Acknowledgements

- [MediaPipe](https://github.com/google-ai-edge/mediapipe) (Google): body and hand landmark detection
- [PyTorch](https://pytorch.org/): model training
- [OpenCV](https://opencv.org/): video capture and processing
- [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) (Microsoft Research): Desai et al., *ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition*, NeurIPS 2023 Datasets and Benchmarks
- Sign references: [Lifeprint / ASLU](https://www.lifeprint.com/), [Handspeak](https://www.handspeak.com/), [SignASL](https://www.signasl.org/)
