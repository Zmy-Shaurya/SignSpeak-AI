"""SignSpeak AI web server: camera + recognition in a worker thread, a browser UI on top.

Usage (from the repo root): python server.py [--camera 0] [--backend any|dshow|msmf] [--port 8000]
Then open http://localhost:8000. Interactive API docs: http://localhost:8000/docs

  GET  /video           MJPEG stream of the annotated camera feed
  GET  /api/events      server-sent events: live state, 10 per second
  GET  /api/state       the same state, once
  GET  /api/health      liveness plus camera status
  GET  /api/metrics     fps, inference latency, counters, model version
  GET  /api/cameras     probe camera indices 0-4
  POST /api/camera      {"index": n}  switch camera at runtime
  POST /api/settings    {"threshold", "hold", "pause", "stride", "mute"}  all optional
  PUT  /api/words       {"words": [...]}  replace the word list (delete, reorder, add)
  POST /api/speak       end the sentence now: phrase and speak the current words
  POST /api/repeat      speak the last sentence again
  POST /api/clear       drop the accepted words
"""
import argparse
import asyncio
import json
import logging
import queue
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import Recognizer, load_model, model_info
from sentence import MODEL, make_speaker, to_sentence

BACKENDS = {"any": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
WEB = Path(__file__).with_name("web")
log = logging.getLogger("signspeak")


def placeholder(title, detail=""):
    """Frame shown instead of video when there is no usable camera picture."""
    img = np.full((360, 640, 3), 24, np.uint8)
    cv2.putText(img, title, (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(img, detail, (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)
    return img


def jpeg(img):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


class Pipeline:
    """Owns the camera, the recognizer and the speech thread. Everything the API reads goes through state()."""

    def __init__(self, camera, backend):
        self.camera, self.backend = camera, BACKENDS[backend]
        model, self.labels = load_model()
        self.model = {**model_info(), "classes": len(self.labels)}
        self.recognizer = Recognizer(model, self.labels, self._on_sentence)
        self.mute, self.sentence = False, ""
        self.status, self.message, self.fps, self.camera_open = "starting", "Opening camera", 0.0, False
        self.started, self.frames, self.sentences_spoken = time.time(), 0, 0
        self.latency_ms = deque(maxlen=300)  # Holistic + model time per processed frame (skipped frames excluded)
        self._result = {"warming": True, "label": None, "confidence": 0.0, "probs": {}}
        self._frame, self._frame_no = jpeg(placeholder("Starting camera...")), 0
        self._speech = queue.Queue()  # ("words", [...]) to phrase and speak, or ("say", text) to speak as-is
        self._reopen, self._stop = threading.Event(), threading.Event()
        self._reopen.set()
        threading.Thread(target=self._speech_worker, daemon=True).start()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        log.info("model %s (%d classes), camera %d", self.model["version"], len(self.labels), camera)

    # --- public API -------------------------------------------------------------------------------------------
    def switch_camera(self, index):
        log.info("switching to camera %d", index)
        self.camera = index
        self._reopen.set()

    def repeat(self):
        if self.sentence and self.sentence != "...":
            self._speech.put(("say", self.sentence))
            return True
        return False

    def state(self):
        r, rec = self._result, self.recognizer
        return {
            "camera": {"index": self.camera, "status": self.status, "message": self.message, "fps": round(self.fps, 1)},
            "warming": r["warming"], "label": r["label"], "confidence": r["confidence"], "probs": r["probs"],
            "streak": r.get("streak", 0), "idle": r.get("idle", 0.0), "quality": r.get("quality"),
            "labels": self.labels, "words": rec.get_words(), "sentence": self.sentence,
            "settings": {"threshold": rec.threshold, "hold": rec.hold, "pause": rec.pause, "stride": rec.stride,
                         "mute": self.mute},
            "model": self.model,
        }

    def metrics(self):
        lat = np.array(self.latency_ms) if self.latency_ms else np.zeros(1)
        return {
            "uptime_s": round(time.time() - self.started, 1),
            "camera_status": self.status,
            "fps": round(self.fps, 1),
            "frames_inferred": self.frames,
            "inference_ms": {"mean": round(float(lat.mean()), 1), "p50": round(float(np.percentile(lat, 50)), 1),
                             "p95": round(float(np.percentile(lat, 95)), 1), "samples": len(self.latency_ms)},
            "sentences_spoken": self.sentences_spoken,
            "model": self.model,
        }

    def latest_frame(self):
        return self._frame, self._frame_no

    def scan(self, limit=5):
        """Probe camera indices. The active camera is already open here, so it can't be probed again."""
        found = []
        for i in range(limit):
            if i == self.camera and self.camera_open:
                found.append({"index": i, "available": True, "active": True})
                continue
            cap = cv2.VideoCapture(i, self.backend)
            found.append({"index": i, "available": cap.isOpened(), "active": i == self.camera})
            cap.release()
        return found

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)
        self.recognizer.close()

    # --- worker threads ---------------------------------------------------------------------------------------
    def _publish(self, img):
        self._frame, self._frame_no = jpeg(img), self._frame_no + 1

    def _set_status(self, status, message=""):
        if status != self.status:
            (log.info if status in ("live", "starting") else log.warning)("camera %d: %s %s", self.camera, status, message)
        self.status, self.message = status, message

    def _on_sentence(self, words):
        self._speech.put(("words", words))

    def _speech_worker(self):
        speak = None  # created on first use, on this thread (COM rule); keeps startup and tests free of audio
        while True:
            kind, text = self._speech.get()
            if kind == "words":
                self.sentence = "..."
                self.sentence = text = to_sentence(text, MODEL)
                log.info("sentence: %s", text)
            if self.mute:
                continue
            try:
                speak = speak or make_speaker()
                speak(text)
                self.sentences_spoken += 1
            except Exception:  # no audio device / SAPI voice: keep showing text, never kill the thread
                log.exception("speech failed")

    def _capture_loop(self):
        cap, count, t0 = None, 0, time.time()
        while not self._stop.is_set():
            if self._reopen.is_set():
                self._reopen.clear()
                if cap:
                    cap.release()
                self.camera_open, self.fps = False, 0.0
                index = self.camera
                self._set_status("starting", f"Opening camera {index}")
                self._publish(placeholder(f"Opening camera {index}..."))
                cap = cv2.VideoCapture(index, self.backend)
                self.recognizer.reset()
                if cap.isOpened():
                    self.camera_open = True
                else:
                    cap = None
                    self._set_status("error", f"Camera {index} could not be opened")
                    self._publish(placeholder(f"Camera {index} unavailable", "Pick another camera or try --backend dshow"))
            if cap is None:
                time.sleep(0.1)
                continue
            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap, self.camera_open = None, False
                self._set_status("error", f"Camera {self.camera} stopped delivering frames")
                self._publish(placeholder(f"Camera {self.camera} stopped", "Pick another camera"))
                continue
            if frame[::8, ::8].std() < 2:  # black or flat-grey picture: shutter closed, privacy lock, virtual cam off
                self._set_status("no_signal", f"Camera {self.camera} opened but sends a blank picture")
                self._publish(placeholder(f"No picture from camera {self.camera}", "Blank frames: try another camera"))
                self._result = {"warming": True, "label": None, "confidence": 0.0, "probs": {}}
                time.sleep(0.05)
                continue
            self._set_status("live")
            start = time.perf_counter()
            self._result = self.recognizer.process(frame)
            if self.recognizer.inferred:
                self.latency_ms.append((time.perf_counter() - start) * 1000)
                self.frames += 1
            self._publish(cv2.flip(frame, 1))  # mirror for display only, after drawing
            count += 1
            if time.time() - t0 >= 1:
                self.fps, count, t0 = count / (time.time() - t0), 0, time.time()
        if cap:
            cap.release()


class Settings(BaseModel):
    threshold: float | None = Field(None, ge=0.5, le=1.0)
    hold: int | None = Field(None, ge=1, le=15)
    pause: float | None = Field(None, ge=0.5, le=10.0)
    stride: int | None = Field(None, ge=1, le=3)
    mute: bool | None = None


class CameraChoice(BaseModel):
    index: int = Field(ge=0, le=9)


class Words(BaseModel):
    words: list[str] = Field(max_length=30)


pipeline: Pipeline
config = {"camera": 0, "backend": "any"}


@asynccontextmanager
async def lifespan(app):
    global pipeline
    pipeline = Pipeline(config["camera"], config["backend"])
    yield
    pipeline.stop()


app = FastAPI(title="SignSpeak AI", version="1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.middleware("http")
async def revalidate_ui_files(request, call_next):
    """Browsers must re-check the UI files each load (a cheap 304), so an update is never hidden by a stale cache."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB / "index.html")


@app.get("/video")
async def video():
    async def frames():
        last = -1
        while True:
            jpg, n = pipeline.latest_frame()
            if n != last:
                last = n
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(0.03)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/events")
async def events():
    async def stream():
        while True:
            yield f"data: {json.dumps(pipeline.state())}\n\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/state")
def state():
    return pipeline.state()


@app.get("/api/health")
def health():
    return {"status": "ok" if pipeline.status == "live" else "degraded", "camera": pipeline.status,
            "model": pipeline.model["version"], "uptime_s": round(time.time() - pipeline.started, 1)}


@app.get("/api/metrics")
def metrics():
    return pipeline.metrics()


@app.get("/api/cameras")
async def cameras():
    return await asyncio.to_thread(pipeline.scan)


@app.post("/api/camera")
def set_camera(choice: CameraChoice):
    pipeline.switch_camera(choice.index)
    return {"index": choice.index}


@app.post("/api/settings")
def set_settings(s: Settings):
    rec = pipeline.recognizer
    for name in ("threshold", "hold", "pause", "stride"):
        if getattr(s, name) is not None:
            setattr(rec, name, getattr(s, name))
    if s.mute is not None:
        pipeline.mute = s.mute
    return pipeline.state()["settings"]


@app.put("/api/words")
def set_words(body: Words):
    allowed = set(pipeline.labels) - {"nothing"}
    unknown = [w for w in body.words if w not in allowed]
    if unknown:
        raise HTTPException(422, f"not in the vocabulary: {', '.join(unknown)}")
    pipeline.recognizer.set_words(body.words)
    return {"words": body.words}


@app.post("/api/speak")
def speak_now():
    if not pipeline.recognizer.finish():
        raise HTTPException(409, "no words to speak")
    return {}


@app.post("/api/repeat")
def repeat():
    if not pipeline.repeat():
        raise HTTPException(409, "no sentence yet")
    return {}


@app.post("/api/clear")
def clear():
    pipeline.recognizer.set_words([])
    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="initial camera index; switchable in the UI")
    parser.add_argument("--backend", choices=BACKENDS, default="any", help="OpenCV capture backend; try dshow if a camera won't open")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    config.update(camera=args.camera, backend=args.backend)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
