"""SignSpeak AI web server: camera + recognition in a worker thread, a browser UI on top.

Usage (from the repo root): python server.py [--camera 0] [--backend any|dshow|msmf] [--port 8000]
Then open http://localhost:8000

  GET  /video           MJPEG stream of the annotated camera feed
  GET  /api/events      server-sent events: live state (sign, probabilities, words, sentence, camera status)
  GET  /api/cameras     probe camera indices 0-4
  POST /api/camera      {"index": n}  switch camera at runtime
  POST /api/settings    {"threshold", "hold", "pause", "mute"}  all optional
  POST /api/clear       drop the accepted words
"""
import argparse
import asyncio
import json
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import Recognizer, load_model
from sentence import MODEL, make_speaker, to_sentence

BACKENDS = {"any": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
WEB = Path(__file__).with_name("web")


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
        self.recognizer = Recognizer(model, self.labels, self._on_sentence)
        self.mute, self.sentence = False, ""
        self.status, self.message, self.fps, self.camera_open = "starting", "Opening camera", 0.0, False
        self._result = {"warming": True, "label": None, "confidence": 0.0, "probs": {}}
        self._frame, self._frame_no = jpeg(placeholder("Starting camera...")), 0
        self._sentences = queue.Queue()
        self._reopen, self._stop = threading.Event(), threading.Event()
        self._reopen.set()
        threading.Thread(target=self._speech_worker, daemon=True).start()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    # --- public API -------------------------------------------------------------------------------------------
    def switch_camera(self, index):
        self.camera = index
        self._reopen.set()

    def clear(self):
        self.recognizer.words.clear()

    def state(self):
        r = self._result
        return {
            "camera": {"index": self.camera, "status": self.status, "message": self.message, "fps": round(self.fps, 1)},
            "warming": r["warming"], "label": r["label"], "confidence": r["confidence"], "probs": r["probs"],
            "words": list(self.recognizer.words), "sentence": self.sentence,
            "settings": {"threshold": self.recognizer.threshold, "hold": self.recognizer.hold,
                         "pause": self.recognizer.pause, "mute": self.mute},
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
        self.status, self.message = status, message

    def _on_sentence(self, words):
        self._sentences.put(words)

    def _speech_worker(self):
        speak = make_speaker()  # created on this thread (COM rule)
        while True:
            words = self._sentences.get()
            self.sentence = "..."
            self.sentence = to_sentence(words, MODEL)
            print("Sentence:", self.sentence)
            if not self.mute:
                speak(self.sentence)

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
            self._result = self.recognizer.process(frame)
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
    mute: bool | None = None


class CameraChoice(BaseModel):
    index: int = Field(ge=0, le=9)


pipeline: Pipeline
config = {"camera": 0, "backend": "any"}


@asynccontextmanager
async def lifespan(app):
    global pipeline
    pipeline = Pipeline(config["camera"], config["backend"])
    yield
    pipeline.stop()


app = FastAPI(title="SignSpeak AI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
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
    for name in ("threshold", "hold", "pause"):
        if getattr(s, name) is not None:
            setattr(rec, name, getattr(s, name))
    if s.mute is not None:
        pipeline.mute = s.mute
    return pipeline.state()["settings"]


@app.post("/api/clear")
def clear():
    pipeline.clear()
    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="initial camera index; switchable in the UI")
    parser.add_argument("--backend", choices=BACKENDS, default="any", help="OpenCV capture backend; try dshow if a camera won't open")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    config.update(camera=args.camera, backend=args.backend)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
