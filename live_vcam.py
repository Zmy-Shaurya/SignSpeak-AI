"""live_predict.py, but the annotated video also goes to a virtual webcam (pick "OBS Virtual Camera" in any app).

Usage: python live_vcam.py [same flags as live_predict.py]
Needs OBS Studio installed (for its virtual camera driver) and `pip install pyvirtualcam`.
Works by wrapping cv2.imshow, so every frame live_predict.py shows is also sent to the virtual camera.
"""
import os
import runpy

import cv2
import pyvirtualcam

cam = None  # opened on the first frame, once the frame size is known
show = cv2.imshow


def imshow(name, frame):
    global cam
    show(name, frame)
    if cam is None:
        cam = pyvirtualcam.Camera(frame.shape[1], frame.shape[0], 30, fmt=pyvirtualcam.PixelFormat.BGR)
        print("Virtual camera:", cam.device)
    cam.send(frame)


cv2.imshow = imshow
try:
    runpy.run_path(os.path.join(os.path.dirname(__file__), "live_predict.py"), run_name="__main__")
finally:
    if cam:
        cam.close()
