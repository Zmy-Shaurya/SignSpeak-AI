import cv2
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def extract_landmarks(hand_landmarks):
    coords = []
    for landmark in hand_landmarks.landmark:
        coords.append([landmark.x, landmark.y, landmark.z])
    return np.array(coords, dtype=np.float32)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permissions.")


with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as hands:
    print("Place your hand in front of the camera and keep it visible.")
    print("Press 's' to save one landmark sample, or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        landmark_array = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmark_array = extract_landmarks(hand_landmarks)

            cv2.putText(
                frame,
                "Hand detected",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        else:
            cv2.putText(
                frame,
                "No hand detected",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv2.putText(
            frame,
            "Press s to save or q to quit",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("SignSpeak AI - Landmark Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s') and landmark_array is not None:
            np.save("learning/hand_sample.npy", landmark_array)
            print(f"Saved {landmark_array.shape} sample to learning/hand_sample.npy, wrist = {landmark_array[0]}")

    cap.release()
    cv2.destroyAllWindows()
