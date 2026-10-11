import cv2
import mediapipe as mp

from keypoints import FEATURES, extract_keypoints, normalize_keypoints


mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(2)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permissions.")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    print("Holistic tracking started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam.")
            break

        # Holistic expects the raw (unmirrored) image; feeding it a mirrored one swaps left/right hands.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        frame = cv2.flip(frame, 1)  # mirror for display only, after drawing

        keypoints = normalize_keypoints(extract_keypoints(results, frame))
        assert keypoints.shape == (FEATURES,)
        # Right wrist relative to shoulders: should barely change when you step closer/further or sideways.
        wrist = keypoints[132 + 63:132 + 65]

        status = f"L hand: {'yes' if results.left_hand_landmarks else 'no'}  R hand: {'yes' if results.right_hand_landmarks else 'no'}"
        cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"R wrist (shoulder units): {wrist[0]:+.2f}, {wrist[1]:+.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Press q to quit", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("SignSpeak AI - Holistic Keypoints", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
