"""
recognize.py
----------------
Step 3 of the pipeline (the actual final system).

Opens the webcam, detects your hand using MediaPipe, feeds the landmark
coordinates into the trained Random Forest model (sign_model.pkl), and
displays the predicted letter live on screen. Confirmed letters are
appended into a running text message shown at the bottom of the window.

CONTROLS
--------
    Hold a sign steadily -> after it is predicted consistently for a
    short time, the letter is "confirmed" and added to the message.
    This avoids accidentally typing every flickering frame.

    SPACE bar -> add a space to the message
    BACKSPACE -> delete the last character
    c          -> clear the whole message
    q          -> quit

HOW TO USE
----------
    python recognize.py
"""

import cv2
import mediapipe as mp
import joblib
import numpy as np
import time
from collections import deque, Counter

MODEL_FILE = "sign_model.pkl"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# How many consecutive stable predictions are needed before a letter
# is "typed" into the message. Higher = more accurate but slower to type.
CONFIRM_FRAMES = 15
PREDICTION_BUFFER_SIZE = 20
COOLDOWN_SECONDS = 0.8  # prevents the same letter being typed repeatedly


def extract_landmark_row(hand_landmarks):
    base_x = hand_landmarks.landmark[0].x
    base_y = hand_landmarks.landmark[0].y
    base_z = hand_landmarks.landmark[0].z

    row = []
    for lm in hand_landmarks.landmark:
        row.append(lm.x - base_x)
        row.append(lm.y - base_y)
        row.append(lm.z - base_z)
    return row


def main():
    model = joblib.load(MODEL_FILE)

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    message = ""
    prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
    last_typed_time = 0
    last_typed_letter = None

    print("Controls: SPACE = space | BACKSPACE = delete | c = clear | q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        current_letter = None
        confidence = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                row = np.array(extract_landmark_row(hand_landmarks)).reshape(1, -1)
                probs = model.predict_proba(row)[0]
                pred_idx = np.argmax(probs)
                current_letter = model.classes_[pred_idx]
                confidence = probs[pred_idx]

                prediction_buffer.append(current_letter)
        else:
            prediction_buffer.clear()

        # Check if the last N predictions are all the same letter (stable sign)
        stable_letter = None
        if len(prediction_buffer) == PREDICTION_BUFFER_SIZE:
            counts = Counter(prediction_buffer)
            most_common, count = counts.most_common(1)[0]
            if count >= CONFIRM_FRAMES:
                stable_letter = most_common

        now = time.time()
        if stable_letter is not None:
            if stable_letter != last_typed_letter or (now - last_typed_time) > COOLDOWN_SECONDS * 3:
                if (now - last_typed_time) > COOLDOWN_SECONDS:
                    message += stable_letter
                    last_typed_letter = stable_letter
                    last_typed_time = now
                    prediction_buffer.clear()

        # --- UI overlay ---
        h, w, _ = frame.shape
        overlay_text = f"Sign: {current_letter if current_letter else '-'}"
        if current_letter:
            overlay_text += f" ({confidence*100:.0f}%)"
        cv2.putText(frame, overlay_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.rectangle(frame, (0, h - 60), (w, h), (30, 30, 30), -1)
        cv2.putText(frame, message[-40:], (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Sign Language to Text", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == 32:  # space bar
            message += " "
        elif key == 8:  # backspace
            message = message[:-1]
        elif key == ord('c'):
            message = ""

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal message: {message}")


if __name__ == "__main__":
    main()
