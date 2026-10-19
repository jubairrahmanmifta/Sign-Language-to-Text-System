"""
collect_data.py
----------------
Step 1 of the pipeline.

Captures hand landmarks from your webcam using MediaPipe Hands and saves
them to a CSV file, labelled with the sign/letter you are currently
performing. You build your own training dataset this way.

HOW TO USE
----------
1. Run the script:
       python collect_data.py
2. It will ask you which letter/sign you are recording (e.g. "A").
3. Hold your hand sign steadily in front of the webcam.
4. Press 's' to start/stop saving frames for that sign (a counter shows
   how many samples have been captured).
5. Press 'n' to move on to a new sign label.
6. Press 'q' to quit and save everything to dataset.csv.

It is recommended to collect at least 80-150 samples per sign, moving
your hand slightly (angle, distance, lighting) between captures so the
model generalizes instead of memorizing one exact pose.
"""

import cv2
import mediapipe as mp
import csv
import os

DATA_FILE = "dataset.csv"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# 21 hand landmarks, each with (x, y, z) -> 63 features per sample
NUM_LANDMARKS = 21


def extract_landmark_row(hand_landmarks):
    """Flatten MediaPipe landmark object into a list of 63 floats,
    normalized relative to the wrist so the model is less sensitive
    to where the hand is in the frame."""
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
    file_exists = os.path.isfile(DATA_FILE)
    csv_file = open(DATA_FILE, mode="a", newline="")
    writer = csv.writer(csv_file)

    if not file_exists:
        header = []
        for i in range(NUM_LANDMARKS):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label")
        writer.writerow(header)

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    current_label = input("Enter the sign/letter you will record first (e.g. A): ").strip().upper()
    recording = False
    sample_count = 0

    print("\nControls: 's' = start/stop recording | 'n' = new label | 'q' = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if recording:
                    row = extract_landmark_row(hand_landmarks)
                    row.append(current_label)
                    writer.writerow(row)
                    sample_count += 1

        status_color = (0, 0, 255) if recording else (0, 255, 0)
        cv2.putText(frame, f"Label: {current_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Samples: {sample_count}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "RECORDING" if recording else "PAUSED", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        cv2.imshow("Data Collection - Sign Language Project", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            recording = not recording
            sample_count = 0
        elif key == ord('n'):
            recording = False
            current_label = input("\nEnter next sign/letter label: ").strip().upper()
            sample_count = 0
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"\nDone. Data saved to {DATA_FILE}")


if __name__ == "__main__":
    main()
