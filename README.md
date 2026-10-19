# Sign Language to Text System

A real-time system that converts hand sign gestures captured from a laptop
webcam into text, using hand-landmark tracking and a trained classifier.

## How it works (architecture)

1. **Hand detection & landmark extraction (MediaPipe Hands)**
   Each webcam frame is passed to MediaPipe's hand-tracking model, which
   detects the hand and outputs 21 (x, y, z) landmark points describing
   finger and joint positions. These coordinates are normalized relative
   to the wrist point so the system works regardless of hand position or
   distance from the camera.

2. **Feature vector**
   The 21 landmarks x 3 coordinates = 63 numerical features per frame.
   This is a compact, structured representation of hand shape - far
   smaller and less noisy than raw pixel data, so it can be classified
   accurately with a lightweight model and no GPU.

3. **Classification (Random Forest)**
   A Random Forest classifier is trained on landmark feature vectors
   labelled with the sign they represent. At runtime, each frame's
   feature vector is classified into the most likely sign.

4. **Temporal stability filter**
   To avoid flickering/incorrect letters from a single noisy frame, the
   system only "confirms" a letter into the output text once the same
   prediction has been stable across a sliding window of recent frames,
   with a short cooldown between confirmations.

5. **Text assembly**
   Confirmed letters are appended to a running message string, shown
   live on the video feed, with manual controls for space/backspace/clear.

## Pipeline / how to run

```
pip install -r requirements.txt

# Step 1: Record your own training samples for each sign
python collect_data.py

# Step 2: Train the classifier on your recorded data
python train_model.py

# Step 3: Run live recognition
python recognize.py
```

## Notes for the project report

- This is a **landmark-based** approach (geometric features), as opposed to
  raw-image CNN approaches. It is a recognized, widely-used method in
  sign-language recognition research because it is robust to lighting,
  background, and skin tone, and trains fast on a CPU.
- Suggested signs to start with: the ASL alphabet (A-Z), or a smaller set
  of common words/phrases if time is limited - fewer classes = higher
  accuracy and a faster demo.
- Accuracy depends heavily on the variety of your training data: vary
  hand angle, distance from camera, and lighting while recording.
- Known limitations to mention in your report: single-hand only, static
  gestures only (no motion-based signs like "J" or "Z" in ASL, which
  require sequence modelling such as an LSTM over time), and sensitivity
  to background clutter or poor lighting.
- Possible future work to mention: extending to dynamic/motion signs
  using a sequence model (e.g., LSTM) over landmark sequences instead of
  single-frame classification, and supporting two-hand signs.

## File overview

| File              | Purpose                                              |
|-------------------|-------------------------------------------------------|
| collect_data.py   | Records labelled hand-landmark samples from webcam   |
| train_model.py    | Trains Random Forest classifier on collected data    |
| recognize.py      | Real-time webcam sign-to-text recognition            |
| dataset.csv       | Generated training data (after running collect_data) |
| sign_model.pkl    | Saved trained model (after running train_model)      |
