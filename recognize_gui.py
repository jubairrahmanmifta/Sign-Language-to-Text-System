"""
recognize_gui.py
-----------------
Polished front-end for the Sign Language to Text project.

Wraps the same MediaPipe hand-tracking + joblib RandomForest model used
in recognize.py inside a clean, light, minimal desktop window built with
Tkinter -- so the final demo shows a real application window instead of
a bare OpenCV video window or a terminal.

FEATURES
--------
    - Live camera feed embedded in the app window
    - Large, clear "current sign" readout
    - Confidence bar showing how sure the model is about the current sign
    - Running message bar that fills in as letters are confirmed
    - Buttons for Space / Backspace / Clear / Save to File / Quit
      (no more remembering keyboard shortcuts)
    - Save button writes the message to a timestamped .txt file so you
      can show your teacher a saved transcript afterwards

HOW TO USE
----------
    python recognize_gui.py

REQUIREMENTS
------------
    Same as recognize.py, plus Pillow for displaying video frames in Tkinter:
        pip install pillow
"""

import tkinter as tk
from tkinter import font as tkfont
import cv2
import mediapipe as mp
import joblib
import numpy as np
import time
import datetime
from collections import deque, Counter
from PIL import Image, ImageTk

MODEL_FILE = "sign_model.pkl"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

CONFIRM_FRAMES = 15
PREDICTION_BUFFER_SIZE = 20
COOLDOWN_SECONDS = 0.8

# ---- Light, clean color palette ----
BG = "#F7F8FA"
CARD_BG = "#FFFFFF"
BORDER = "#E3E6EA"
TEXT_MAIN = "#1F2430"
TEXT_MUTED = "#7A8194"
ACCENT = "#3B6CF6"
ACCENT_SOFT = "#EAF0FF"
SUCCESS = "#22A559"
DANGER = "#E5484D"


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


class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language to Text")
        self.root.configure(bg=BG)
        self.root.geometry("980x680")
        self.root.minsize(900, 640)

        # ---- Fonts ----
        self.font_title = tkfont.Font(family="Segoe UI", size=18, weight="bold")
        self.font_label = tkfont.Font(family="Segoe UI", size=11)
        self.font_letter = tkfont.Font(family="Segoe UI", size=48, weight="bold")
        self.font_message = tkfont.Font(family="Segoe UI", size=16)
        self.font_button = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        self.font_small = tkfont.Font(family="Segoe UI", size=9)

        # ---- Model + MediaPipe setup ----
        self.model = joblib.load(MODEL_FILE)
        self.cap = cv2.VideoCapture(0)
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        self.message = ""
        self.prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
        self.last_typed_time = 0
        self.last_typed_letter = None
        self.current_letter = None
        self.confidence = 0.0

        self._build_ui()
        self._update_frame()

    # ------------------------------------------------------------------
    def _build_ui(self):
        # ---- Header ----
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=24, pady=(20, 10))

        tk.Label(
            header, text="Sign Language to Text", font=self.font_title,
            bg=BG, fg=TEXT_MAIN
        ).pack(side="left")

        self.status_dot = tk.Canvas(header, width=12, height=12, bg=BG, highlightthickness=0)
        self.status_dot.pack(side="right", padx=(0, 6), pady=4)
        self.status_dot.create_oval(2, 2, 10, 10, fill=SUCCESS, outline="")
        tk.Label(
            header, text="Camera active", font=self.font_small,
            bg=BG, fg=TEXT_MUTED
        ).pack(side="right", padx=(0, 4))

        # ---- Main content: video (left) + info panel (right) ----
        content = tk.Frame(self.root, bg=BG)
        content.pack(fill="both", expand=True, padx=24, pady=(0, 12))

        # Video card
        video_card = tk.Frame(content, bg=CARD_BG, highlightbackground=BORDER,
                               highlightthickness=1)
        video_card.pack(side="left", fill="both", expand=True, padx=(0, 12))

        self.video_label = tk.Label(video_card, bg="#000000")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Right info panel
        panel = tk.Frame(content, bg=BG, width=260)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)

        # Current sign card
        sign_card = tk.Frame(panel, bg=CARD_BG, highlightbackground=BORDER,
                              highlightthickness=1)
        sign_card.pack(fill="x", pady=(0, 12))

        tk.Label(sign_card, text="CURRENT SIGN", font=self.font_small,
                 bg=CARD_BG, fg=TEXT_MUTED).pack(anchor="w", padx=16, pady=(14, 0))

        self.letter_var = tk.StringVar(value="-")
        tk.Label(sign_card, textvariable=self.letter_var, font=self.font_letter,
                 bg=CARD_BG, fg=ACCENT).pack(padx=16, pady=(2, 6))

        tk.Label(sign_card, text="Confidence", font=self.font_small,
                 bg=CARD_BG, fg=TEXT_MUTED).pack(anchor="w", padx=16)

        bar_track = tk.Frame(sign_card, bg=ACCENT_SOFT, height=10)
        bar_track.pack(fill="x", padx=16, pady=(4, 16))
        self.bar_track = bar_track
        self.bar_fill = tk.Frame(bar_track, bg=ACCENT, height=10, width=0)
        self.bar_fill.place(x=0, y=0, relheight=1)

        # Instructions card
        info_card = tk.Frame(panel, bg=CARD_BG, highlightbackground=BORDER,
                              highlightthickness=1)
        info_card.pack(fill="x", pady=(0, 12))
        tk.Label(info_card, text="HOW IT WORKS", font=self.font_small,
                 bg=CARD_BG, fg=TEXT_MUTED).pack(anchor="w", padx=16, pady=(14, 4))
        tip = ("Hold a sign steadily. Once it's recognized\n"
               "consistently for a moment, the letter is\n"
               "added to your message automatically.")
        tk.Label(info_card, text=tip, font=self.font_small, justify="left",
                 bg=CARD_BG, fg=TEXT_MAIN).pack(anchor="w", padx=16, pady=(0, 14))

        # ---- Message bar (full width, bottom) ----
        message_card = tk.Frame(self.root, bg=CARD_BG, highlightbackground=BORDER,
                                 highlightthickness=1)
        message_card.pack(fill="x", padx=24, pady=(0, 12))

        tk.Label(message_card, text="MESSAGE", font=self.font_small,
                 bg=CARD_BG, fg=TEXT_MUTED).pack(anchor="w", padx=16, pady=(12, 0))

        self.message_var = tk.StringVar(value="")
        tk.Label(message_card, textvariable=self.message_var, font=self.font_message,
                 bg=CARD_BG, fg=TEXT_MAIN, anchor="w", justify="left",
                 wraplength=900).pack(fill="x", padx=16, pady=(2, 14))

        # ---- Controls ----
        controls = tk.Frame(self.root, bg=BG)
        controls.pack(fill="x", padx=24, pady=(0, 20))

        def make_button(parent, text, command, bg=ACCENT, fg="white"):
            btn = tk.Button(
                parent, text=text, command=command, font=self.font_button,
                bg=bg, fg=fg, activebackground=bg, activeforeground=fg,
                relief="flat", padx=16, pady=10, cursor="hand2",
                borderwidth=0
            )
            return btn

        make_button(controls, "Space", self.add_space, bg=CARD_BG, fg=TEXT_MAIN).pack(side="left", padx=(0, 8))
        make_button(controls, "Backspace", self.backspace, bg=CARD_BG, fg=TEXT_MAIN).pack(side="left", padx=8)
        make_button(controls, "Clear", self.clear_message, bg="#FFF1F1", fg=DANGER).pack(side="left", padx=8)
        make_button(controls, "Save to File", self.save_message, bg=ACCENT_SOFT, fg=ACCENT).pack(side="left", padx=8)
        make_button(controls, "Quit", self.quit_app, bg=TEXT_MAIN, fg="white").pack(side="right")

        # style the flat borderless "card" look for outline buttons
        for w in controls.winfo_children():
            w.configure(highlightbackground=BORDER, highlightthickness=1)

        self.save_status_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.save_status_var, font=self.font_small,
                 bg=BG, fg=SUCCESS).pack(anchor="e", padx=28, pady=(0, 8))

    # ------------------------------------------------------------------
    def _update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            self.current_letter = None
            self.confidence = 0.0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    row = np.array(extract_landmark_row(hand_landmarks)).reshape(1, -1)
                    probs = self.model.predict_proba(row)[0]
                    pred_idx = np.argmax(probs)
                    self.current_letter = self.model.classes_[pred_idx]
                    self.confidence = probs[pred_idx]
                    self.prediction_buffer.append(self.current_letter)
            else:
                self.prediction_buffer.clear()

            stable_letter = None
            if len(self.prediction_buffer) == PREDICTION_BUFFER_SIZE:
                counts = Counter(self.prediction_buffer)
                most_common, count = counts.most_common(1)[0]
                if count >= CONFIRM_FRAMES:
                    stable_letter = most_common

            now = time.time()
            if stable_letter is not None:
                if stable_letter != self.last_typed_letter or (now - self.last_typed_time) > COOLDOWN_SECONDS * 3:
                    if (now - self.last_typed_time) > COOLDOWN_SECONDS:
                        self.message += stable_letter
                        self.last_typed_letter = stable_letter
                        self.last_typed_time = now
                        self.prediction_buffer.clear()
                        self.message_var.set(self.message)

            # Update letter + confidence UI
            self.letter_var.set(self.current_letter if self.current_letter else "-")
            frac = self.confidence if self.current_letter else 0.0
            track_width = self.bar_track.winfo_width() or 228
            self.bar_fill.place(width=int(track_width * frac))

            # Convert frame for Tkinter display
            rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_display)

            # Fit to label size while keeping aspect ratio
            label_w = self.video_label.winfo_width() or 600
            label_h = self.video_label.winfo_height() or 450
            img_ratio = img.width / img.height
            box_ratio = label_w / label_h
            if img_ratio > box_ratio:
                new_w = label_w
                new_h = int(label_w / img_ratio)
            else:
                new_h = label_h
                new_w = int(label_h * img_ratio)
            if new_w > 0 and new_h > 0:
                img = img.resize((new_w, new_h))

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(15, self._update_frame)

    # ------------------------------------------------------------------
    def add_space(self):
        self.message += " "
        self.message_var.set(self.message)

    def backspace(self):
        self.message = self.message[:-1]
        self.message_var.set(self.message)

    def clear_message(self):
        self.message = ""
        self.message_var.set(self.message)
        self.save_status_var.set("")

    def save_message(self):
        if not self.message.strip():
            self.save_status_var.set("Nothing to save yet.")
            return
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"message_{timestamp}.txt"
        with open(filename, "w") as f:
            f.write(self.message)
        self.save_status_var.set(f"Saved to {filename}")

    def quit_app(self):
        self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.quit_app)
    root.mainloop()


if __name__ == "__main__":
    main()
