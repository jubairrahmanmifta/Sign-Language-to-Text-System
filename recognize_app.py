"""
SignSpeak (recognize_app.py)
-----------------------------
Professional front-end for the Sign Language to Text final-year project.

DESIGN
------
Palette: deep indigo + cyan on a clean cool-white background -- the kind
of restrained, confident color pairing used in serious technical
products, rather than a purely playful scheme. Cards have real elevation
(a soft drop shadow drawn behind each rounded panel), and structure is
organized like a small dashboard: live camera feed, a focus-ring
confidence gauge, a session stats panel (letters typed, average
confidence, session time, and the model's actual measured test
accuracy), an alphabet strip, and a message composer with word
suggestions.

Showing real stats (especially model accuracy, taken directly from the
classification report produced by train_model.py) is a deliberate
choice: it demonstrates the system's rigor at a glance, not just its
visual polish -- useful for a project being evaluated academically.

ACCURACY BEHAVIOR
------------------
    - A sign only counts toward confirmation once the model's confidence
      passes MIN_CONFIDENCE, and only after CONFIRM_FRAMES consecutive
      matching frames -- reduces mix-ups between similar letters
      (M/N, U/V, etc.).
    - "Also close" hint shows a runner-up letter when the top two
      guesses are close together.

HOW TO USE
----------
    python recognize_app.py

REQUIREMENTS
------------
    pip install pillow
    pip install pyttsx3      (optional, enables the Speak button)
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

CONFIRM_FRAMES = 10
PREDICTION_BUFFER_SIZE = 20
COOLDOWN_SECONDS = 0.8

MIN_CONFIDENCE = 0.35
CLOSE_MARGIN = 0.18

# Reported directly from train_model.py's classification report -- shown
# in the stats panel as a credibility signal, not recalculated at runtime.
MODEL_TEST_ACCURACY = 98.41

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# ---- Professional palette: deep indigo + cyan on cool white ----
INK = "#0F172A"          # slate-900, headings/primary text
MUTED = "#64748B"        # slate-500, secondary text
BG = "#F5F7FA"
CARD_BG = "#FFFFFF"
BORDER = "#E4E9F0"
SHADOW = "#D8DEE8"

PRIMARY = "#4338CA"      # indigo-700 -- primary actions, brand
PRIMARY_SOFT = "#EEF0FD"
CYAN = "#0891B2"         # cyan-600 -- secondary accent, gradient partner
CYAN_SOFT = "#E3F6FA"
AMBER = "#D97706"        # warnings / low confidence
GREEN = "#0F9D68"        # success / confirm flash
RED = "#DC2626"          # destructive actions

COMMON_WORDS = (
    "THE OF AND TO IN IS YOU THAT IT HE WAS FOR ON ARE AS WITH HIS THEY "
    "AT BE THIS HAVE FROM OR ONE HAD BY WORD BUT NOT WHAT ALL WERE WE "
    "WHEN YOUR CAN SAID THERE USE EACH WHICH SHE DO HOW THEIR IF WILL "
    "UP OTHER ABOUT OUT MANY THEN THEM THESE SO SOME HER WOULD MAKE "
    "LIKE HIM INTO TIME HAS LOOK TWO MORE WRITE GO SEE NUMBER NO WAY "
    "COULD PEOPLE MY THAN FIRST WATER BEEN CALL WHO OIL NOW FIND LONG "
    "DOWN DAY DID GET COME MADE MAY PART HELLO THANKS PLEASE YES GOOD "
    "MORNING NIGHT NAME NEED HELP WANT LOVE FAMILY FRIEND WORK SCHOOL "
    "HOME FOOD WATER TODAY TOMORROW"
).split()


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


def lerp_color(c1, c2, t):
    c1 = c1.lstrip("#")
    c2 = c2.lstrip("#")
    r1, g1, b1 = int(c1[0:2], 16), int(c1[2:4], 16), int(c1[4:6], 16)
    r2, g2, b2 = int(c2[0:2], 16), int(c2[2:4], 16), int(c2[4:6], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def rounded_rect(canvas, x1, y1, x2, y2, r=18, **kwargs):
    points = [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


class Card(tk.Canvas):
    """A canvas that draws itself as a white rounded-corner card with a
    soft drop shadow beneath it, for a sense of real elevation."""

    def __init__(self, parent, bg=BG, card_bg=CARD_BG, border=BORDER,
                 radius=20, shadow=True, accent_top=None, **kwargs):
        super().__init__(parent, bg=bg, highlightthickness=0, **kwargs)
        self.radius = radius
        self.card_bg = card_bg
        self.border = border
        self.shadow = shadow
        self.accent_top = accent_top
        self.bind("<Configure>", self._redraw)

    def _redraw(self, event=None):
        self.delete("card_bg")
        w = self.winfo_width()
        h = self.winfo_height()
        if w > 2 and h > 2:
            if self.shadow:
                # Layered, progressively lighter + more offset rects fake a
                # soft blurred drop shadow instead of one hard-edged rect.
                layers = [
                    (8, 0.22), (6, 0.38), (4, 0.60), (2, 0.85),
                ]
                for offset, weight in layers:
                    shade = lerp_color(self.cget("bg"), SHADOW, weight)
                    rounded_rect(self, 2, 2 + offset, w - 2, h - 2 + offset,
                                  r=self.radius, fill=shade, outline="",
                                  tags="card_bg")
            rounded_rect(self, 1, 1, w - 3, h - 3, r=self.radius,
                          fill=self.card_bg, outline=self.border, width=1,
                          tags="card_bg")
            if self.accent_top:
                self.create_line(self.radius, 3, w - self.radius, 3,
                                  fill=self.accent_top, width=3,
                                  tags="card_bg", capstyle="round")
            self.tag_lower("card_bg")


class SignSpeakApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SignSpeak \u2014 Sign Language Recognition System")
        self.root.configure(bg=BG)
        self.root.geometry("1120x780")
        self.root.minsize(1020, 720)

        self.font_wordmark = tkfont.Font(family="Segoe UI Semibold", size=19)
        self.font_tagline = tkfont.Font(family="Segoe UI", size=10)
        self.font_badge = tkfont.Font(family="Segoe UI Semibold", size=8)
        self.font_letter = tkfont.Font(family="Segoe UI", size=46, weight="bold")
        self.font_close = tkfont.Font(family="Segoe UI Semibold", size=10)
        self.font_pct = tkfont.Font(family="Consolas", size=13, weight="bold")
        self.font_label = tkfont.Font(family="Segoe UI Semibold", size=9)
        self.font_message = tkfont.Font(family="Segoe UI", size=17)
        self.font_button = tkfont.Font(family="Segoe UI Semibold", size=10)
        self.font_tile = tkfont.Font(family="Segoe UI Semibold", size=11)
        self.font_suggest = tkfont.Font(family="Segoe UI", size=10)
        self.font_stat_value = tkfont.Font(family="Segoe UI Semibold", size=15)
        self.font_stat_label = tkfont.Font(family="Segoe UI", size=8)
        self.font_footer = tkfont.Font(family="Segoe UI", size=8)

        self.model = joblib.load(MODEL_FILE)
        self.classes = list(self.model.classes_)
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
        self.close_letter = None
        self.flash_until = 0

        # session stats
        self.session_start = time.time()
        self.letters_typed = 0
        self.confidence_history = deque(maxlen=150)

        self.tts_engine = None
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
            except Exception:
                self.tts_engine = None

        self._build_ui()
        self._update_frame()
        self._update_stats()

    # ------------------------------------------------------------------
    def _build_ui(self):
        # Textured backdrop -- soft glow blobs + a faint dot grid, visible
        # through the padding gaps around the cards for a sense of depth.
        self.bg_canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        self.bg_canvas.bind("<Configure>", self._draw_backdrop)

        # Header
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=30, pady=(22, 4))

        logo = tk.Canvas(header, width=40, height=40, bg=BG, highlightthickness=0)
        logo.pack(side="left")
        rounded_rect(logo, 0, 0, 40, 40, r=12, fill=PRIMARY, outline="")
        logo.create_text(20, 20, text="S", fill="white", font=("Segoe UI Semibold", 18))

        title_box = tk.Frame(header, bg=BG)
        title_box.pack(side="left", padx=(12, 0))
        title_row = tk.Frame(title_box, bg=BG)
        title_row.pack(anchor="w")
        tk.Label(title_row, text="SignSpeak", font=self.font_wordmark,
                 bg=BG, fg=INK).pack(side="left")

        badge = tk.Label(title_row, text="FINAL YEAR PROJECT \u00b7 CSE",
                          font=self.font_badge, bg=PRIMARY_SOFT, fg=PRIMARY,
                          padx=8, pady=3)
        badge.pack(side="left", padx=(10, 0))

        tk.Label(title_box, text="Real-Time Sign Language Recognition System",
                 font=self.font_tagline, bg=BG, fg=MUTED).pack(anchor="w")

        status = tk.Frame(header, bg=BG)
        status.pack(side="right")
        dot = tk.Canvas(status, width=10, height=10, bg=BG, highlightthickness=0)
        dot.pack(side="left", padx=(0, 6))
        dot.create_oval(1, 1, 9, 9, fill=GREEN, outline="")
        tk.Label(status, text="CAMERA ACTIVE", font=self.font_label,
                 bg=BG, fg=MUTED).pack(side="left")

        # thin divider
        divider = tk.Frame(self.root, bg=BORDER, height=1)
        divider.pack(fill="x", padx=30, pady=(14, 16))

        # Main split: camera+alphabet (left column) | ring+stats+tips (right column)
        content = tk.Frame(self.root, bg=BG)
        content.pack(fill="both", expand=True, padx=30, pady=(0, 10))

        left_col = tk.Frame(content, bg=BG)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 16))

        cam_card = Card(left_col, radius=22, accent_top=PRIMARY)
        cam_card.pack(fill="both", expand=True)
        cam_inner = tk.Frame(cam_card, bg=CARD_BG)
        cam_card.create_window(14, 16, window=cam_inner, anchor="nw", tags="cam_inner_win")

        def size_cam_inner(event):
            cam_card.itemconfigure("cam_inner_win", width=event.width - 30,
                                    height=event.height - 32)
        cam_card.bind("<Configure>", lambda e: (Card._redraw(cam_card, e), size_cam_inner(e)))

        self.video_label = tk.Label(cam_inner, bg="#0B1220")
        self.video_label.pack(fill="both", expand=True)

        right_col = tk.Frame(content, bg=BG, width=300)
        right_col.pack(side="right", fill="y")
        right_col.pack_propagate(False)

        ring_card = Card(right_col, radius=22, height=248, accent_top=CYAN)
        ring_card.pack(fill="x", pady=(0, 14))
        self.ring_canvas = ring_card
        ring_card.bind("<Configure>", self._draw_ring)

        stats_card = Card(right_col, radius=22, height=168)
        stats_card.pack(fill="x", pady=(0, 14))
        stats_inner = tk.Frame(stats_card, bg=CARD_BG)
        stats_card.create_window(16, 14, window=stats_inner, anchor="nw")
        tk.Label(stats_inner, text="SESSION STATS", font=self.font_label,
                 bg=CARD_BG, fg=PRIMARY).grid(row=0, column=0, columnspan=2, sticky="w")

        def stat_block(parent, row, col, label):
            box = tk.Frame(parent, bg=CARD_BG)
            box.grid(row=row, column=col, sticky="w", padx=(0, 24), pady=(10, 0))
            val = tk.Label(box, text="--", font=self.font_stat_value, bg=CARD_BG, fg=INK)
            val.pack(anchor="w")
            tk.Label(box, text=label, font=self.font_stat_label, bg=CARD_BG, fg=MUTED).pack(anchor="w")
            return val

        self.stat_letters = stat_block(stats_inner, 1, 0, "LETTERS TYPED")
        self.stat_time = stat_block(stats_inner, 1, 1, "SESSION TIME")
        self.stat_conf = stat_block(stats_inner, 2, 0, "AVG CONFIDENCE")
        self.stat_accuracy = stat_block(stats_inner, 2, 1, "MODEL ACCURACY")
        self.stat_accuracy.configure(text=f"{MODEL_TEST_ACCURACY:.1f}%", fg=GREEN)

        tips_card = Card(right_col, radius=22)
        tips_card.pack(fill="both", expand=True)
        tips_inner = tk.Frame(tips_card, bg=CARD_BG)
        tips_card.create_window(16, 14, window=tips_inner, anchor="nw")
        tk.Label(tips_inner, text="TIP", font=self.font_label,
                 bg=CARD_BG, fg=CYAN).pack(anchor="w")
        tip_text = ("Hold your sign steady and centered.\n"
                     "If two letters look alike (like M/N\n"
                     "or U/V), watch the \"also close\" hint\n"
                     "and adjust your hand until only one\n"
                     "letter stays confident.")
        tk.Label(tips_inner, text=tip_text, font=self.font_suggest, justify="left",
                 bg=CARD_BG, fg=INK).pack(anchor="w", pady=(6, 0))

        # Alphabet strip
        strip_card = Card(self.root, radius=18, height=56)
        strip_card.pack(fill="x", padx=30, pady=(0, 14))
        self.strip_inner = tk.Frame(strip_card, bg=CARD_BG)
        strip_card.create_window(10, 8, window=self.strip_inner, anchor="nw")
        self.tile_labels = {}
        for letter in self.classes:
            lbl = tk.Label(self.strip_inner, text=letter, font=self.font_tile,
                            bg=CARD_BG, fg=MUTED, width=2, height=1)
            lbl.pack(side="left", padx=3, pady=6)
            self.tile_labels[letter] = lbl

        # Message card
        msg_card = Card(self.root, radius=22, height=140, accent_top=PRIMARY)
        msg_card.pack(fill="x", padx=30, pady=(0, 14))
        msg_inner = tk.Frame(msg_card, bg=CARD_BG)
        msg_card.create_window(18, 16, window=msg_inner, anchor="nw")

        top_row = tk.Frame(msg_inner, bg=CARD_BG)
        top_row.pack(fill="x")
        tk.Label(top_row, text="MESSAGE", font=self.font_label,
                 bg=CARD_BG, fg=PRIMARY).pack(side="left")

        self.suggest_frame = tk.Frame(msg_inner, bg=CARD_BG)
        self.suggest_frame.pack(fill="x", pady=(6, 2))

        self.message_var = tk.StringVar(value="Start signing to type a message...")
        self.message_label = tk.Label(msg_inner, textvariable=self.message_var,
                                       font=self.font_message, bg=CARD_BG, fg=MUTED,
                                       anchor="w", justify="left", wraplength=1020)
        self.message_label.pack(fill="x", pady=(4, 0))

        # Controls
        controls = tk.Frame(self.root, bg=BG)
        controls.pack(fill="x", padx=30, pady=(0, 6))

        def make_button(parent, text, command, bg, fg):
            f = Card(parent, bg=BG, card_bg=bg, border=bg, radius=14, height=40, shadow=False)
            btn = tk.Label(f, text=text, font=self.font_button, bg=bg, fg=fg,
                            padx=16, cursor="hand2")
            btn_window = f.create_window(10, 20, window=btn, anchor="w")

            def size_btn(event):
                f.coords(btn_window, 10, event.height / 2)

            f.bind("<Configure>", lambda e: (Card._redraw(f, e), size_btn(e)))
            f.bind("<Button-1>", lambda e: command())
            btn.bind("<Button-1>", lambda e: command())
            return f

        make_button(controls, "Space", self.add_space, PRIMARY_SOFT, PRIMARY).pack(side="left", padx=(0, 8), ipadx=2)
        make_button(controls, "Backspace", self.backspace, PRIMARY_SOFT, PRIMARY).pack(side="left", padx=8)
        make_button(controls, "Clear", self.clear_message, "#FDECEC", RED).pack(side="left", padx=8)
        make_button(controls, "Save to File", self.save_message, PRIMARY, "white").pack(side="left", padx=8)
        speak_label = "Speak Message" if TTS_AVAILABLE else "Speak (install pyttsx3)"
        make_button(controls, speak_label, self.speak_message, CYAN, "white").pack(side="left", padx=8)
        make_button(controls, "Quit", self.quit_app, INK, "white").pack(side="right")

        self.save_status_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.save_status_var, font=self.font_label,
                 bg=BG, fg=PRIMARY).pack(anchor="e", padx=32, pady=(6, 4))

        # Footer
        footer = tk.Frame(self.root, bg=BG)
        footer.pack(fill="x", padx=30, pady=(0, 16))
        tk.Label(footer, text="Built with Python \u00b7 OpenCV \u00b7 MediaPipe \u00b7 scikit-learn",
                 font=self.font_footer, bg=BG, fg=MUTED).pack(side="left")

    # ------------------------------------------------------------------
    def _draw_backdrop(self, event=None):
        c = self.bg_canvas
        c.delete("backdrop")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10:
            return

        # Soft glow blobs: concentric circles fading from a tinted color
        # back into the background color, top-left indigo / bottom-right cyan.
        def glow(cx, cy, radius, color, rings=10):
            for i in range(rings, 0, -1):
                t = i / rings
                rad = radius * t
                shade = lerp_color(BG, color, (1 - t) * 0.16)
                c.create_oval(cx - rad, cy - rad, cx + rad, cy + rad,
                              fill=shade, outline="", tags="backdrop")

        glow(w * 0.08, h * 0.05, min(w, h) * 0.55, PRIMARY)
        glow(w * 0.97, h * 0.9, min(w, h) * 0.6, CYAN)
        c.tag_lower("backdrop")

        # Faint dot grid texture across the whole background.
        spacing = 26
        dot_color = lerp_color(BG, BORDER, 0.7)
        for gy in range(0, h, spacing):
            for gx in range(0, w, spacing):
                c.create_oval(gx, gy, gx + 1.4, gy + 1.4, fill=dot_color,
                              outline="", tags="backdrop")
        c.tag_lower("backdrop")

    def _draw_ring(self, event=None):
        c = self.ring_canvas
        c.delete("ring")
        w = c.winfo_width()
        if w < 10:
            return
        cx, cy, r = w / 2, 108, 66

        c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=BORDER,
                      width=11, tags="ring")

        frac = self.confidence if self.current_letter else 0.0
        if frac > 0:
            total_degrees = 360 * frac
            segments = max(int(total_degrees / 4), 1)
            for i in range(segments):
                seg_t = i / max(segments - 1, 1)
                color = lerp_color(PRIMARY, CYAN, seg_t)
                seg_start = 90 - (total_degrees * i / segments)
                seg_extent = -(total_degrees / segments) - 0.5
                c.create_arc(cx - r, cy - r, cx + r, cy + r, start=seg_start,
                              extent=seg_extent, style="arc", outline=color,
                              width=11, tags="ring")

        letter_text = self.current_letter if self.current_letter else "-"
        c.create_text(cx, cy - 6, text=letter_text, font=self.font_letter,
                      fill=INK, tags="ring")

        pct_text = f"{int(self.confidence * 100)}%" if self.current_letter else "--"
        pct_color = GREEN if self.confidence >= MIN_CONFIDENCE else AMBER
        c.create_text(cx, cy + 34, text=pct_text, font=self.font_pct,
                      fill=pct_color, tags="ring")

        close_text = f"Also close: {self.close_letter}" if self.close_letter else "Hold a sign steady"
        c.create_text(cx, cy + r + 22, text=close_text, font=self.font_close,
                      fill=MUTED, tags="ring")

    def _refresh_alphabet_strip(self):
        for letter, lbl in self.tile_labels.items():
            if letter == self.current_letter:
                bg = GREEN if time.time() < self.flash_until else PRIMARY
                lbl.configure(bg=bg, fg="white")
            else:
                lbl.configure(bg=CARD_BG, fg=MUTED)

    def _refresh_suggestions(self):
        for w in self.suggest_frame.winfo_children():
            w.destroy()
        words = self.message.split(" ")
        current_word = words[-1].upper() if words else ""
        if not current_word:
            return
        matches = [w for w in COMMON_WORDS if w.startswith(current_word)][:5]
        for m in matches:
            chip = tk.Label(self.suggest_frame, text=m.title(), font=self.font_suggest,
                             bg=CYAN_SOFT, fg=CYAN, padx=10, pady=3, cursor="hand2")
            chip.pack(side="left", padx=(0, 6))
            chip.bind("<Button-1>", lambda e, word=m: self._apply_suggestion(word))

    def _apply_suggestion(self, word):
        words = self.message.split(" ")
        words[-1] = word.title()
        self.message = " ".join(words) + " "
        self._refresh_message_display()

    def _refresh_message_display(self):
        if self.message:
            self.message_var.set(self.message)
            self.message_label.configure(fg=INK)
        else:
            self.message_var.set("Start signing to type a message...")
            self.message_label.configure(fg=MUTED)
        self._refresh_suggestions()

    def _update_stats(self):
        elapsed = int(time.time() - self.session_start)
        minutes, seconds = divmod(elapsed, 60)
        self.stat_time.configure(text=f"{minutes:02d}:{seconds:02d}")
        self.stat_letters.configure(text=str(self.letters_typed))
        if self.confidence_history:
            avg = sum(self.confidence_history) / len(self.confidence_history)
            self.stat_conf.configure(text=f"{avg * 100:.0f}%")
        self.root.after(500, self._update_stats)

    # ------------------------------------------------------------------
    def _apply_vignette(self, arr):
        """Darken the corners of the camera feed slightly for a more
        cinematic, dimensional look. Cached per frame size since the
        mask only depends on (h, w), not pixel content."""
        h, w = arr.shape[:2]
        cache_key = (h, w)
        if getattr(self, "_vignette_key", None) != cache_key:
            yy, xx = np.mgrid[0:h, 0:w]
            cx, cy = w / 2, h / 2
            dist = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
            mask = 1 - 0.35 * np.clip(dist - 0.55, 0, 1)
            self._vignette_mask = np.clip(mask, 0, 1)[..., None]
            self._vignette_key = cache_key
        return (arr * self._vignette_mask).astype(np.uint8)

    def _update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            self.current_letter = None
            self.confidence = 0.0
            self.close_letter = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    row = np.array(extract_landmark_row(hand_landmarks)).reshape(1, -1)
                    probs = self.model.predict_proba(row)[0]
                    order = np.argsort(probs)[::-1]
                    top_idx, second_idx = order[0], order[1]

                    self.current_letter = self.model.classes_[top_idx]
                    self.confidence = probs[top_idx]
                    self.confidence_history.append(self.confidence)

                    if probs[top_idx] - probs[second_idx] < CLOSE_MARGIN:
                        self.close_letter = self.model.classes_[second_idx]

                    if self.confidence >= MIN_CONFIDENCE:
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
                        self.flash_until = now + 0.4
                        self.letters_typed += 1
                        self.prediction_buffer.clear()
                        self._refresh_message_display()

            self._draw_ring()
            self._refresh_alphabet_strip()

            rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_display = self._apply_vignette(rgb_display)
            img = Image.fromarray(rgb_display)
            label_w = self.video_label.winfo_width() or 600
            label_h = self.video_label.winfo_height() or 450
            img_ratio = img.width / img.height
            box_ratio = label_w / max(label_h, 1)
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
        self._refresh_message_display()

    def backspace(self):
        self.message = self.message[:-1]
        self._refresh_message_display()

    def clear_message(self):
        self.message = ""
        self._refresh_message_display()
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

    def speak_message(self):
        if not self.message.strip():
            self.save_status_var.set("Nothing to speak yet.")
            return
        if not TTS_AVAILABLE or self.tts_engine is None:
            self.save_status_var.set("Install voice support: pip install pyttsx3")
            return
        self.tts_engine.say(self.message)
        self.tts_engine.runAndWait()

    def quit_app(self):
        self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SignSpeakApp(root)
    root.protocol("WM_DELETE_WINDOW", app.quit_app)
    root.mainloop()


if __name__ == "__main__":
    main()
