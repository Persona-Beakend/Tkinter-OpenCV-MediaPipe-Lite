"""
AI Body Tracker — покращена версія
Залежності:
    pip install opencv-python mediapipe pillow

Під час першого запуску модель MediaPipe завантажується автоматично.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import urllib.request
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import cv2
import mediapipe as mp
from PIL import Image, ImageTk


# ------------------------------------------------------------
# НАЛАШТУВАННЯ
# ------------------------------------------------------------

APP_TITLE = "AI Body Tracker"
WINDOW_SIZE = "1100x720"
CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30
UI_INTERVAL_MS = 15

# Аналізувати кожен кадр (1) або кожен другий (2).
AI_EVERY_N_FRAMES = 1

# Згладжування координат: менше значення = плавніше, але повільніше.
SMOOTHING_ALPHA = 0.55
MIN_VISIBILITY = 0.35

BACKGROUND = "#101217"
PANEL = "#191c23"
GREEN = (0, 230, 118)       # BGR
WHITE = "#ffffff"
MUTED = "#858b98"
RED = "#ff5252"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/"
    "pose_landmarker_lite.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "pose_landmarker_lite.task"

CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]

BODY_INDICES = [
    11, 12, 13, 14, 15, 16,
    23, 24, 25, 26, 27, 28,
    29, 30, 31, 32,
]


# ------------------------------------------------------------
# МОДЕЛЬ
# ------------------------------------------------------------

def ensure_model() -> Path:
    """Завантажує модель у тимчасовий файл і лише потім перейменовує."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100_000:
        return MODEL_PATH

    temporary_path = MODEL_PATH.with_suffix(".download")

    print("Завантаження моделі MediaPipe Pose Lite...")
    try:
        request = urllib.request.Request(
            MODEL_URL,
            headers={"User-Agent": "AI-Body-Tracker/1.0"},
        )
        with urllib.request.urlopen(request, timeout=40) as response:
            with temporary_path.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

        if temporary_path.stat().st_size < 100_000:
            raise RuntimeError("Завантажений файл моделі занадто малий.")

        os.replace(temporary_path, MODEL_PATH)
        print("Модель успішно завантажена.")
        return MODEL_PATH

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


# ------------------------------------------------------------
# ОБРОБКА ПОЗИ
# ------------------------------------------------------------

def smooth_landmarks(previous, current, alpha=SMOOTHING_ALPHA):
    """Експоненційне згладжування координат для зменшення тремтіння."""
    if previous is None:
        return [
            (p.x, p.y, p.z, getattr(p, "visibility", 1.0))
            for p in current
        ]

    result = []
    for old, point in zip(previous, current):
        visibility = getattr(point, "visibility", 1.0)
        if visibility < MIN_VISIBILITY:
            result.append(old)
            continue

        x = old[0] + alpha * (point.x - old[0])
        y = old[1] + alpha * (point.y - old[1])
        z = old[2] + alpha * (point.z - old[2])
        result.append((x, y, z, visibility))

    return result


def draw_pose(frame, landmarks):
    """Малює скелет, точки та рамку голови."""
    height, width = frame.shape[:2]

    # Лінії скелета
    for start, end in CONNECTIONS:
        p1 = landmarks[start]
        p2 = landmarks[end]

        if p1[3] < MIN_VISIBILITY or p2[3] < MIN_VISIBILITY:
            continue

        x1, y1 = int(p1[0] * width), int(p1[1] * height)
        x2, y2 = int(p2[0] * width), int(p2[1] * height)

        cv2.line(
            frame, (x1, y1), (x2, y2),
            GREEN, 3, cv2.LINE_AA
        )

    # Точки тіла
    for index in BODY_INDICES:
        point = landmarks[index]
        if point[3] < MIN_VISIBILITY:
            continue

        x = int(point[0] * width)
        y = int(point[1] * height)

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 3, GREEN, -1, cv2.LINE_AA)

    # Рамка голови за носом і вухами.
    nose, left_ear, right_ear = landmarks[0], landmarks[7], landmarks[8]
    if min(nose[3], left_ear[3], right_ear[3]) >= MIN_VISIBILITY:
        nose_x, nose_y = int(nose[0] * width), int(nose[1] * height)
        left_x = int(left_ear[0] * width)
        right_x = int(right_ear[0] * width)

        head_width = max(abs(right_x - left_x), 30)
        center_x = (left_x + right_x) // 2

        x1 = max(0, center_x - int(head_width * 0.75))
        x2 = min(width - 1, center_x + int(head_width * 0.75))
        y1 = max(0, nose_y - int(head_width * 0.95))
        y2 = min(height - 1, nose_y + int(head_width * 0.60))

        if x2 > x1 and y2 > y1:
            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                GREEN, 2, cv2.LINE_AA
            )


# ------------------------------------------------------------
# ФОНОВИЙ ПОТІК КАМЕРИ ТА AI
# ------------------------------------------------------------

class CameraWorker(threading.Thread):
    """Камера й MediaPipe працюють окремо від інтерфейсу Tkinter."""

    def __init__(self, output_queue, stop_event, model_path):
        super().__init__(daemon=True)
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.model_path = model_path

    def publish(self, packet):
        """Зберігає лише найновіший кадр, щоб черга не накопичувалась."""
        try:
            self.output_queue.put_nowait(packet)
        except queue.Full:
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.output_queue.put_nowait(packet)
            except queue.Full:
                pass

    def run(self):
        camera = None
        landmarker = None

        try:
            # CAP_DSHOW доступний у Windows; на інших ОС — звичайний backend.
            if os.name == "nt":
                camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
            else:
                camera = cv2.VideoCapture(CAMERA_INDEX)

            if not camera.isOpened():
                if camera:
                    camera.release()
                camera = cv2.VideoCapture(CAMERA_INDEX)

            if not camera.isOpened():
                raise RuntimeError("Не вдалося відкрити камеру.")

            camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            camera.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(self.model_path)
                ),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.45,
                min_pose_presence_confidence=0.45,
                min_tracking_confidence=0.45,
            )
            landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
                options
            )

            self.publish({"status": "live"})
            frame_number = 0
            smoothed = None
            latest_pose = None
            started_at = time.perf_counter()

            while not self.stop_event.is_set():
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("Камера не повернула відеокадр.")

                frame = cv2.flip(frame, 1)
                frame_number += 1

                if frame_number % AI_EVERY_N_FRAMES == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb,
                    )
                    timestamp_ms = int(
                        (time.perf_counter() - started_at) * 1000
                    )
                    result = landmarker.detect_for_video(
                        mp_image, timestamp_ms
                    )

                    if result.pose_landmarks:
                        smoothed = smooth_landmarks(
                            smoothed, result.pose_landmarks[0]
                        )
                        latest_pose = smoothed
                    else:
                        smoothed = None
                        latest_pose = None

                if latest_pose is not None:
                    draw_pose(frame, latest_pose)

                self.publish({
                    "status": "frame",
                    "frame": frame,
                    "people": 1 if latest_pose is not None else 0,
                    "points": (
                        sum(
                            1 for i in BODY_INDICES
                            if latest_pose[i][3] >= MIN_VISIBILITY
                        )
                        if latest_pose is not None else 0
                    ),
                })

        except Exception as error:
            self.publish({"status": "error", "message": str(error)})

        finally:
            if landmarker is not None:
                landmarker.close()
            if camera is not None:
                camera.release()
            self.publish({"status": "stopped"})


# ------------------------------------------------------------
# ГРАФІЧНИЙ ІНТЕРФЕЙС
# ------------------------------------------------------------

class BodyTracker:
    def __init__(self, root, model_path):
        self.root = root
        self.model_path = model_path

        self.root.title(APP_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.minsize(850, 560)
        self.root.configure(bg=BACKGROUND)

        self.worker = None
        self.stop_event = None
        self.frames = queue.Queue(maxsize=1)
        self.running = False
        self.photo = None
        self.last_frame_time = time.perf_counter()
        self.fps = 0.0

        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(UI_INTERVAL_MS, self.process_queue)

    def build_ui(self):
        header = tk.Frame(self.root, bg=PANEL, height=66)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header, text="AI BODY TRACKER",
            bg=PANEL, fg=WHITE,
            font=("Segoe UI", 19, "bold")
        ).pack(side="left", padx=24)

        self.status = tk.Label(
            header, text="● OFFLINE",
            bg=PANEL, fg=RED,
            font=("Segoe UI", 10, "bold")
        )
        self.status.pack(side="right", padx=24)

        main = tk.Frame(self.root, bg=BACKGROUND)
        main.pack(fill="both", expand=True, padx=18, pady=18)

        video_frame = tk.Frame(main, bg="#050607")
        video_frame.pack(
            side="left", fill="both", expand=True, padx=(0, 18)
        )

        self.video = tk.Label(
            video_frame, bg="#050607",
            text="Натисни START, щоб увімкнути камеру",
            fg=MUTED, font=("Segoe UI", 13)
        )
        self.video.pack(fill="both", expand=True)

        panel = tk.Frame(main, bg=PANEL, width=235)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)

        tk.Label(
            panel, text="SYSTEM STATUS",
            bg=PANEL, fg=MUTED,
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=18, pady=(22, 12))

        self.fps_label = self.info(panel, "FPS", "0")
        self.people_label = self.info(panel, "PEOPLE", "0")
        self.points_label = self.info(panel, "VISIBLE BODY POINTS", "0")

        self.start_button = tk.Button(
            panel, text="▶  START",
            command=self.start,
            bg="#00a94f", fg="white",
            activebackground="#00c853",
            activeforeground="white",
            relief="flat", font=("Segoe UI", 10, "bold"),
            height=2, cursor="hand2"
        )
        self.start_button.pack(fill="x", padx=18, pady=(28, 7))

        self.stop_button = tk.Button(
            panel, text="■  STOP",
            command=self.stop,
            bg="#b3261e", fg="white",
            activebackground="#d50000",
            activeforeground="white",
            relief="flat", font=("Segoe UI", 10, "bold"),
            height=2, cursor="hand2", state="disabled"
        )
        self.stop_button.pack(fill="x", padx=18)

        tk.Label(
            panel,
            text="MediaPipe Pose Lite\nOpenCV • Tkinter\nAI runs in background",
            bg=PANEL, fg="#777d89",
            font=("Segoe UI", 9), justify="left"
        ).pack(side="bottom", anchor="w", padx=18, pady=20)

    @staticmethod
    def info(parent, title, value):
        box = tk.Frame(parent, bg=PANEL)
        box.pack(fill="x", padx=18, pady=8)

        tk.Label(
            box, text=title, bg=PANEL, fg=MUTED,
            font=("Segoe UI", 9)
        ).pack(anchor="w")

        label = tk.Label(
            box, text=value, bg=PANEL, fg=WHITE,
            font=("Segoe UI", 20, "bold")
        )
        label.pack(anchor="w")
        return label

    def start(self):
        if self.running:
            return

        self.frames = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.worker = CameraWorker(
            self.frames, self.stop_event, self.model_path
        )
        self.running = True
        self.fps = 0.0
        self.last_frame_time = time.perf_counter()

        self.status.config(text="● STARTING", fg="#ffca28")
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.worker.start()

    def stop(self):
        if self.stop_event is not None:
            self.stop_event.set()

        self.running = False
        self.status.config(text="● STOPPING", fg="#ffca28")
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.people_label.config(text="0")
        self.points_label.config(text="0")
        self.fps_label.config(text="0")
        self.video.config(image="", text="Камера зупинена")
        self.video.image = None
        self.photo = None

    def process_queue(self):
        try:
            while True:
                packet = self.frames.get_nowait()
                status = packet.get("status")

                if status == "live":
                    self.status.config(text="● LIVE", fg="#00e676")

                elif status == "error":
                    self.status.config(text="● CAMERA ERROR", fg=RED)
                    messagebox.showerror(
                        "Помилка AI Body Tracker",
                        packet.get("message", "Невідома помилка.")
                    )

                elif status == "stopped":
                    if self.running:
                        self.running = False
                        self.start_button.config(state="normal")
                        self.stop_button.config(state="disabled")
                        if self.status.cget("text") != "● CAMERA ERROR":
                            self.status.config(text="● OFFLINE", fg=RED)

                elif status == "frame":
                    self.show_frame(packet)

        except queue.Empty:
            pass

        if self.root.winfo_exists():
            self.root.after(UI_INTERVAL_MS, self.process_queue)

    def show_frame(self, packet):
        frame = packet["frame"]
        now = time.perf_counter()
        delta = now - self.last_frame_time
        self.last_frame_time = now

        if delta > 0:
            instant_fps = 1.0 / delta
            self.fps = (
                instant_fps if self.fps == 0
                else self.fps * 0.85 + instant_fps * 0.15
            )

        self.fps_label.config(text=str(round(min(self.fps, 120))))
        self.people_label.config(text=str(packet["people"]))
        self.points_label.config(text=str(packet["points"]))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        target_width = max(self.video.winfo_width(), 2)
        target_height = max(self.video.winfo_height(), 2)
        image.thumbnail(
            (target_width, target_height),
            Image.Resampling.LANCZOS
        )

        self.photo = ImageTk.PhotoImage(image)
        self.video.config(image=self.photo, text="")
        self.video.image = self.photo

    def close(self):
        if self.stop_event is not None:
            self.stop_event.set()

        self.running = False
        self.root.destroy()


def main():
    logging.basicConfig(level=logging.INFO)

    root = tk.Tk()
    root.withdraw()

    try:
        model_path = ensure_model()
    except Exception as error:
        messagebox.showerror(
            "Помилка завантаження моделі",
            f"Не вдалося завантажити MediaPipe модель:\n{error}"
        )
        root.destroy()
        return

    root.deiconify()
    BodyTracker(root, model_path)
    root.mainloop()


if __name__ == "__main__":
    main()
