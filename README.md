<div align="center">

# ⚡ FastPoseTk (AI Body Tracker)

**A lightweight, lightning-fast real-time body and pose tracking desktop application built with Python, Tkinter, OpenCV, and MediaPipe Lite.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Latest-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20Lite-orange.svg)](https://developers.google.com/mediapipe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRzkwRy-T4yUOA9De7zui5oSLQq88B-e9P3Fv0JluYw1A&s=10" alt="AI Body Tracker Preview" width="700px" style="border-radius: 10px;">
</p>

---

## 🧠 What it does

**FastPoseTk** is a lightweight, real-time computer vision desktop application that tracks human body posture and landmarks using Google's MediaPipe and OpenCV, presented in a modern dark-themed Tkinter dashboard.

* 🚀 **Performance Optimization:** Performs AI inference on every $N$-th frame (`AI_EVERY_N_FRAMES = 2`), keeping the video stream and user interface butter-smooth at 30 FPS.
* 🦴 **Custom Skeleton Rendering:** Draws body joints and custom head bounding boxes without facial clutter.
* 🖥️ **Interactive Dark UI:** Built natively with Tkinter featuring real-time system metrics (FPS, active people count, and body points tracker).
* 🪶 **Plug & Play:** Automatically downloads the required lightweight MediaPipe pose model on the first launch.

---

## 🛠️ Tech Stack

* **GUI:** Tkinter, Pillow (PIL)
* **Computer Vision:** OpenCV (`cv2`)
* **AI/ML:** Google MediaPipe Tasks Vision (`PoseLandmarker`)
* **Language:** Python 3.x

---

## 🚀 How to Run

Follow these simple steps to set up and run the project locally:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/fast-pose-tk.git](https://github.com/YOUR_USERNAME/fast-pose-tk.git)
   cd fast-pose-tk
