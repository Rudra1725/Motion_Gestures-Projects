# 🎭🖐️ Meme Gesture Controller

A real-time computer vision application that detects facial expressions and hand gestures to dynamically overlay memes/videos onto your face and play synchronized audio effects using OpenCV, MediaPipe, and Pygame.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20%26%20Hand%20Tracking-orange)](https://developers.google.com/mediapipe)
[![Pygame-CE](https://img.shields.io/badge/Pygame--CE-Audio%20Engine-red)](https://pyga.me/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 💡 Overview

The **Meme Gesture Controller** leverages machine learning pipelines to detect real-time kinematic markers without requiring custom sensor hardware:
* Tracks **468 3D facial landmarks** via MediaPipe Face Mesh to compute real-time metrics such as Mouth Aspect Ratio (MAR) and 3D head pitch orientation.
* Identifies **21 3D hand landmarks** to classify discrete physical gestures.
* Performs dynamic affine transformations, animated alpha compositing, and multi-channel low-latency audio playback simultaneously.

---

## 🛠️ Built With

* **Python 3.12** – Core programming language.
* **OpenCV (`opencv-python`)** – Real-time webcam frame acquisition, bounding-box transformation, animated alpha blending, and rendering.
* **MediaPipe** – Face Mesh (468 3D facial landmarks for Mouth Aspect Ratio and 3D head pitch) and Hand tracking.
* **Pygame Community Edition (`pygame-ce`)** – Low-latency multi-channel audio playback and loop management.
* **NumPy** – Vector arithmetic for landmark distance and spatial ratio calculations.

---

## 🎮 Trigger & Gesture Mapping

| Action / Gesture | Physical Pose | Asset Output | Output Type |
| :--- | :--- | :--- | :--- |
| **Shock / Gasp** | Open mouth wide ($MAR > 0.32$) | `shock.jpg` + `faaah.mp3` | Image Overlay + Audio |
| **Looking Up** | Tilt head upward toward ceiling | `up.jpg` | Image Overlay |
| **Stop Palm** | Open palm pushed forward toward camera | `meme.jpg` | Image Overlay |
| **Jaldi Callout** | Cupped / C-shaped hand rolled at lips | `jaldi.mp4` + `jaldi_audio.wav` | Video Overlay + Audio |
| **Shh** | Single index finger upright on lips | `aprit.mp3` | Audio Only |
| **Hi / Wave** | Open palm held upright in air | `rom.mp3` | Audio Only |
| **Head Grab** | Hand placed on forehead / hairline | `arebap.mp3` | Audio Only |
| **Phone to Ear** | Hand held beside ear tragion | `kuchu.mp3` | Audio Only |
| **Hands Raised** | Both hands held above head level | `dep.mp3` | Audio Only |

---

## 📁 Repository Structure

```text
meme-gesture-controller/
│
├── app.py                  # Main real-time detection & overlay loop
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation
├── .gitignore              # Ignored files (venv, cache, raw media)
│
└── Assets/                 # Media assets directory
    ├── meme.jpg
    ├── shock.jpg
    ├── up.jpg
    ├── jaldi.mp4
    ├── jaldi_audio.wav
    ├── aprit.mp3
    ├── arebap.mp3
    ├── dep.mp3
    ├── faaah.mp3
    ├── kuchu.mp3
    └── rom.mp3
