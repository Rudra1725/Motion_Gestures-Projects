# Meme Gesture Controller 🎭🖐️

A real-time computer vision application that detects facial expressions and hand gestures to dynamically overlay memes/videos onto your face and play synchronized audio effects using OpenCV, MediaPipe, and Pygame.

---

## ✨ Features

- **Face-First Architecture:** Expressions like mouth opening and looking upward take instant priority, temporarily disabling conflicting hand gestures.
- **Mutual Exclusivity & Debouncing:** Multi-frame hysteresis filters prevent noisy false triggers and overlapping audio channels.
- **Dynamic Face-Tracking Overlays:** Images and video frames continuously scale, translate, and blend directly over your tracked face mesh.
- **Smooth Alpha Transitions:** Custom slide & fade animations (up/down) when overlaying visuals onto the webcam feed.
- **Low-Latency Audio:** Multi-channel playback powered by `pygame.mixer` with automatic loops and instant state cleanup.

---

## 🎮 Trigger & Gesture Mapping

| Action / Gesture | Physical Pose | Asset Output | Output Type |
| :--- | :--- | :--- | :--- |
| **Shock / Gasp** | Open mouth wide ($MAR > 0.32$) | `shock.jpg` + `faaah.mp3` | Image Overlay + Audio |
| **Looking Up** | Tilt head upward toward ceiling | `up.jpg` | Image Overlay |
| **Stop Palm** | Open palm pushed forward toward lens | `meme.jpg` | Image Overlay |
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
├── app.py                  # Main real-time processing loop
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation
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
