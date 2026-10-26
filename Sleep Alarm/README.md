# 🚗 Driver Safety Monitor

<p align="center">
  <strong>Real-Time Drowsiness Detection Using Computer Vision</strong>
  <br>
  A Python-based driver monitoring system that detects prolonged eye closure, downward head movement, and face-tracking loss to trigger an audio alert.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/MediaPipe-Face_Mesh-0097A7?style=for-the-badge" alt="MediaPipe">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Pygame-Audio-4B8BBE?style=for-the-badge" alt="Pygame">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License">
</p>

---

## 📌 Overview

**Driver Safety Monitor** is a real-time computer vision project built with Python, OpenCV, and MediaPipe Face Mesh. It analyzes facial landmarks through a webcam to monitor eye closure and downward head movement.

The system calibrates itself using the user's initial facial posture, establishes personalized detection thresholds, and monitors subsequent frames for potential drowsiness indicators. If the detected condition persists for a specified number of consecutive frames, an audio alarm plays to alert the user.

The application displays a live webcam feed with a heads-up display (HUD) showing eye and head-pose measurements, detection thresholds, and alarm status.

## ✨ Features

- 👁️ **Eye Closure Detection** — Calculates the Eye Aspect Ratio (EAR) to identify potentially prolonged eye closure.
- 🧠 **Head Pose Estimation** — Uses facial landmarks and OpenCV's `solvePnP()` to estimate head pitch.
- 🎯 **Automatic Calibration** — Uses 45 frames to establish baseline eye and head-pose measurements.
- 📊 **Real-Time HUD** — Displays EAR, pitch angle, detection thresholds, and system status.
- 🔊 **Audio Alert System** — Plays a looping alarm when a detected condition persists.
- 📉 **Noise Reduction** — Uses median filtering over a five-sample window to smooth measurements.
- ⏳ **Consecutive-Frame Detection** — Requires 24 consecutive frames of a detected condition before activating the alarm.
- 🎥 **Live Webcam Processing** — Processes video frames continuously using OpenCV.
- 🔄 **Manual Recalibration** — Press `C` to recalibrate the detection thresholds.
- 🛑 **Keyboard Controls** — Press `Q` to exit the application.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| OpenCV | Webcam capture, image processing, and head-pose estimation |
| MediaPipe Face Mesh | Facial landmark detection |
| NumPy | Numerical calculations and landmark processing |
| Pygame | Audio playback for alerts |
| Collections `deque` | Rolling windows for measurement smoothing |

## ⚙️ How It Works

The application follows a continuous detection pipeline:

```text
       Webcam Input
            |
            v
      Capture Frame
            |
            v
   MediaPipe Face Mesh
            |
            v
   Extract Facial Landmarks
            |
       +----+----+
       |         |
       v         v
  Eye Aspect   Head Pose
  Ratio (EAR)  Estimation
       |         |
       +----+----+
            |
            v
    Median Filtering
            |
            v
    Initial Calibration
            |
            v
  Compare Against Thresholds
            |
            v
 Consecutive-Frame Detection
            |
       +----+----+
       |         |
       v         v
 Normal State  Persistent
               Detection
                  |
                  v
              Audio Alarm
```

## 🧮 Detection Logic

### 1. Eye Aspect Ratio (EAR)

The Eye Aspect Ratio measures the relationship between the vertical and horizontal distances of selected eye landmarks.

The implementation calculates EAR using six landmarks for each eye and averages the left and right eye measurements.

\[
EAR = \frac{d_1+d_2}{2d_h}
\]

Where:

- \(d_1\) and \(d_2\) represent vertical distances between eye landmarks.
- \(d_h\) represents the horizontal distance between the eye corners.

A lower EAR indicates that the eyes may be closed.

### 2. Head Pose Estimation

The application estimates head orientation using six facial landmarks and a predefined 3D head model.

OpenCV's `cv2.solvePnP()` estimates the rotation of the head relative to the camera. The rotation matrix is then used to calculate a pitch angle.

If the estimated pitch falls below the calibrated threshold, the application flags downward head movement.

### 3. Automatic Calibration

During startup, the system collects 45 frames containing detected facial landmarks.

It calculates median baseline values and establishes personalized thresholds:

- **EAR threshold:** 72% of the baseline EAR.
- **Pitch threshold:** Baseline pitch minus 14 degrees.

Pressing `C` resets the calibration process.

### 4. Alarm Activation

The system flags a potential drowsiness condition when the eyes appear closed or the head is tilted downward. Loss of face detection after calibration also triggers the condition flag.

The alarm activates after 24 consecutive flagged frames. When the condition clears, the counter resets and the alarm stops.

**Note:** These thresholds are heuristic settings, not medically validated measures of drowsiness. Face-tracking loss can also occur because of lighting, camera position, or obstruction.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 recommended; dependency compatibility may vary by Python version.
- A working webcam.
- A microphone is not required.
- An audio output device for hearing alerts.

### 1. Clone the Repository

```bash
git clone https://github.com/Rudra1725/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace `YOUR-REPOSITORY` with your actual repository name.

### 2. Install Dependencies

```bash
pip install opencv-python numpy mediapipe pygame
```

If installation fails, check the Python version supported by the installed MediaPipe release.

### 3. Set Up the Audio File

Create an `assets` directory in the same folder as your Python script and place the alarm audio file inside it.

Your project structure should look like this:

```text
Driver-Safety-Monitor/
├── main.py
├── assets/
│   └── uth.mp3
├── README.md
├── LICENSE
└── .gitignore
```

The filename must be `uth.mp3` because the current code expects that exact path.

If the audio file is missing, the program prints a warning and continues without an audible alarm.

### 4. Run the Application

Save the source code as `main.py` and execute:

```bash
python main.py
```

Allow webcam access if your operating system requests permission.

## 🎮 Controls

| Key | Action |
|---|---|
| `C` | Reset and repeat calibration |
| `Q` | Quit the application |

## 📺 On-Screen Indicators

- **EAR:** Current smoothed eye aspect ratio and its threshold.
- **PITCH:** Estimated head pitch and its threshold.
- **Green indicator:** Alarm is inactive.
- **Red indicator:** Alarm is active.
- **WAKE UP!:** Alert displayed when the alarm is triggered.
- **Calibration progress:** Indicates progress during initial calibration.

## ⚠️ Limitations and Safety

- This project is an educational prototype, not a certified driver safety system.
- Detection accuracy depends on camera quality, lighting, face visibility, and calibration posture.
- Head movement and face-tracking loss may produce false alerts.
- The 24-frame threshold represents a number of video frames, not a fixed duration in seconds; the duration depends on processing speed and frame rate.
- The system does not directly measure sleepiness, attention, or driving ability.
- Do not rely on this application as a substitute for adequate rest or established vehicle safety systems. Never use it to justify continuing to drive while drowsy.

## 🔮 Future Improvements

- 📈 Add a configurable alarm sensitivity setting.
- ⏱️ Use elapsed time instead of a fixed frame count for alert timing.
- 👀 Detect prolonged eye closure and blink duration separately.
- 📊 Record detection events and session statistics.
- 🎛️ Add configurable audio alerts and volume controls.
- 🧪 Evaluate detection accuracy under different lighting and camera conditions.
- 🖥️ Add a dashboard for calibration and monitoring settings.

## 👨‍💻 Author

**Rudra1725**

GitHub: [github.com/Rudra1725](https://github.com/Rudra1725)

## 📄 License

This project is intended to be distributed under the MIT License. See the `LICENSE` file for details.

---

⭐ If you find this project useful for learning computer vision and Python, consider starring the repository!
