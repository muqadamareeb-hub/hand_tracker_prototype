
# hand_tracker_prototype

A real-time hand tracking and visual effects system built with Python, OpenCV, and MediaPipe. Detects hand gestures from a live camera feed and generates synchronized visual effects based on detected gestures.

---


## Features

- Real-time hand landmark detection (21 points per hand)
- Hand skeleton rendering with joint connections
- Bezier curve trail effect with Z-depth based thickness
- EMA (Exponential Moving Average) smoothing for stable tracking
- Distance-based gesture recognition (orientation independent)
- Dot product thumb direction detection
- Two-hand pinch-to-zoom with normalized distance and drift correction
- FPS counter overlay

---

## Gesture Controls

| Gesture | Effect |
|---|---|
| ✋ Open Hand | Purple bezier trail follows index fingertip |
| ✊ Closed Fist | Trail clears + red screen flash |
| ✌️ Peace Sign | Exits the program |
| 🤏 Pinch | Trail color changes to blue |
| 👍 Thumbs Up | thumb extended fingers closed |
| 🤜🤛 Two Hands — One Fist + One Pinch | Pinch-to-zoom camera feed |

---

## Requirements

- Python 3.11
- Webcam or USB camera (tested with Pixel 6A in webcam mode)

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/hand_tracker_prototype.git
cd hand_tracker_prototype
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install opencv-python mediapipe numpy
```

**4. Download the MediaPipe hand landmark model**
```bash
curl -o hand_landmarker.task -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```
Or on Windows PowerShell:
```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "hand_landmarker.task"
```

**5. Run**
```bash
python handtrack.py
```

---

## Configuration

At the top of `handtrack.py` you can adjust these constants:

```python
SMOOTHING_ALPHA = 0.3      # lower = smoother, higher = more responsive
TRAIL_DURATION = 1         # trail length in seconds
FLASH_DURATION = 5         # flash duration in frames
PINCH_THRESHOLD = 0.08     # pinch sensitivity
THUMBS_UP_THRESHOLD = 0.02 # thumbs up sensitivity
```

---

## Camera Setup

By default the code uses camera index `1`. If using a built-in webcam change this line:

```python
cap = cv2.VideoCapture(1)  # change to 0 for built-in webcam
```

Press `Q` to quit.

---

## Tech Stack

- [Python 3.11](https://www.python.org/)
- [OpenCV](https://opencv.org/) — camera feed, frame processing, drawing
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) — hand landmark detection
- [NumPy](https://numpy.org/) — numerical operations

---

## Roadmap

- [ ] Threading for improved FPS
- [ ] Kalman filter for smoother tracking
- [ ] TouchDesigner integration for advanced visual effects
- [ ] Pan gesture support
- [ ] More gesture-to-effect mappings
- [ ] Pre-recorded video file support

---

## Author

Built as a prototype during my first year of engineering — started from zero Python knowledge.




