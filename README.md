HAND TRACKER PROTOTYPE

Real-time hand gesture tracking and interaction system using computer vision. Built with Python, OpenCV, and MediaPipe.

This project detects hand landmarks from a live camera feed and generates visual effects such as motion trails, gesture recognition, and pinch-to-zoom interaction.

####  FEATURES

- Real-time hand landmark detection (21 points per hand)
- Hand skeleton rendering with joint connections
- Smooth Bezier curve trail effects
- EMA smoothing for stable tracking
- Gesture recognition using geometric rules
- Thumb direction detection using vector math
- Two-hand pinch-to-zoom interaction
- FPS counter display

####  HOW IT WORKS

The system uses MediaPipe to detect 21 hand landmarks per frame from a webcam feed.

These landmarks are used to:
- Track finger positions
- Detect finger states (open/closed)
- Calculate distances and angles
- Recognize gestures using rule-based logic
- Trigger visual effects like trails, zoom, and flash

####  GESTURE CONTROLS

- Open Hand → Purple trail follows finger
- Closed Fist → Clears trail + red flash effect
- Peace Sign → Exit program
- Pinch → Blue trail mode
- Thumbs Up → Thumb-based gesture detection
- Two Hands (Fist + Pinch) → Pinch-to-zoom

####  REQUIREMENTS

- Python 3.11
- Webcam

####  INSTALLATION

--1. Clone repository:
git clone https://github.com/yourusername/hand_tracker_prototype.git
cd hand_tracker_prototype

--2. Create virtual environment:
python -m venv venv
source venv/bin/activate

--3. Install dependencies:
pip install opencv-python mediapipe numpy

####  DOWNLOAD MODEL FILE

--macOS / Linux:
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

--Windows:
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "hand_landmarker.task"

####  Place file in project root:
hand_tracker_prototype/

####  IMPORTANT

hand_tracker_prototype/
- handtrack.py
- hand_landmarker.task

####  RUN

python handtrack.py

Press Q to quit.

####  CONFIGURATION

SMOOTHING_ALPHA = 0.3
TRAIL_DURATION = 1
FLASH_DURATION = 5
PINCH_THRESHOLD = 0.08
THUMBS_UP_THRESHOLD = 0.02

####  TECH STACK

- Python
- OpenCV
- MediaPipe
- NumPy

AUTHOR

Built it as a first year engineering student with no previous experience in python
