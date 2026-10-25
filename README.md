# Face & Eye-Gaze Verification System

A real-time AI-powered biometric verification web application combining face matching with eye-gaze tracking and head orientation estimation. Built with **FastAPI**, **MediaPipe**, **OpenCV**, and **React + Vite**.

---

## 📋 Prerequisites

Before running the project, ensure you have:
- **Python 3.9 – 3.11** (Required: `mediapipe==0.10.14` requires Python 3.9–3.11 for prebuilt wheels and stable `mp.solutions` API support)
- **Node.js** (v18.x or later) & **npm**
- A working **Webcam**

---

## 🐳 Docker Quickstart (Recommended)

You can build and spin up the complete stack (both backend and frontend) with a single command:

```bash
# Build and run containers in the foreground
docker compose up --build

# Or run in the background (detached mode)
docker compose up --build -d
```

- **Frontend App:** [http://localhost:3000](http://localhost:3000)
- **Backend API:** [http://localhost:8000](http://localhost:8000)
- **API Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

To view logs or stop the containers:
```bash
# View backend & frontend logs
docker compose logs -f

# Stop and remove containers
docker compose down
```

---

## 🚀 Manual Local Setup (Without Docker)

### 2. Run the Backend (FastAPI)

Open a new terminal window:

```bash
# Navigate to the backend directory
cd backend

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows (PowerShell / Command Prompt):
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

# Install required Python packages
pip install -r requirements.txt

# Start the FastAPI backend server
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- **Backend URL:** [http://localhost:8000](http://localhost:8000)
- **API Swagger Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

---

### 3. Run the Frontend (React + Vite)

Open a **second** terminal window:

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the Vite development server
npm run dev
```

- **Frontend URL:** [http://localhost:5173](http://localhost:5173)

---

## 🎯 How to Use the App

1. Open [http://localhost:5173](http://localhost:5173) in your browser.
2. Grant camera permissions when prompted.
3. **Reference Capture**: Take a snapshot or upload a reference face photo. The AI will register your biometric landmarks and anchor points.
4. **Live Verification**: Begin the live verification stream. The system will continuously verify:
   - **Identity Match**: Facial landmark geometry compared against reference photo.
   - **Eye-Gaze Direction**: Real-time iris positioning (CENTER, LEFT, RIGHT, UP, DOWN).
   - **Gaze Deviation Alerts**: Flags when gaze shifts away from screen.
   - **Multi-Face & Absence Detection**: Flags if multiple faces or no face appears in frame.
5. **Session Results**: Click **Stop Session** to view an analytics dashboard with match score %, gaze deviation breakdown, and event timestamps.

---

## 📁 Project Structure

```
face_gaze_demo/
├── .gitignore               # Git ignore rules for frontend and backend
├── README.md                # Project documentation & run guide
├── backend/
│   ├── database.py          # In-memory thread-safe session database & state tracking
│   ├── evaluator.py         # Pure functional evaluation, interval tracker & independent verdicts (L10)
│   ├── main.py              # FastAPI endpoints & CORS configuration
│   ├── vision_engine.py     # YuNet face detector, SFace recognizer & MediaPipe Face Mesh
│   ├── requirements.txt     # Python dependencies (including pytest)
│   ├── tests/
│   │   └── test_rules.py    # Offline pytest test suite for intervals, baseline & verdicts (L10)
│   └── models/
│       └── schemas.py       # Pydantic data schemas
└── frontend/
    ├── index.html           # HTML entry point
    ├── vite.config.js       # Vite configuration
    ├── package.json         # Frontend dependencies & scripts
    └── src/
        ├── App.jsx          # Main application & step management
        ├── CameraCapture.jsx# Reference photo capture component
        ├── LiveVerification.jsx # Live camera feed, face box & gaze tracking
        ├── SessionResults.jsx # Verification summary, intervals & metrics
        └── config.js        # Centralized API base URL (L2)
```

---

## 🧪 Running Automated Tests

A comprehensive, offline unit test suite is included to test rule evaluation, interval tracking, and verdict decisions without requiring camera hardware:

```bash
cd backend
.\.venv\Scripts\pytest tests/test_rules.py -v
```

All tests execute in milliseconds using in-memory synthetic record batches.

---

## 🔍 About this Demo: Models & Limitations (L8)

| Component | Model / Engine | Purpose | Operational Limitations |
| :--- | :--- | :--- | :--- |
| **Face Detection** | OpenCV YuNet (`face_detection_yunet_2023mar.onnx`) | Real-time face bounding box & 5-point facial landmarks | Requires well-lit frontal face; extreme profile angles (>60°) or heavy occlusion may drop detection. |
| **Face Recognition** | OpenCV SFace (`face_recognition_sface_2021dec.onnx`) | 128-D cosine similarity embedding matching | Matches 2D appearance against reference snapshot. Threshold tuned to 0.363 for high-confidence match. |
| **Gaze & Head Pose** | MediaPipe Face Mesh (478 3D landmarks) + SolvePnP | Iris horizontal/vertical displacement + 3D head yaw/pitch | Baseline-calibrated to each user's camera mounting position (M4). Rapid head jerks or extreme eye squinting may register as unknown gaze. |
| **Liveness / Anti-Spoofing** | Heuristic (Blink & Pose variance monitoring) | Warns on static photo printouts held still for 30+ seconds | Baseline behavioral safeguard only; does not replace dedicated passive 3D anti-spoofing models (M8). |

---

## ⚠️ Known Limitations & Anti-Spoofing (Liveness)

### Current Demo Scope
This application is designed as a demonstration of 2D facial recognition and eye-gaze tracking using standard webcams. **It does not incorporate full 3D passive anti-spoofing or active challenge-response liveness models.**
- A high-resolution printed photograph or phone screen displaying the enrolled user can pass facial matching.
- An informative disclaimer is displayed on the audit results screen: *"Anti-Spoofing / Liveness Note: No 3D passive anti-spoofing or active challenge-response check is enforced in this demo."*

### Heuristic Static Face Detection (Included)
To provide a baseline safeguard against static printouts and frozen video feeds, the backend incorporates a lightweight behavioral check:
- **Blink Monitoring**: Tracks natural eye-blinking via normalized eyelid openness ratios (`openness < 0.18`).
- **Pose Micro-Movements**: Tracks standard deviation of head yaw and pitch over trailing 30-second windows.
- If a face maintains biometric matching continuously for **30+ seconds with 0 blinks and near-zero head movement** (`std < 0.35°`), the engine emits a `STATIC_FACE_WARNING` audit event and flags the face assessment as `REVIEW` (`static_face_detected`).

### Recommended Production Architecture (Future Work)
For enterprise or proctoring environments requiring tamper-proof anti-spoofing:
1. **Passive Single-Frame Liveness**: Integrate a neural network (such as MiniFASNet, Silent-Face-Anti-Spoofing, or CDCN) trained on physical texture, screen moiré artifacts, and planar reflection cues.
2. **Active Challenge-Response Verification**: Issue randomized on-screen instructions during verification (e.g., "Blink twice", "Turn head 20° to the left", or "Track the moving on-screen target with your eyes").
3. **Hardware-Assisted Depth**: Utilize RGB-D depth sensors (Intel RealSense, Apple FaceID TrueDepth, or IR cameras) to reject planar representations.
