# 👁️ Face & Eye-Gaze Verification System

[![Python](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-YuNet%20%2B%20SFace-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh%20(478%20pts)-00B4D8)](https://mediapipe.dev/)
[![React](https://img.shields.io/badge/React-19.x-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8.x-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![Docker](https://img.shields.io/badge/Docker-Compose%20Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A real-time, AI-powered biometric identity and attention verification platform. It combines **128-D facial feature recognition**, a **478-point 3D facial landmark mesh**, **sub-millimeter iris gaze tracking**, and **head pose orientation (Yaw/Pitch/Roll)** to continuously verify user identity and monitor attention through a standard commodity webcam.

---

## 📑 Table of Contents

- [🌟 System Architecture](#-system-architecture)
- [✨ Key Features](#-key-features)
- [📋 Prerequisites & Compatibility](#-prerequisites--compatibility)
- [🐳 Method 1: Docker Quickstart (Recommended)](#-method-1-docker-quickstart-recommended)
- [🚀 Method 2: Manual Local Setup (Step-by-Step)](#-method-2-manual-local-setup-step-by-step)
  - [Step 1: Clone and Inspect Project](#step-1-clone-and-inspect-project)
  - [Step 2: Database Setup (PostgreSQL or In-Memory)](#step-2-database-setup-postgresql-or-in-memory)
  - [Step 3: Backend Setup (FastAPI)](#step-3-backend-setup-fastapi)
  - [Step 4: Frontend Setup (React + Vite)](#step-4-frontend-setup-react--vite)
- [🎯 How to Use the Application (Walkthrough)](#-how-to-use-the-application-walkthrough)
- [⚖️ Verdict Decision Logic & Rules](#️-verdict-decision-logic--rules)
- [📁 Project Structure](#-project-structure)
- [⚙️ Configuration Reference (`config.py`)](#️-configuration-reference-configpy)
- [🔌 API Endpoints Reference](#-api-endpoints-reference)
- [🧪 Running Automated Tests](#-running-automated-tests)
- [🔍 AI Models & Pipeline Details](#-ai-models--pipeline-details)
- [⚠️ Limitations & Anti-Spoofing Architecture](#️-limitations--anti-spoofing-architecture)
- [🛠️ Troubleshooting & FAQ](#️-troubleshooting--faq)
- [📄 License](#-license)

---

## 🌟 System Architecture

```mermaid
flowchart TD
    subgraph Browser ["Frontend (React 19 + Vite @ localhost:3000 / 5173)"]
        Cam[Webcam Stream] --> Cap[CameraCapture: Reference Photo]
        Cam --> Live[LiveVerification: 5 FPS Frames]
        Live -. "POST /api/verification/frame (Base64 JPEG)" .-> API
        LiveHud[Real-time HUD Overlay: Match %, Gaze Direction, Pose]
        ReportUI[Session Report & Proof Snapshots UI]
        HistoryUI[Session History & Audit View]
    end

    subgraph Backend ["Backend API (FastAPI @ localhost:8000)"]
        API[FastAPI Router: main.py]
        
        subgraph VisionEngine ["Vision Pipeline: engine.py"]
            YuNet["OpenCV YuNet (Face Detection & BBox)"]
            SFace["OpenCV SFace (128-D Embedding & Cosine Sim)"]
            Mesh["MediaPipe Face Mesh (478 3D Landmarks & Iris)"]
            PnP["OpenCV SolvePnP (Head Yaw / Pitch / Roll)"]
        end
        
        subgraph Trackers ["State Machine & Evaluation"]
            Calib["Adaptive Baseline Tracker (First 5s)"]
            Hyst["Hysteresis Interval Tracker (Smooths Jitter)"]
            Liveness["Static-Face / Photo Heuristic Tracker"]
            Eval["Verdict Evaluator (PASSED / SUSPICIOUS / FAILED)"]
        end
        
        Store[Thread-Safe Session Store]
    end

    subgraph Persistence ["Persistence Layer"]
        DB[(PostgreSQL 16 DB)]
        MemFallback["In-Memory Cache (Graceful Fallback)"]
    end

    API --> VisionEngine
    VisionEngine --> Trackers
    Trackers --> Store
    Store --> DB
    Store -. "Fallback if DB unavailable" .-> MemFallback
    API -. "JSON Analysis" .-> LiveHud
    API -. "Final Report" .-> ReportUI
    DB -. "Audit Queries" .-> HistoryUI
```

---

## ✨ Key Features

- **👤 Biometric Face Identification**:
  - Real-time frontal face detection via **OpenCV YuNet**.
  - 128-D deep metric face embedding extraction and cosine similarity verification via **OpenCV SFace**.
- **👁️ Eye-Gaze & Iris Landmark Tracking**:
  - Sub-millimeter iris center displacement tracking using **MediaPipe Face Mesh** with 478 3D dense facial landmarks.
  - 3D head yaw, pitch, and roll calculation via **OpenCV SolvePnP** against a canonical 3D facial model.
- **⚖️ Adaptive Baseline Calibration**:
  - Automatically calibrates the user's neutral screen-facing posture during the first **5 seconds** of a session.
  - Compensates for variations in webcam positioning, display angle, and ergonomic sitting heights.
- **⏱️ Hysteresis Interval Violation Tracking**:
  - Filters out transient single-frame noise (e.g., natural quick glances or camera autofocus hiccups).
  - Requires sustained anomalies (looking away, missing face, multiple faces, unknown identity) before recording violations.
- **🛡️ Heuristic Static-Face Safeguard**:
  - Monitors micro-movements (head pose standard deviation) and natural blink frequency over 30-second sliding windows.
  - Flags potential photo spoofing if a motionless 2D printout or static image is placed in front of the lens.
- **💾 Comprehensive Audit Reports & Proof Snapshots**:
  - Calculates detailed session metrics: Match %, Gaze deviations, Total duration, and Violation timelines.
  - Automatically captures visual thumbnail snapshots (**Proof of Violation**) during rule breaches for audit review.
  - Stores all records in **PostgreSQL**, with an automatic in-memory fallback if the database is offline.

---

## 📋 Prerequisites & Compatibility

Before starting, ensure your host environment meets the following specifications:

| Requirement | Supported Version | Notes |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11, macOS, or Ubuntu 20.04+ | Cross-platform compatible. |
| **Python** | **`3.9`**, **`3.10`**, or **`3.11`** | ⚠️ **Crucial**: MediaPipe (`==0.10.14`) wheels and the stable `mp.solutions` API require Python 3.9–3.11. Python 3.12+ will fail during `pip install mediapipe`. |
| **Node.js & npm** | `v18.x` or higher (LTS recommended) | Required for frontend build and dependencies. |
| **Webcam** | Built-in or USB webcam | Must be functional and accessible via WebRTC. |
| **Browser** | Google Chrome, Edge, Firefox, or Brave | Requires camera permission enabled. |
| **Docker & Docker Compose** | Docker Desktop 20+ *(Optional)* | Only required if running via Docker Quickstart. |

---

## 🐳 Method 1: Docker Quickstart (Recommended)

The simplest way to run the entire application (PostgreSQL, FastAPI Backend, and React Frontend via Nginx) with zero manual environment configuration:

### 1. Start the stack

Open a terminal in the project root directory (`face_gaze_demo`):

```bash
# Build and start all containers in foreground
docker compose up --build

# Or start in detached mode (background)
docker compose up --build -d
```

### 2. Access the services

| Service | URL | Description |
| :--- | :--- | :--- |
| **Frontend Web App** | [http://localhost:3000](http://localhost:3000) | Full verification UI and session dashboard |
| **Backend API** | [http://localhost:8000](http://localhost:8000) | FastAPI REST service |
| **Interactive API Docs (Swagger)** | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive Swagger UI for testing endpoints |
| **API Health Check** | [http://localhost:8000/health](http://localhost:8000/health) | Backend and database readiness status |
| **PostgreSQL Database** | `localhost:5432` | Persisted relational database |

### 3. Manage the containers

```bash
# Check status of running containers
docker compose ps

# View live streaming logs across all services
docker compose logs -f

# View logs for a specific service (e.g., backend)
docker compose logs -f backend

# Stop all containers safely
docker compose down

# Stop containers and remove persisted volumes (resets database)
docker compose down -v
```

---

## 🚀 Method 2: Manual Local Setup (Step-by-Step)

If you prefer running services directly on your host machine without Docker:

### Step 1: Clone and Inspect Project

Ensure your working directory is set to the project root:

```bash
cd face_gaze_demo
```

Verify that the pretrained model weights exist in `backend/weights/`:
- `face_detection_yunet_2023mar.onnx` (~233 KB)
- `face_recognition_sface_2021dec.onnx` (~38.7 MB)

---

### Step 2: Database Setup (PostgreSQL or In-Memory)

You have two options for the database:

#### Option A: Run PostgreSQL via Docker (Simplest)
```bash
docker compose up -d db
```

#### Option B: Use an Existing Local PostgreSQL Server
Ensure PostgreSQL is running locally on port `5432`. Create a database named `face_gaze_db` and verify credentials in `backend/.env`:

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=Admin@123
POSTGRES_DB=face_gaze_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
DATABASE_URL=postgresql+psycopg2://postgres:Admin%40123@localhost:5432/face_gaze_db
```

> 💡 **Graceful Fallback**: If you do not have PostgreSQL installed, you can skip this step! The backend will log a warning during startup and seamlessly fall back to its thread-safe in-memory cache. Active verification and reporting will still function fully.

---

### Step 3: Backend Setup (FastAPI)

Open a terminal and navigate into the `backend/` directory:

```bash
cd backend
```

#### 3.1 Create a Python Virtual Environment (Python 3.9–3.11)

- **Windows (PowerShell or CMD)**:
  ```powershell
  python -m venv .venv
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv .venv
  ```

#### 3.2 Activate the Virtual Environment

- **Windows PowerShell**:
  ```powershell
  # If PowerShell script execution is restricted, run:
  # Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  .\.venv\Scripts\Activate.ps1
  ```
- **Windows Command Prompt (cmd.exe)**:
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
- **macOS / Linux**:
  ```bash
  source .venv/bin/activate
  ```

#### 3.3 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3.4 Start the FastAPI Server

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Confirm that the backend is running:
- **API URL:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

### Step 4: Frontend Setup (React + Vite)

Open a **new, separate terminal window** and navigate into the `frontend/` directory:

```bash
cd frontend
```

#### 4.1 Install Dependencies

```bash
npm install
```

#### 4.2 Start the Development Server

```bash
npm run dev
```

The terminal will display the local development URL:
- **Local Application URL:** [http://localhost:5173](http://localhost:5173)

---

## 🎯 How to Use the Application (Walkthrough)

Follow this 6-step walkthrough to test the verification pipeline:

### 1. Open the Web Application
Open your browser and navigate to:
- **Docker mode**: [http://localhost:3000](http://localhost:3000)
- **Manual local mode**: [http://localhost:5173](http://localhost:5173)

### 2. Allow Camera Access
When prompted by your browser, click **Allow** to grant access to your webcam.

### 3. Capture Reference Photo (Enrollment)
1. Position your face in the center of the video preview under good, even lighting.
2. Click **Capture Reference** (or upload a clear portrait image).
3. The system executes automated quality checks:
   - ✅ Face detected & single individual present
   - ✅ Face fills sufficient frame area (> 5% of pixels, min 80px)
   - ✅ Image sharpness (Laplacian variance > 30.0)
   - ✅ Balanced lighting (brightness between 35.0 and 220.0)
   - ✅ Neutral frontal pose (yaw/pitch angle within ±25°)
   - ✅ Eyes open
4. Once validated, the system registers the 128-D biometric embedding and enables the session.

### 4. Baseline Calibration Phase (First 5 Seconds)
1. Click **Start Live Verification**.
2. A green progress bar will indicate the **5-second calibration window**.
3. **Important**: Look naturally at the center of your screen with normal posture.
4. The system collects samples to calculate your neutral pitch, yaw, and iris coordinates.

### 5. Continuous Live Verification
Once calibrated, the real-time HUD displays live tracking metrics at ~5 FPS:
- **Identity Match %**: Biometric cosine similarity score compared against your reference photo.
- **Gaze Direction**: Indicates `CENTER`, `LEFT`, `RIGHT`, `UP`, or `DOWN`.
- **Face Status**: Badges for `MATCH`, `NO_MATCH`, `NO_FACE`, or `MULTIPLE_FACES`.
- **Orientation Compass**: Live Euler angles (Yaw and Pitch degrees).
- **Attention Alerts**: Warns immediately if you look away or if multiple faces appear.

### 6. Stop Session & Review Audit Report
1. Click **Stop Session** at any time.
2. The evaluator compiles an audit report:
   - **Overall Verdict**: `PASSED`, `SUSPICIOUS`, or `FAILED`.
   - **Performance Metrics**: Identity match percentage, gaze deviation rate, and session duration.
   - **Interval Breakdown Timeline**: Interactive timeline showing exact start and end timestamps of each violation.
   - **Failure Proof Snapshots**: Visual thumbnail evidence captured at the moment any rule was breached.
3. Click the **Session History** tab in the top navigation bar to inspect past sessions stored in PostgreSQL.

---

## ⚖️ Verdict Decision Logic & Rules

The session verdict is determined by evaluating all recorded frame intervals against predefined security rules (`evaluator.py`):

### 1. Facial Identity Verdict (`face_verdict`)

| Condition | Threshold | Outcome | Reason Code |
| :--- | :--- | :--- | :--- |
| **Different Person Detected** | Sustained `NO_MATCH` $\ge 2.0$ seconds | **FAIL** | `different_person` |
| **Multiple Faces in Frame** | Sustained `MULTIPLE_FACES` $\ge 5.0$ seconds | **FAIL** | `multiple_faces` |
| **Face Not Detected (Severe)** | Missing face $> 30\%$ of session duration | **FAIL** | `no_face` |
| **Face Not Detected (Moderate)**| Missing face $> 15\%$ of session duration | **REVIEW** | `no_face` |
| **Low Quality / Ambiguous Pose**| `UNKNOWN` identity $> 25\%$ of session | **REVIEW** | `identity_unknown` |
| **Static Photo / No Blinks** | 0 blinks + pose variance $< 0.6^\circ$ over 30s | **REVIEW** | `static_face` |

### 2. Attention & Eye-Gaze Verdict (`gaze_verdict`)

| Condition | Threshold | Outcome | Reason Code |
| :--- | :--- | :--- | :--- |
| **Prolonged Distraction** | Single continuous looking-away interval $\ge 10.0$ seconds | **FAIL** | `long_away` |
| **Excessive Looking Away** | Total looking away $> 35\%$ of session | **FAIL** | `looking_away` |
| **Moderate Looking Away** | Total looking away $> 20\%$ of session | **REVIEW** | `looking_away` |
| **Unstable Calibration** | Calibration incomplete or high standard deviation | **REVIEW** | `no_stable_baseline` |
| **Gaze Untracked** | Gaze undetected $> 25\%$ of session | **REVIEW** | `gaze_unknown` |

### 3. Overall Aggregate Verdict

```
                    ┌───────────────────────────────┐
                    │ Face Verdict + Gaze Verdict   │
                    └──────────────┬────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
   Either is FAIL           Either is REVIEW         Both are PASS
         │                         │                         │
         ▼                         ▼                         ▼
    💥 FAILED                ⚠️ SUSPICIOUS              ✅ PASSED
```

---

## 📁 Project Structure

```
face_gaze_demo/
├── docker-compose.yml              # Multi-container orchestration (DB, Backend, Frontend)
├── .gitignore                      # Git ignore file
├── README.md                       # Comprehensive documentation
│
├── backend/
│   ├── Dockerfile                  # Container definition for FastAPI backend
│   ├── requirements.txt            # Python dependencies (FastAPI, OpenCV, MediaPipe, etc.)
│   ├── pytest.ini                  # Pytest configuration settings
│   ├── .env                        # PostgreSQL database connection credentials
│   ├── config.py                   # Central configuration & tunable threshold parameters
│   ├── main.py                     # FastAPI routes, CORS, lifespan, and HTTP endpoints
│   ├── db_models.py                # SQLAlchemy ORM models & PostgreSQL database layer
│   ├── session_store.py            # Thread-safe in-memory session manager & state machine
│   ├── evaluator.py                # Session evaluation engine & rule evaluation logic
│   ├── intervals.py                # Hysteresis interval tracker for gaze and identity
│   ├── records.py                  # Dataclasses for frame records and state constants
│   ├── models/
│   │   └── schemas.py              # Pydantic v2 validation models for requests/responses
│   ├── vision/
│   │   ├── engine.py               # Unified vision inference pipeline orchestrator
│   │   ├── face.py                 # OpenCV YuNet detection & SFace feature extraction
│   │   └── gaze.py                 # MediaPipe Face Mesh iris tracking & SolvePnP head pose
│   ├── weights/
│   │   ├── face_detection_yunet_2023mar.onnx      # Pretrained YuNet ONNX model (~233 KB)
│   │   └── face_recognition_sface_2021dec.onnx   # Pretrained SFace ONNX model (~38.7 MB)
│   └── tests/
│       ├── helpers.py              # Synthetic data generators & test mocks
│       ├── test_database_store.py  # Unit tests for SQLAlchemy storage layer
│       ├── test_evaluator.py       # Unit tests for verdict rules & scoring
│       ├── test_face_select.py     # Unit tests for multi-face priority selection
│       ├── test_gaze.py            # Unit tests for gaze angles, iris math & baseline
│       ├── test_intervals.py       # Unit tests for hysteresis opening/closing trackers
│       ├── test_reference_checks.py# Unit tests for reference photo validation
│       ├── test_session_store.py   # Unit tests for session lifecycle management
│       ├── test_vision.py          # Vision engine landmark & alignment unit tests
│       └── test_e2e.py             # End-to-end integration test
│
└── frontend/
    ├── Dockerfile                  # Multi-stage production container build (Vite -> Nginx)
    ├── nginx.conf                  # Nginx reverse proxy & SPA router config
    ├── package.json                # Frontend dependencies & npm scripts
    ├── vite.config.js              # Vite configuration
    ├── index.html                  # HTML5 application entry point
    └── src/
        ├── main.jsx                # React root application bootstrapper
        ├── App.jsx                 # Application state controller & navigation
        ├── config.js               # Centralized backend API base URL config
        ├── components/
        │   ├── CameraCapture.jsx   # Reference photo capture & quality validation UI
        │   ├── LiveVerification.jsx# Real-time webcam inference loop & HUD overlay
        │   ├── SessionReport.jsx   # Post-session audit report & failure proofs UI
        │   ├── SessionHistory.jsx  # Past sessions history list & inspection
        │   └── ui.jsx              # UI component library (Cards, Buttons, Badges)
        ├── css/                    # Component-specific stylesheet modules
        └── lib/                    # Client-side utility functions
```

---

## ⚙️ Configuration Reference (`config.py`)

All threshold numbers and timeouts are centralized in [backend/config.py](file:///c:/Users/Srushti/Desktop/face_gaze_demo/backend/config.py) and can be overridden via environment variables:

| Environment Variable | Default | Purpose |
| :--- | :--- | :--- |
| `FRAME_PERIOD_S` | `0.2` | Expected interval between frontend frames (5 FPS = 200 ms). |
| `SFACE_MATCH_THRESHOLD` | `0.363` | SFace cosine similarity threshold for biometric identity match. |
| `YUNET_SCORE_THRESHOLD` | `0.8` | YuNet confidence threshold for accepting face detections. |
| `CALIBRATION_SECONDS` | `5.0` | Initial baseline calibration duration (seconds). |
| `CALIBRATION_MIN_SAMPLES` | `20` | Minimum frames required to compute valid baseline pose. |
| `GAZE_YAW_AWAY_DEG` | `20.0` | Horizontal head rotation angle threshold to flag looking away. |
| `GAZE_PITCH_DOWN_DEG` | `15.0` | Downward pitch rotation threshold to flag looking away. |
| `GAZE_PITCH_UP_DEG` | `18.0` | Upward pitch rotation threshold to flag looking away. |
| `GAZE_OPEN_AFTER` | `5` | Consecutive looking-away frames before opening a violation interval. |
| `GAZE_CLOSE_AFTER` | `3` | Consecutive center frames before closing a violation interval. |
| `FACE_DIFFERENT_PERSON_FAIL_S` | `2.0` | Sustained different person duration to trigger FAIL verdict. |
| `FACE_MULTIPLE_FACES_FAIL_S` | `5.0` | Sustained multiple faces duration to trigger FAIL verdict. |
| `STATIC_FACE_MIN_DURATION_S` | `30.0`| Minimum session length before checking static photo heuristic. |
| `ALLOWED_ORIGINS` | `http://localhost:3000,...` | Comma-separated list of allowed CORS browser origins. |

---

## 🔌 API Endpoints Reference

Interactive OpenAPI documentation with live testing is available at `http://localhost:8000/docs`.

### Summary Table

| HTTP Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/reference` | Upload reference image; validates quality and registers face embedding. |
| `POST` | `/api/verification/start` | Start live verification session and reset baseline calibration. |
| `POST` | `/api/verification/frame` | Stream video frame (Base64 JPEG); returns real-time face & gaze inference. |
| `POST` | `/api/verification/stop` | Stop session, evaluate final verdict rules, and persist report. |
| `GET` | `/api/session/{id}` | Query current active in-memory session status and frame counts. |
| `GET` | `/api/sessions` | Fetch list of completed verification sessions stored in PostgreSQL. |
| `GET` | `/api/session/{id}/report` | Retrieve completed evaluation report for a session. |
| `GET` | `/api/session/{id}/stored` | Retrieve full persisted database record including reference metadata. |
| `GET` | `/api/session/{id}/failures`| Retrieve visual proof thumbnail images captured during violations. |
| `GET` | `/health` | Health check endpoint returning status and timestamp. |

---

### Key API Schemas & Examples

#### 1. Reference Photo Upload (`POST /api/reference`)
**Request:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQE..."
}
```
**Response (Success):**
```json
{
  "success": true,
  "session_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "faces_detected": 1,
  "quality_score": 94.2,
  "checks": [
    {"id": "face_detected", "label": "Face detected", "passed": true, "detail": null},
    {"id": "single_face", "label": "Only one person", "passed": true, "detail": null},
    {"id": "face_size", "label": "Close enough to the camera", "passed": true, "detail": "Face area 24.5%"},
    {"id": "sharpness", "label": "Image is sharp", "passed": true, "detail": "Sharpness 142.3"},
    {"id": "lighting", "label": "Good lighting", "passed": true, "detail": "Brightness 118.0"},
    {"id": "facing_camera", "label": "Facing the camera", "passed": true, "detail": "Angle 4.1 deg"},
    {"id": "eyes_open", "label": "Eyes open", "passed": true, "detail": null}
  ],
  "issues": [],
  "message": "Photo accepted."
}
```

#### 2. Live Frame Analysis (`POST /api/verification/frame`)
**Request:**
```json
{
  "session_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```
**Response:**
```json
{
  "session_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "elapsed_s": 12.4,
  "faces_detected": 1,
  "verification_status": "MATCH",
  "similarity": 0.742,
  "is_match": true,
  "gaze_direction": "CENTER",
  "is_looking_away": false,
  "calibrating": false,
  "calibration_progress": 1.0,
  "gaze_yaw": -2.3,
  "gaze_pitch": 4.1,
  "iris_h": 0.495,
  "iris_v": 0.512,
  "face_box": {"x": 180, "y": 110, "w": 280, "h": 320},
  "live_status": {
    "identity": "MATCH",
    "gaze": "CENTER",
    "looking_away": false,
    "multiple_faces": false,
    "no_face": false
  },
  "events": [],
  "timestamp": 1729918200.123
}
```

---

## 🧪 Running Automated Tests

The backend includes a comprehensive unit test suite covering verdict rules, interval tracking, baseline calibration, vision calculations, and database storage.

### Running Backend Tests (Offline)

From the `backend/` directory with the virtual environment activated:

```bash
cd backend

# Windows PowerShell:
.\.venv\Scripts\python -m pytest -v

# macOS / Linux:
python3 -m pytest -v
```

### Running Specific Test Modules

```bash
# Test verdict evaluation rules (different person, no face, away thresholds)
pytest tests/test_evaluator.py -v

# Test hysteresis interval opening and closing
pytest tests/test_intervals.py -v

# Test gaze math, iris displacement, and baseline calibration
pytest tests/test_gaze.py -v

# Test SQLAlchemy database models and queries
pytest tests/test_database_store.py -v

# Test reference photo quality assurance checks
pytest tests/test_reference_checks.py -v
```

> ⚡ **Fast Execution**: The test suite uses synthetic landmark records and mock frames. All 50+ unit tests run in ~5 seconds offline without requiring physical webcam hardware.

---

## 🔍 AI Models & Pipeline Details

| Component | Model / Engine | Purpose | Output / Metric |
| :--- | :--- | :--- | :--- |
| **Face Detection** | OpenCV YuNet (`face_detection_yunet_2023mar.onnx`) | High-speed frontal face bounding box & 5-point landmark detector. | Bounding box coordinates, detection confidence score. |
| **Face Recognition** | OpenCV SFace (`face_recognition_sface_2021dec.onnx`) | Deep metric learning face recognizer. | 128-D normalized embedding; cosine similarity threshold: `0.363`. |
| **Face Mesh & Iris** | MediaPipe Face Mesh (`refine_landmarks=True`) | 478 3D dense facial landmarks including dedicated iris contours. | Normalized iris horizontal/vertical ratios (`0.0` - `1.0`). |
| **Head Pose** | OpenCV `solvePnP` | 3D head rotation matrix against a canonical 3D facial model. | Euler angles (Yaw, Pitch, Roll in degrees). |
| **Static Face Heuristic** | Custom Behavioral Evaluator | Tracks eye-blink frequency and pose standard deviation over 30s. | Emits `STATIC_FACE_WARNING` if 0 blinks and pose variance is near zero. |

---

## ⚠️ Limitations & Anti-Spoofing Architecture

### Current Scope & Capabilities
This project demonstrates **2D facial recognition and attention tracking** using standard commodity webcams without specialized hardware.
- It robustly identifies faces, matches against enrolled references, and computes precise gaze directions and attention lapses.
- It includes a heuristic safeguard against static printouts by flagging zero blink counts and absence of natural micro-movements.

### Known Limitations
- A high-resolution printed photograph or phone screen displaying a moving video of the enrolled individual can pass 2D facial matching.
- Standard webcams lack hardware depth sensors (RGB-D / Infrared).

### Enterprise Production Recommendations
For high-stakes proctoring or enterprise security deployments, the following additions are recommended:
1. **Passive Neural Anti-Spoofing**: Integrate networks such as MiniFASNet, Silent-Face-Anti-Spoofing, or CDCN to detect print/screen reflection moiré artifacts.
2. **Active Challenge-Response Liveness**: Issue randomized prompts (e.g., *"Blink twice"*, *"Turn head 20° left"*, *"Follow the bouncing circle on screen"*).
3. **Hardware-Assisted Depth**: Deploy structured light or time-of-flight depth sensors (e.g., Apple TrueDepth or Intel RealSense).

---

## 🛠️ Troubleshooting & FAQ

### 1. Camera access denied or black screen
- Ensure your browser has camera access permitted in **Settings > Privacy and security > Site Settings > Camera**.
- Check if another application (such as Zoom, Microsoft Teams, OBS Virtual Camera, or Skype) has locked the webcam exclusively.
- On Chrome, ensure you access the app via `localhost` (e.g. `http://localhost:5173` or `http://localhost:3000`), which is automatically treated as a secure origin (`HTTPS/localhost`).

### 2. MediaPipe installation fails (`pip install mediapipe`)
- MediaPipe wheels require **Python 3.9, 3.10, or 3.11**. If your default system Python is 3.12 or newer, the prebuilt wheel will fail to find matching binaries.
- Check your Python version using `python --version`. If it is 3.12+, install Python 3.10 or 3.11 and create the virtual environment using that specific binary:
  ```bash
  py -3.10 -m venv .venv
  ```

### 3. PowerShell script execution error (`Activate.ps1 cannot be loaded`)
- On Windows, PowerShell may block script execution by default. Run the following command in PowerShell:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  .\.venv\Scripts\Activate.ps1
  ```

### 4. Database connection warning (`Database not ready (attempt 1/10)`)
- This indicates that PostgreSQL is not currently running on `localhost:5432`.
- You can either launch PostgreSQL via Docker (`docker compose up -d db`) or continue running without it. The backend will automatically fall back to in-memory session management after 10 attempts.

### 5. Port already in use (`8000`, `3000`, or `5173`)
- **Port 8000 (Backend)**: If another process is using port 8000, specify a different port:
  ```bash
  uvicorn main:app --reload --port 8080
  ```
  *(Remember to update `VITE_API_URL=http://localhost:8080` in `frontend/`)*
- **Port 5173 (Frontend)**: Vite will automatically suggest the next open port (e.g., 5174).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
