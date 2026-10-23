# Face & Eye-Gaze Verification System

A real-time AI-powered biometric verification web application combining face matching with eye-gaze tracking and head orientation estimation. Built with **FastAPI**, **MediaPipe**, **OpenCV**, and **React + Vite**.

---

## 📋 Prerequisites

Before running the project, ensure you have:
- **Python** (Recommended: Python 3.9 - 3.11 for MediaPipe compatibility)
- **Node.js** (v18.x or later) & **npm**
- A working **Webcam**

---

## 🚀 Getting Started After Clone

### 1. Clone the Repository
```bash
git clone https://github.com/srushtipatel143/face_gaze_Demo.git
cd face_gaze_Demo
```

---

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
│   ├── database.py          # In-memory thread-safe session database
│   ├── main.py              # FastAPI endpoints & CORS configuration
│   ├── vision_engine.py     # MediaPipe Face Mesh & iris gaze estimation
│   ├── requirements.txt     # Python dependencies
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
        └── SessionResults.jsx # Verification summary & metrics
```
