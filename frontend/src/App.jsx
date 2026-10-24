import React, { useState, useEffect } from 'react';
import CameraCapture from './CameraCapture';
import LiveVerification from './LiveVerification';
import SessionResults from './SessionResults';
import { API_BASE } from './config';
import './App.css';

export default function App() {
  // Navigation / Phase state: 'capture' | 'live' | 'results'
  const [appStage, setAppStage] = useState('capture');

  const [confirmedPhoto, setConfirmedPhoto] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [validationInfo, setValidationInfo] = useState(null);
  const [sessionSummary, setSessionSummary] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking'); // 'online' | 'offline' | 'checking'

  // Health check to ensure FastAPI backend on port 8000 is accessible
  useEffect(() => {
    let isMounted = true;
    const checkBackend = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
          if (isMounted) setBackendStatus('online');
        } else {
          if (isMounted) setBackendStatus('offline');
        }
      } catch (err) {
        if (isMounted) setBackendStatus('offline');
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 10000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  // Called when reference photo is validated by backend in Phase 2
  const handlePhotoConfirmed = (imageData, newSessionId, validationData) => {
    setConfirmedPhoto(imageData);
    setSessionId(newSessionId);
    setValidationInfo(validationData);
  };

  // Launch Phase 3 & 4: Live Verification
  const handleStartLiveVerification = () => {
    if (!sessionId) return;
    setAppStage('live');
  };

  // Called when user clicks "Stop Verification" in Phase 3/4
  const handleStopVerification = (summary) => {
    setSessionSummary(summary);
    setAppStage('results');
  };

  // Reset to Phase 1 for a new session
  const handleNewSession = () => {
    setConfirmedPhoto(null);
    setSessionId(null);
    setValidationInfo(null);
    setSessionSummary(null);
    setAppStage('capture');
  };

  // Return to reference capture (M6)
  const handleBackToCapture = () => {
    setConfirmedPhoto(null);
    setSessionId(null);
    setValidationInfo(null);
    setSessionSummary(null);
    setAppStage('capture');
  };

  return (
    <div className="app-layout">
      {/* Top Global Navigation Bar */}
      <header className="navbar">
        <div className="nav-brand">
          <div className="brand-logo">👁️</div>
          <div className="brand-text">
            <h1>Face & Gaze AI</h1>
            <span className="brand-sub">Real-Time Biometric & Attention Verification</span>
          </div>
        </div>

        <div className="nav-actions">
          {/* Current Stage Indicator */}
          <div className="phase-pill">
            <span className="phase-pill-step">STAGE:</span>
            <span className="phase-pill-name">
              {appStage === 'capture' && !sessionId
                ? 'Reference Photo'
                : appStage === 'capture' && sessionId
                ? 'Reference Accepted'
                : appStage === 'live'
                ? 'Live Verification'
                : 'Session Report'}
            </span>
          </div>

          {/* Backend Status Indicator */}
          <div className={`status-pill ${backendStatus}`} title="FastAPI Backend Health">
            <span className="status-indicator-dot"></span>
            <span className="status-label">
              Backend: {backendStatus === 'online' ? 'Online (8000)' : backendStatus === 'offline' ? 'Offline' : 'Checking...'}
            </span>
          </div>

          <a
            href={`${API_BASE}/docs`}
            target="_blank"
            rel="noreferrer"
            className="nav-link-docs"
          >
            API Docs ↗
          </a>
        </div>
      </header>

      {/* Main Workspace Router */}
      <main className="main-content">
        {/* STAGE 1 & 2: Reference Photo Capture & Validation */}
        {appStage === 'capture' && (
          <div className="workspace-grid">
            <div className="camera-section">
              <CameraCapture
                onConfirmPhoto={handlePhotoConfirmed}
                confirmedPhoto={confirmedPhoto}
              />
            </div>

            <aside className="sidebar-card">
              <div className="sidebar-header">
                <h3>Reference Session Status</h3>
              </div>

              {confirmedPhoto && sessionId ? (
                <div className="reference-summary">
                  <div className="confirmed-thumb-wrapper">
                    <img src={confirmedPhoto} alt="Confirmed Face Reference" className="confirmed-thumb" />
                    <span className="confirmed-badge">Reference Photo Accepted</span>
                  </div>

                  <div className="info-list">
                    <div className="info-row">
                      <span className="info-label">Session ID</span>
                      <span className="info-value font-mono text-sky">{sessionId}</span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">Quality Score</span>
                      <span className="info-value text-success">
                        {validationInfo?.quality_score ? `${validationInfo.quality_score}%` : 'Accepted'}
                      </span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">Faces Detected</span>
                      <span className="info-value">1 (Single Face Detected)</span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">Face Embedding</span>
                      <span className="info-value text-emerald">SFace Generated</span>
                    </div>
                  </div>

                  <div className="next-phase-action-box">
                    <h4>Ready for Live Verification</h4>
                    <p>
                      Reference photo accepted and biometric embedding generated. You can now begin live verification.
                    </p>
                    <button
                      className="btn btn-primary btn-large btn-launch"
                      onClick={handleStartLiveVerification}
                      id="btn-start-live-verification"
                    >
                      🚀 Start Live Verification
                    </button>
                  </div>
                </div>
              ) : (
                <div className="reference-empty">
                  <div className="empty-icon">🛡️</div>
                  <h4>No Reference Photo Yet</h4>
                  <p>
                    Click <strong>Open Camera</strong> on the left, align your face inside the oval guide, and click <strong>Capture Photo</strong>.
                  </p>
                  <ul className="guidelines-list">
                    <li>Single face in frame</li>
                    <li>Good ambient lighting</li>
                    <li>Look directly at the lens</li>
                  </ul>
                </div>
              )}
            </aside>
          </div>
        )}

        {/* Real-time Live Face Verification & Eye-Gaze Tracking */}
        {appStage === 'live' && (
          <LiveVerification
            sessionId={sessionId}
            referencePhoto={confirmedPhoto}
            onStopVerification={handleStopVerification}
            onBackToCapture={handleBackToCapture}
          />
        )}

        {/* Final Demo Results & Compliance Audit Dashboard */}
        {appStage === 'results' && (
          <SessionResults
            summary={sessionSummary}
            referencePhoto={confirmedPhoto}
            onNewSession={handleNewSession}
          />
        )}
      </main>

      {/* Global Footer */}
      <footer className="app-footer">
        <p>Face Verification & Eye-Gaze Tracking Demo • OpenCV (YuNet + SFace) & MediaPipe Face Mesh</p>
      </footer>
    </div>
  );
}
