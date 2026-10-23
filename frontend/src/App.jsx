import React, { useState, useEffect } from 'react';
import CameraCapture from './CameraCapture';
import LiveVerification from './LiveVerification';
import SessionResults from './SessionResults';
import './App.css';

const PHASES = [
  { id: 1, title: 'Phase 1: Reference Photo', subtitle: 'Webcam capture, preview & confirm' },
  { id: 2, title: 'Phase 2: Face Validation', subtitle: 'Single-face validation & session' },
  { id: 3, title: 'Phase 3: Live Verification', subtitle: 'Real-time face matching stream' },
  { id: 4, title: 'Phase 4: Gaze Detection', subtitle: 'Iris tracking & looking-away flags' },
  { id: 5, title: 'Phase 5: Session Dashboard', subtitle: 'Timestamped events & final metrics' },
];

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
        const res = await fetch('http://localhost:8000/health');
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

  // Return to reference capture
  const handleBackToCapture = () => {
    setAppStage('capture');
  };

  // Calculate current active phase index for the header roadmap
  const getCurrentPhaseIndex = () => {
    if (appStage === 'results') return 5;
    if (appStage === 'live') return 3; // Phases 3 & 4 active concurrently
    if (confirmedPhoto && sessionId) return 2;
    return 1;
  };

  const activePhase = getCurrentPhaseIndex();

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
          {/* Active Phase Badge */}
          <div className="phase-pill">
            <span className="phase-pill-step">STAGE:</span>
            <span className="phase-pill-name">
              {appStage === 'capture' && !sessionId
                ? 'Phase 1 • Reference Photo'
                : appStage === 'capture' && sessionId
                ? 'Phase 2 • Face Validated'
                : appStage === 'live'
                ? 'Phases 3 & 4 • Live Analysis'
                : 'Phase 5 • Audit Report'}
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
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="nav-link-docs"
          >
            API Docs ↗
          </a>
        </div>
      </header>

      {/* 5-Phase Roadmap Bar */}
      <section className="roadmap-bar">
        <div className="roadmap-header">
          <span className="tracker-badge">Implementation Roadmap</span>
          <span className="tracker-counter">
            {activePhase === 5
              ? 'Phase 5 Complete (5/5)'
              : activePhase === 3
              ? 'Live Verification In Progress (Phases 3 & 4)'
              : activePhase === 2
              ? 'Phase 2 Complete • Ready for Live Stream (2/5)'
              : 'Phase 1 In Progress (0/5)'}
          </span>
        </div>
        <div className="phases-list">
          {PHASES.map((phase) => {
            let status = 'upcoming';
            if (activePhase > phase.id || (activePhase === 5 && phase.id <= 5)) {
              status = 'completed';
            } else if (activePhase === phase.id || (activePhase === 3 && (phase.id === 3 || phase.id === 4))) {
              status = 'current';
            }

            return (
              <div key={phase.id} className={`phase-step ${status}`}>
                <div className="step-circle">
                  {status === 'completed' ? '✓' : phase.id}
                </div>
                <div className="step-content">
                  <div className="step-title">{phase.title}</div>
                  <div className="step-subtitle">{phase.subtitle}</div>
                </div>
              </div>
            );
          })}
        </div>
      </section>

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
                    <span className="confirmed-badge">Validated for Phases 3 & 4</span>
                  </div>

                  <div className="info-list">
                    <div className="info-row">
                      <span className="info-label">Session ID</span>
                      <span className="info-value font-mono text-sky">{sessionId}</span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">Quality Score</span>
                      <span className="info-value text-success">
                        {validationInfo?.quality_score ? `${validationInfo.quality_score}%` : 'High'}
                      </span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">Faces Detected</span>
                      <span className="info-value">1 (Single Person Confirmed)</span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">Biometric Embedding</span>
                      <span className="info-value text-emerald">Generated & Cached</span>
                    </div>
                  </div>

                  <div className="next-phase-action-box">
                    <h4>Phase 2 Succeeded</h4>
                    <p>
                      Your biometric reference embedding is generated and stored in temporary backend session memory.
                    </p>
                    <button
                      className="btn btn-primary btn-large btn-launch"
                      onClick={handleStartLiveVerification}
                      id="btn-start-live-verification"
                    >
                      🚀 Start Live Verification (Phases 3 & 4)
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

        {/* STAGE 3 & 4: Real-time Live Face Verification & Eye-Gaze Tracking */}
        {appStage === 'live' && (
          <LiveVerification
            sessionId={sessionId}
            referencePhoto={confirmedPhoto}
            onStopVerification={handleStopVerification}
            onBackToCapture={handleBackToCapture}
          />
        )}

        {/* STAGE 5: Final Demo Results & Compliance Audit Dashboard */}
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
        <p>Face Verification & Eye-Gaze Detection System • Complete 5-Phase End-to-End Pipeline</p>
      </footer>
    </div>
  );
}
