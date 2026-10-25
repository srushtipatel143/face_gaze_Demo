import { useState, useEffect } from 'react';
import { Plus, History, WifiOff } from 'lucide-react';
import CameraCapture from './components/CameraCapture';
import LiveVerification from './components/LiveVerification';
import SessionResults from './components/SessionResults';
import { Button, Card, Stepper, Alert } from './components/ui';
import { API_BASE } from './config';
import './App.css';

const STEPS = ['Reference photo', 'Live check', 'Report'];

export default function App() {
  const [view, setView] = useState('capture');
  const [reference, setReference] = useState(null); // { photo, sessionId, checks } or null
  const [report, setReport] = useState(null);
  const [reportSource, setReportSource] = useState('session'); // 'session' | 'history'
  const [backendOnline, setBackendOnline] = useState(true);

  // 10-second health check
  useEffect(() => {
    let isMounted = true;
    const checkBackend = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (isMounted) setBackendOnline(res.ok);
      } catch {
        if (isMounted) setBackendOnline(false);
      }
    };
    checkBackend();
    const interval = setInterval(checkBackend, 10000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleNewSession = () => {
    setReference(null);
    setReport(null);
    setReportSource('session');
    setView('capture');
  };

  const handleOpenHistory = () => {
    setView('history');
  };

  const handlePhotoConfirmed = (photo, sessionId, checks) => {
    setReference({ photo, sessionId, checks });
  };

  const handleStartLive = () => {
    if (!reference?.sessionId) return;
    setView('live');
  };

  const handleStopVerification = (summary) => {
    setReport(summary);
    setReportSource('session');
    setView('report');
  };

  const handleBackToCapture = () => {
    setView('capture');
  };

  const showStepper = view !== 'history' && !(view === 'report' && reportSource === 'history');
  const currentStep = view === 'capture' ? 0 : view === 'live' ? 1 : 2;

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="header-top">
          <h1>Identity & Attention Check</h1>
          <div className="header-actions">
            <Button
              variant="secondary"
              icon={Plus}
              onClick={handleNewSession}
              disabled={view === 'live'}
            >
              New session
            </Button>
            <Button
              variant="ghost"
              icon={History}
              onClick={handleOpenHistory}
              disabled={view === 'live'}
            >
              History
            </Button>
          </div>
        </div>
        {showStepper && (
          <div className="header-stepper">
            <Stepper steps={STEPS} current={currentStep} />
          </div>
        )}
      </header>

      {!backendOnline && (
        <div className="app-offline-banner">
          <Alert tone="warning" icon={WifiOff}>
            Can't reach the verification service. Check that the backend is running.
          </Alert>
        </div>
      )}

      <main className="app-main">
        {view === 'capture' && (
          <div className="workspace-grid">
            <div className="camera-section">
              <CameraCapture
                onConfirmPhoto={handlePhotoConfirmed}
                confirmedPhoto={reference?.photo}
              />
            </div>

            <aside className="sidebar-card">
              {reference?.sessionId ? (
                <Card title="Reference photo accepted" subtitle="Biometric baseline ready">
                  <div style={{ display: 'grid', gap: '16px' }}>
                    <img
                      src={reference.photo}
                      alt="Reference photo"
                      style={{
                        width: '100%',
                        maxHeight: '200px',
                        objectFit: 'cover',
                        borderRadius: 'var(--radius-sm)',
                        border: '1px solid var(--border)',
                      }}
                    />
                    <p className="muted">
                      Reference photo accepted. You can now start the live check.
                    </p>
                    <Button
                      variant="primary"
                      size="lg"
                      onClick={handleStartLive}
                      id="btn-start-live-check"
                    >
                      Start live check
                    </Button>
                  </div>
                </Card>
              ) : (
                <Card title="No reference photo yet" subtitle="Capture a reference face">
                  <p className="muted" style={{ marginBottom: '12px' }}>
                    Position your face clearly in the camera frame, then capture and confirm.
                  </p>
                  <ul className="muted" style={{ margin: 0, paddingLeft: '20px', display: 'grid', gap: '4px' }}>
                    <li>Single face in frame</li>
                    <li>Good ambient lighting</li>
                    <li>Look directly at the camera</li>
                  </ul>
                </Card>
              )}
            </aside>
          </div>
        )}

        {view === 'live' && (
          <LiveVerification
            sessionId={reference?.sessionId}
            referencePhoto={reference?.photo}
            onStopVerification={handleStopVerification}
            onBackToCapture={handleBackToCapture}
          />
        )}

        {view === 'report' && (
          <SessionResults
            summary={report}
            referencePhoto={reference?.photo}
            onNewSession={handleNewSession}
          />
        )}

        {view === 'history' && (
          <Card title="Session history" subtitle="Past verification audits">
            <p className="muted">Session history will be available here.</p>
          </Card>
        )}
      </main>

      <footer className="app-footer subtle">
        <span>Service: {backendOnline ? 'online' : 'offline'}</span>
        <span> · </span>
        <a href={`${API_BASE}/docs`} target="_blank" rel="noreferrer">
          API docs
        </a>
      </footer>
    </div>
  );
}
