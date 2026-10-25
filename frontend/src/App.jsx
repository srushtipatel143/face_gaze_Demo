import { useState, useEffect } from 'react';
import { Plus, History, WifiOff } from 'lucide-react';
import CameraCapture from './components/CameraCapture';
import LiveVerification from './components/LiveVerification';
import SessionReport from './components/SessionReport';
import SessionHistory from './components/SessionHistory';
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

  const [activeSessionId, setActiveSessionId] = useState(null);

  const handlePhotoConfirmed = (photo, sessionId, checks) => {
    if (sessionId) setActiveSessionId(sessionId);
    setReference({ photo, sessionId, checks });
  };

  const handleStartLive = (directSessionId) => {
    const sid = (typeof directSessionId === 'string' && directSessionId) ? directSessionId : reference?.sessionId;
    if (sid) {
      setActiveSessionId(sid);
      setReference((prev) => ({ ...(prev || {}), sessionId: sid }));
    }
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
          <CameraCapture
            onAccepted={({ photo, sessionId, checks }) => {
              handlePhotoConfirmed(photo, sessionId, checks);
            }}
            onStartLive={handleStartLive}
            initialReference={reference}
          />
        )}

        {view === 'live' && (
          <LiveVerification
            sessionId={activeSessionId || reference?.sessionId}
            referencePhoto={reference?.photo}
            onStopVerification={handleStopVerification}
            onBackToCapture={handleBackToCapture}
          />
        )}

        {view === 'report' && (
          <SessionReport
            report={report}
            reportSource={reportSource}
            onNewSession={handleNewSession}
            onBackToHistory={() => setView('history')}
          />
        )}

        {view === 'history' && (
          <SessionHistory
            onOpenReport={(rep) => {
              setReport(rep);
              setReportSource('history');
              setView('report');
            }}
            onNewSession={handleNewSession}
          />
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
