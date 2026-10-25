import { useCallback, useEffect, useRef, useState } from 'react';
import { Square } from 'lucide-react';
import { useCamera } from '../hooks/useCamera';
import { Button, Badge, Alert, Disclosure, KeyValue } from './ui';
import {
  IDENTITY_STATUS,
  ATTENTION_STATUS,
  INTERVAL_TYPE,
  formatClock,
  formatSeconds,
} from '../lib/format';
import { API_BASE } from '../config';
import '../css/LiveVerification.css';

const TONE_HEX = {
  success: '#10b981',
  danger: '#ef4444',
  warning: '#f59e0b',
  neutral: '#94a3b8',
  info: '#3b82f6',
};

function mergeEvents(prev, incoming) {
  if (!incoming || incoming.length === 0) return prev;
  const result = [...prev];
  for (const item of incoming) {
    const idx = result.findIndex((e) => e.id === item.id);
    if (idx >= 0) {
      result[idx] = item;
    } else {
      result.unshift(item);
    }
  }
  return result.slice(0, 50);
}

export default function LiveVerification({
  sessionId,
  referencePhoto,
  onStopVerification,
  onFinished,
  onBackToCapture,
}) {
  const camera = useCamera();
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const isProcessingRef = useRef(false);
  const intervalRef = useRef(null);
  const consecutiveFailuresRef = useRef(0);
  const framesSentRef = useRef(0);
  const responsesCountRef = useRef(0);

  const [latest, setLatest] = useState(null);
  const [events, setEvents] = useState([]);
  const [connectionWarning, setConnectionWarning] = useState(false);
  const [fatalSessionEnded, setFatalSessionEnded] = useState(false);
  const [sessionStarted, setSessionStarted] = useState(false);
  const [initError, setInitError] = useState(null);
  const [isStopping, setIsStopping] = useState(false);
  const [stopError, setStopError] = useState(false);

  // Draw face box on overlay canvas
  const drawOverlay = useCallback((data) => {
    const overlay = overlayRef.current;
    const video = camera.videoRef.current;
    if (!overlay || !video) return;

    const vidW = video.clientWidth || video.videoWidth || 640;
    const vidH = video.clientHeight || video.videoHeight || 480;
    overlay.width = vidW;
    overlay.height = vidH;

    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, vidW, vidH);

    if (!data?.face_box) return;

    const fb = data.face_box;
    const rawW = video.videoWidth || 640;
    const rawH = video.videoHeight || 480;
    const sentW = Math.min(640, rawW);
    const sentH = Math.round((sentW / rawW) * rawH);

    const scaleX = vidW / sentW;
    const scaleY = vidH / sentH;

    const boxW = fb.width * scaleX;
    const boxH = fb.height * scaleY;
    const boxX = fb.x * scaleX;
    const boxY = fb.y * scaleY;

    // Mirrored display: displayX = width - (x + w)
    const displayX = vidW - (boxX + boxW);

    const identKey = data.live_status?.identity || data.verification_status || 'CHECKING';
    const identStatus = IDENTITY_STATUS[identKey] || IDENTITY_STATUS.CHECKING;
    const strokeColor = TONE_HEX[identStatus.tone] || '#10b981';
    const labelText = identStatus.label;

    // Thin face box (2px)
    ctx.lineWidth = 2;
    ctx.strokeStyle = strokeColor;
    ctx.strokeRect(displayX, boxY, boxW, boxH);

    // Small filled pill above the box with identity label only
    ctx.font = '500 12px Inter, -apple-system, sans-serif';
    const textWidth = ctx.measureText(labelText).width;
    const pillW = textWidth + 12;
    const pillH = 20;
    const pillY = Math.max(0, boxY - pillH - 4);

    ctx.fillStyle = strokeColor;
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(displayX, pillY, pillW, pillH, 4);
    } else {
      ctx.rect(displayX, pillY, pillW, pillH);
    }
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.fillText(labelText, displayX + 6, pillY + 14);
  }, [camera.videoRef]);

  // Capture frame and send to /api/verification/frame
  const sendFrame = useCallback(async () => {
    if (isProcessingRef.current || !sessionId || !sessionStarted) return;

    const video = camera.videoRef.current;
    if (!video || video.readyState < 2) return;

    const rawW = video.videoWidth || 640;
    const rawH = video.videoHeight || 480;
    const targetW = Math.min(640, rawW);
    const targetH = Math.round((targetW / rawW) * rawH);

    if (!canvasRef.current) {
      canvasRef.current = document.createElement('canvas');
    }
    const canvas = canvasRef.current;
    canvas.width = targetW;
    canvas.height = targetH;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Unmirrored frame sent to backend
    ctx.drawImage(video, 0, 0, targetW, targetH);
    const image = canvas.toDataURL('image/jpeg', 0.7);

    isProcessingRef.current = true;
    framesSentRef.current += 1;

    try {
      const res = await fetch(`${API_BASE}/api/verification/frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          image,
        }),
      });

      if (!res.ok) {
        if (res.status === 400 || res.status === 404 || res.status === 409) {
          // Session ended
          if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
          }
          setFatalSessionEnded(true);
          return;
        }
        throw new Error(`Frame HTTP ${res.status}`);
      }

      const data = await res.json();
      responsesCountRef.current += 1;
      consecutiveFailuresRef.current = 0;
      setConnectionWarning(false);
      setLatest(data);

      if (data.events && data.events.length > 0) {
        setEvents((prev) => mergeEvents(prev, data.events));
      }

      drawOverlay(data);
    } catch (err) {
      console.warn('Frame send failed:', err.message);
      consecutiveFailuresRef.current += 1;
      if (consecutiveFailuresRef.current >= 3) {
        setConnectionWarning(true);
      }
    } finally {
      isProcessingRef.current = false;
    }
  }, [sessionId, sessionStarted, camera.videoRef, drawOverlay]);

  // Mount effect: Start camera immediately and start session
  useEffect(() => {
    let mounted = true;

    // Start camera immediately so video preview starts right away
    camera.start().catch((camErr) => {
      console.warn('Initial camera start error:', camErr);
    });

    async function initSession() {
      if (!sessionId) {
        console.warn('LiveVerification mounted without sessionId yet');
        return;
      }

      try {
        const startRes = await fetch(`${API_BASE}/api/verification/start`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId }),
        });

        if (!startRes.ok) {
          throw new Error(`Start request failed (${startRes.status})`);
        }
        if (mounted) {
          setSessionStarted(true);
        }
      } catch (err) {
        console.error('Init live check error:', err);
        if (mounted) {
          setInitError('Could not start the live check. Check backend connection.');
        }
      }
    }

    initSession();

    return () => {
      mounted = false;
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      camera.stop();
    };
  }, [sessionId]); // eslint-disable-line react-hooks/exhaustive-deps

  // When camera turns on and session is verified started, start sending frames every 200 ms
  useEffect(() => {
    if (camera.status !== 'on' || !sessionStarted || fatalSessionEnded) return;

    intervalRef.current = setInterval(sendFrame, 200);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [camera.status, sessionStarted, fatalSessionEnded, sendFrame]);

  // Stop session
  const handleStop = async () => {
    if (isStopping) return;
    setIsStopping(true);
    setStopError(false);

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    camera.stop();

    try {
      const res = await fetch(`${API_BASE}/api/verification/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });

      if (!res.ok) {
        throw new Error(`Stop failed with HTTP ${res.status}`);
      }

      const report = await res.json();
      if (onFinished) onFinished(report);
      else if (onStopVerification) onStopVerification(report);
    } catch (err) {
      console.error('Stop verification error:', err);
      setStopError(true);
      setIsStopping(false);
    }
  };

  const identKey = latest?.live_status?.identity || 'CHECKING';
  const identStatus = IDENTITY_STATUS[identKey] || IDENTITY_STATUS.CHECKING;

  const attnKey = latest?.live_status?.attention || 'UNKNOWN';
  const attnStatus = ATTENTION_STATUS[attnKey] || ATTENTION_STATUS.UNKNOWN;

  const isCalibrating = latest?.live_status?.attention === 'CALIBRATING';
  const calibProgress = Math.round((latest?.calibration_progress || 0) * 100);

  const techItems = [
    { label: 'Session ID', value: sessionId || '—' },
    { label: 'Similarity', value: latest?.similarity != null ? latest.similarity.toFixed(3) : '—' },
    {
      label: 'Head yaw / pitch',
      value:
        latest?.gaze_yaw != null && latest?.gaze_pitch != null
          ? `${latest.gaze_yaw.toFixed(1)}° / ${latest.gaze_pitch.toFixed(1)}°`
          : '—',
    },
    {
      label: 'Iris h / v',
      value:
        latest?.iris_h != null && latest?.iris_v != null
          ? `${latest.iris_h.toFixed(2)} / ${latest.iris_v.toFixed(2)}`
          : '—',
    },
    { label: 'Frames sent', value: String(framesSentRef.current) },
    {
      label: 'Responses per second',
      value:
        latest?.elapsed_s > 0
          ? (responsesCountRef.current / latest.elapsed_s).toFixed(1)
          : '—',
    },
    { label: 'Calibration progress', value: `${calibProgress}%` },
  ];

  return (
    <div className="live-screen">
      {/* Top Header */}
      <header className="live-header">
        <div className="live-header-status">
          <span className="live-dot" aria-hidden="true" />
          <span className="live-title">Session in progress</span>
          <span className="live-timer num">{formatClock(latest?.elapsed_s ?? 0)}</span>
        </div>

        <Button
          variant="danger"
          icon={Square}
          onClick={handleStop}
          disabled={isStopping}
          id="btn-stop-verification"
        >
          {isStopping ? 'Ending…' : 'End session'}
        </Button>
      </header>

      {/* Alerts */}
      {fatalSessionEnded && (
        <Alert
          tone="danger"
          action={
            <Button variant="primary" onClick={onBackToCapture}>
              New session
            </Button>
          }
        >
          This session has ended. Start a new session.
        </Alert>
      )}

      {initError && (
        <Alert tone="danger">{initError}</Alert>
      )}

      {camera.error && (
        <Alert
          tone="danger"
          action={
            <Button variant="secondary" size="sm" onClick={camera.start}>
              Try again
            </Button>
          }
        >
          {camera.error}
        </Alert>
      )}

      {connectionWarning && !fatalSessionEnded && (
        <Alert tone="warning">Connection lost. Retrying…</Alert>
      )}

      {stopError && (
        <Alert
          tone="danger"
          action={
            <Button variant="secondary" onClick={handleStop}>
              Try again
            </Button>
          }
        >
          Could not finish the session. Try again.
        </Alert>
      )}

      {/* Grid: 3fr / 1fr, gap 24px */}
      <div className="live-grid">
        {/* Left Column: Video + Calibration Overlay + Status Bar */}
        <div className="live-main">
          {/* 16:9 Mirrored Video Area */}
          <div className="live-viewport">
            <video
              ref={camera.videoRef}
              autoPlay
              playsInline
              muted
              className="live-video"
            />
            <canvas ref={overlayRef} className="overlay-canvas" />

            {/* Calibration Overlay */}
            {isCalibrating && (
              <div className="calibration-overlay">
                <div className="calibration-dot" aria-hidden="true" />
                <p className="calibration-text">Look at the screen normally while we calibrate</p>
                <div className="calibration-progress-track">
                  <div
                    className="calibration-progress-fill"
                    style={{ width: `${calibProgress}%` }}
                  />
                </div>
              </div>
            )}
          </div>

          {/* One-Line Status Bar */}
          <div className="live-status-bar">
            <div className="status-item">
              <span className="status-label">Identity</span>
              <Badge tone={identStatus.tone}>{identStatus.label}</Badge>
            </div>
            <div className="status-item">
              <span className="status-label">Attention</span>
              <Badge tone={attnStatus.tone}>{attnStatus.label}</Badge>
            </div>
          </div>
        </div>

        {/* Right Column: Activity + Reference */}
        <div className="live-sidebar">
          {/* Activity Card */}
          <div className="activity-card">
            <h3>Activity</h3>
            <div className="activity-list">
              {events.length === 0 ? (
                <div className="activity-empty muted">No issues so far</div>
              ) : (
                events.map((evt) => {
                  const intervalType = INTERVAL_TYPE[evt.type] || { tone: 'neutral' };
                  const dotColor = TONE_HEX[intervalType.tone] || '#94a3b8';
                  const isOngoing = evt.end_s === null || evt.end_s === undefined;

                  return (
                    <div key={evt.id} className="activity-row">
                      <span className="activity-time num">{formatClock(evt.start_s)}</span>
                      <div className="activity-main">
                        <span
                          className="activity-dot"
                          style={{ backgroundColor: dotColor }}
                          aria-hidden="true"
                        />
                        <span className="activity-message">{evt.message}</span>
                      </div>
                      <span className={`activity-duration ${isOngoing ? 'ongoing' : 'num'}`}>
                        {isOngoing ? 'ongoing' : formatSeconds(evt.duration_s)}
                      </span>
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {/* Reference Photo Card */}
          <div className="reference-card">
            <h3>Reference</h3>
            <div className="reference-content">
              {referencePhoto && (
                <img src={referencePhoto} alt="Reference photo" className="reference-thumb" />
              )}
              <span className="reference-text">Photo accepted</span>
            </div>
          </div>
        </div>
      </div>

      {/* Technical Details Section */}
      <div className="tech-details-section">
        <Disclosure title="Technical details">
          <KeyValue items={techItems} />
        </Disclosure>
      </div>
    </div>
  );
}
