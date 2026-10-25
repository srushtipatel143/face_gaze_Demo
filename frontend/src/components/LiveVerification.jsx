import { useState, useRef, useEffect, useCallback } from 'react';
import { Disclosure } from './ui';
import { API_BASE } from '../config';
import '../css/LiveVerification.css';

export default function LiveVerification({ sessionId, referencePhoto, onStopVerification, onBackToCapture }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const isProcessingRef = useRef(false);
  const intervalRef = useRef(null);

  const [error, setError] = useState(null);
  const [latestFrame, setLatestFrame] = useState(null);
  const [events, setEvents] = useState([]);
  const [stats, setStats] = useState({
    total: 0,
    matched: 0,
    unmatched: 0,
    noFace: 0,
    lookingAway: 0,
  });
  const [isStopping, setIsStopping] = useState(false);
  const [connectionWarning, setConnectionWarning] = useState(null);
  const [fatalSessionError, setFatalSessionError] = useState(null);
  const consecutiveFailuresRef = useRef(0);

  // Stop camera tracks cleanly
  const stopCamera = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const addLocalEvent = useCallback((title, detail, level) => {
    setEvents((prev) => {
      const timeStr = new Date().toLocaleTimeString();
      const last = prev[0];
      if (last && last.title === title && Date.now() - last.ts < 2500) {
        return prev; // Debounce repetitive frame events in UI
      }
      const newEvt = { id: Math.random().toString(), time: timeStr, title, detail, level, ts: Date.now() };
      return [newEvt, ...prev.slice(0, 40)];
    });
  }, []);

  // Draw overlay bounding box on top of the live stream
  const drawOverlay = useCallback((frameData, vidW, vidH, sentW, sentH) => {
    const overlay = overlayCanvasRef.current;
    if (!overlay) return;

    overlay.width = vidW;
    overlay.height = vidH;
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, vidW, vidH);

    if (frameData.face_box) {
      const fb = frameData.face_box;
      const scaleX = sentW ? vidW / sentW : 1.0;
      const scaleY = sentH ? vidH / sentH : 1.0;

      const x = fb.x * scaleX;
      const y = fb.y * scaleY;
      const width = fb.width * scaleX;
      const height = fb.height * scaleY;

      // Because the <video> element is mirrored in the UI via CSS (transform: scaleX(-1))
      // for a natural mirror-view user experience, map raw camera box X to screen display X (L4):
      const displayX = vidW - (x + width);

      let strokeColor = '#10B981'; // Green for match & center
      let statusText = 'Match';

      if (frameData.verification_status === 'NO_MATCH') {
        strokeColor = '#EF4444';
        statusText = 'No match';
      } else if (frameData.verification_status === 'UNKNOWN') {
        strokeColor = '#F59E0B';
        statusText = 'Face turned away';
      } else if (frameData.verification_status === 'MULTIPLE_FACES') {
        strokeColor = '#DC2626';
        statusText = 'Multiple faces';
      } else if (frameData.is_looking_away === true) {
        strokeColor = '#F59E0B';
        statusText = 'Looking away';
      }

      // Rounded Bounding Box
      ctx.lineWidth = 3;
      ctx.strokeStyle = strokeColor;
      ctx.strokeRect(displayX, y, width, height);

      // Corner Accents
      const cLen = Math.min(24, width * 0.2);
      ctx.lineWidth = 5;
      ctx.beginPath();
      // Top-Left
      ctx.moveTo(displayX, y + cLen);
      ctx.lineTo(displayX, y);
      ctx.lineTo(displayX + cLen, y);
      // Top-Right
      ctx.moveTo(displayX + width - cLen, y);
      ctx.lineTo(displayX + width, y);
      ctx.lineTo(displayX + width, y + cLen);
      // Bottom-Left
      ctx.moveTo(displayX, y + height - cLen);
      ctx.lineTo(displayX, y + height);
      ctx.lineTo(displayX + cLen, y + height);
      // Bottom-Right
      ctx.moveTo(displayX + width - cLen, y + height);
      ctx.lineTo(displayX + width, y + height);
      ctx.lineTo(displayX + width, y + height - cLen);
      ctx.stroke();

      // Label background & text (drawn in unmirrored canvas space so text remains readable)
      const simText = frameData.similarity !== null ? ` ${(frameData.similarity * 100).toFixed(0)}%` : '';
      const label = `${statusText}${simText}`;
      ctx.font = 'bold 13px Inter, sans-serif';
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = strokeColor;
      ctx.fillRect(displayX, Math.max(0, y - 26), textWidth + 16, 26);

      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(label, displayX + 8, Math.max(16, y - 8));
    }
  }, []);

  // Send a live video frame to backend
  const captureAndSendFrame = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || isProcessingRef.current || video.readyState < 2) return;

    const rawWidth = video.videoWidth || 640;
    const rawHeight = video.videoHeight || 480;

    // Scale capture canvas to target ~640px wide for reduced payload & latency (L3)
    const targetWidth = Math.min(640, rawWidth);
    const targetHeight = Math.round((targetWidth / rawWidth) * rawHeight);

    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Draw raw unmirrored video frame to canvas for backend face & gaze analysis
    ctx.drawImage(video, 0, 0, targetWidth, targetHeight);

    // Use JPEG quality 0.70 for live stream frames (L3)
    const base64Image = canvas.toDataURL('image/jpeg', 0.70);

    isProcessingRef.current = true;
    try {
      const response = await fetch(`${API_BASE}/api/verification/frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          image: base64Image,
        }),
      });

      if (!response.ok) {
        if (response.status === 400 || response.status === 404) {
          // Fatal session error: session completed, inactive, or missing from memory (M5)
          const errData = await response.json().catch(() => ({}));
          const errMsg = errData.detail || `Session is no longer active (HTTP ${response.status}). Please start a new session.`;
          setFatalSessionError(errMsg);
          setLatestFrame(null);
          if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
          }
          const overlay = overlayCanvasRef.current;
          if (overlay) {
            const overlayCtx = overlay.getContext('2d');
            if (overlayCtx) overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
          }
          return;
        }
        throw new Error(`Frame analysis failed (${response.status})`);
      }

      const data = await response.json();

      // Reset failure streak on success (M5)
      consecutiveFailuresRef.current = 0;
      setConnectionWarning(null);

      setLatestFrame(data);

      // Update local rolling stats
      setStats((prev) => {
        const next = { ...prev, total: prev.total + 1 };
        if (data.verification_status === 'MATCH') next.matched += 1;
        else if (data.verification_status === 'NO_MATCH') next.unmatched += 1;
        else if (data.verification_status === 'NO_FACE') next.noFace += 1;

        if (data.is_looking_away === true) next.lookingAway += 1;
        return next;
      });

      // Add events to the activity feed
      if (data.verification_status === 'NO_FACE') {
        addLocalEvent('Face not visible', 'No face detected in camera view', 'warning');
      } else if (data.verification_status === 'MULTIPLE_FACES') {
        addLocalEvent('Another person', `${data.faces_detected} faces detected in frame`, 'danger');
      } else if (data.verification_status === 'NO_MATCH') {
        addLocalEvent('Different person', `Similarity: ${(data.similarity * 100).toFixed(1)}%`, 'danger');
      } else if (data.verification_status === 'UNKNOWN') {
        addLocalEvent('Face turned away', 'Face turned far from camera (>35°)', 'warning');
      } else if (data.calibrating || data.gaze_direction === 'CALIBRATING') {
        // Calibrating baseline gaze
      } else if (data.is_looking_away === true) {
        addLocalEvent('Looked away', `Direction: ${data.gaze_direction ? data.gaze_direction.replace('_', ' ').toLowerCase() : 'away'}`, 'warning');
      }

      // Draw bounding box & landmark indicators on overlay canvas (scale box coordinates if downscaled)
      drawOverlay(data, rawWidth, rawHeight, targetWidth, targetHeight);
    } catch (err) {
      console.warn('Frame processing glitch:', err.message);
      consecutiveFailuresRef.current += 1;
      // After 3 consecutive failures, clear latestFrame, show warning banner, and grey out overlay (M5)
      if (consecutiveFailuresRef.current >= 3) {
        setLatestFrame(null);
        setConnectionWarning('Connection lost. Retrying…');
        const overlay = overlayCanvasRef.current;
        if (overlay) {
          const overlayCtx = overlay.getContext('2d');
          if (overlayCtx) {
            overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
            overlayCtx.fillStyle = 'rgba(15, 23, 42, 0.65)';
            overlayCtx.fillRect(0, 0, overlay.width, overlay.height);
          }
        }
      }
    } finally {
      isProcessingRef.current = false;
    }
  }, [sessionId, addLocalEvent, drawOverlay]);

  // Start Live Verification on component mount
  useEffect(() => {
    let active = true;

    async function initLive() {
      setError(null);
      try {
        // 1. Notify backend verification has started
        const startRes = await fetch(`${API_BASE}/api/verification/start`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId }),
        });

        if (!startRes.ok) {
          throw new Error('Backend failed to start verification session');
        }

        // 2. Start webcam
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280, min: 640 },
            height: { ideal: 720, min: 480 },
            facingMode: 'user',
          },
          audio: false,
        });

        if (!active) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play().then(() => {
              // Send live frames at ~5 FPS (every 200ms)
              intervalRef.current = setInterval(captureAndSendFrame, 200);
            });
          };
        }
      } catch (err) {
        console.error('Live verification start error:', err);
        setError(err.message || 'Failed to start live camera verification.');
      }
    }

    initLive();

    return () => {
      active = false;
      stopCamera();
    };
  }, [sessionId, captureAndSendFrame, stopCamera]);

  // Handle Stop Verification button click
  const handleStop = async () => {
    setIsStopping(true);
    stopCamera();

    try {
      const res = await fetch(`${API_BASE}/api/verification/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });

      if (!res.ok) {
        throw new Error('Failed to stop verification session on backend');
      }

      const summary = await res.json();
      onStopVerification(summary);
    } catch (err) {
      console.error('Stop error:', err);
      setError(`Could not finish the session: ${err.message}. No result was produced.`);
      setIsStopping(false);
    }
  };

  const matchRate = stats.total > 0 ? Math.round((stats.matched / stats.total) * 100) : null;
  const isCalibratingGaze = latestFrame?.calibrating || latestFrame?.gaze_direction === 'CALIBRATING';
  const gazeAttentionRate = (!isCalibratingGaze && stats.total > 0)
    ? Math.round(((stats.total - stats.lookingAway) / stats.total) * 100)
    : null;

  return (
    <div className="live-verification-container">
      {/* Hidden off-screen canvas used for capturing jpeg frame */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {/* Top Header */}
      <div className="live-header">
        <div className="header-left">
          <div className="pulse-indicator">
            <span className="live-dot"></span>
            <span className="live-label">Session in progress</span>
          </div>
        </div>

        <div className="header-right">
          <button
            type="button"
            className="btn btn-danger-stop"
            onClick={handleStop}
            id="btn-stop-verification"
            disabled={isStopping}
          >
            {isStopping ? 'Ending…' : 'End session'}
          </button>
        </div>
      </div>

      {error && (
        <div className="live-error-banner">
          <span><strong>Error:</strong> {error}</span>
          <div style={{ display: 'inline-flex', gap: '8px', marginLeft: '12px' }}>
            <button className="btn-small-link" onClick={handleStop} disabled={isStopping}>
              {isStopping ? 'Retrying…' : 'Retry'}
            </button>
            <button className="btn-small-link" onClick={onBackToCapture}>
              Back to reference photo
            </button>
          </div>
        </div>
      )}

      {/* Main Workspace: Live Viewport + Sidebar */}
      <div className="live-grid">
        {/* Live Camera Viewport */}
        <div className="viewport-card">
          <div className="video-viewport">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="live-video"
            />
            {/* Overlay Canvas for Bounding Box & Landmarks */}
            <canvas ref={overlayCanvasRef} className="overlay-canvas" />

            {/* Connection Warning Banner */}
            {connectionWarning && (
              <div className="connection-lost-banner" role="alert">
                <span>{connectionWarning}</span>
              </div>
            )}

            {/* Calibration Banner */}
            {latestFrame && (latestFrame.calibrating || latestFrame.gaze_direction === 'CALIBRATING') && (
              <div className="calibration-banner">
                <span className="pulse-dot"></span>
                <span>Look at the screen normally while we calibrate</span>
              </div>
            )}

            {/* Fatal Session Error Modal */}
            {fatalSessionError && (
              <div className="fatal-error-overlay">
                <div className="fatal-error-card">
                  <h3>Session ended</h3>
                  <p>{fatalSessionError}</p>
                  <button className="btn-primary" onClick={onBackToCapture}>
                    Back to reference photo
                  </button>
                </div>
              </div>
            )}

            {/* Live Status Overlay Ribbon */}
            <div className="hud-overlay-top">
              {latestFrame ? (
                <div className={`status-tag ${latestFrame.verification_status.toLowerCase()}`}>
                  <span className="tag-text">
                    {latestFrame.verification_status === 'MATCH'
                      ? 'Match'
                      : latestFrame.verification_status === 'NO_MATCH'
                      ? 'Different face'
                      : latestFrame.verification_status === 'UNKNOWN'
                      ? 'Face turned away'
                      : latestFrame.verification_status === 'NO_FACE'
                      ? 'No face'
                      : 'Another person'}
                  </span>
                </div>
              ) : (
                <div className="status-tag initializing">
                  <span className="spinner-dot"></span> Starting…
                </div>
              )}

              {latestFrame && (
                <div
                  className={`gaze-tag ${
                    latestFrame.is_looking_away === true
                      ? 'away'
                      : latestFrame.is_looking_away === false
                      ? 'center'
                      : 'unknown'
                  }`}
                >
                  <span>
                    {latestFrame.calibrating || latestFrame.gaze_direction === 'CALIBRATING'
                      ? 'Calibrating'
                      : latestFrame.gaze_direction === 'CENTER'
                      ? 'On screen'
                      : latestFrame.gaze_direction === 'LOOKING_LEFT'
                      ? 'Looking left'
                      : latestFrame.gaze_direction === 'LOOKING_RIGHT'
                      ? 'Looking right'
                      : latestFrame.gaze_direction === 'LOOKING_UP'
                      ? 'Looking up'
                      : latestFrame.gaze_direction === 'LOOKING_DOWN'
                      ? 'Looking down'
                      : latestFrame.is_looking_away
                      ? 'Looking away'
                      : 'On screen'}
                  </span>
                </div>
              )}
            </div>

            {/* Bottom HUD Bar */}
            <div className="hud-overlay-bottom">
              <div className="hud-metric">
                <span className="metric-title">Similarity</span>
                <span className="metric-number">
                  {latestFrame?.similarity !== null && latestFrame?.similarity !== undefined
                    ? `${(latestFrame.similarity * 100).toFixed(1)}%`
                    : '—'}
                </span>
              </div>

              <div className="hud-metric">
                <span className="metric-title">Head pose</span>
                <span className="metric-number">
                  {latestFrame?.gaze_yaw !== null && latestFrame?.gaze_yaw !== undefined
                    ? `Y:${latestFrame.gaze_yaw > 0 ? '+' : ''}${latestFrame.gaze_yaw}° P:${latestFrame.gaze_pitch > 0 ? '+' : ''}${latestFrame.gaze_pitch}°`
                    : '—'}
                </span>
              </div>

              <div className="hud-metric">
                <span className="metric-title">Frames</span>
                <span className="metric-number">{stats.total}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div className="live-sidebar">
          {/* Reference Photo Comparison Card */}
          <div className="live-card">
            <h4>Reference photo</h4>
            <div className="reference-compare-row">
              <img src={referencePhoto} alt="Reference face" className="ref-compare-thumb" />
              <div className="ref-compare-details">
                <div className="detail-item">
                  <span className="detail-label">Status</span>
                  <span className="detail-value text-success">Active</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Face check</span>
                  <span className="detail-value">Face matching</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Attention</span>
                  <span className="detail-value">Iris tracking</span>
                </div>
              </div>
            </div>
          </div>

          {/* Running Gauges */}
          <div className="live-card">
            <h4>Live status</h4>
            <div className="gauge-group">
              <div className="gauge-item">
                <div className="gauge-header">
                  <span>Identity match</span>
                  <span className="gauge-val">{matchRate !== null ? `${matchRate}%` : '—'}</span>
                </div>
                <div className="gauge-track">
                  <div
                    className={`gauge-fill ${matchRate !== null && matchRate >= 80 ? 'good' : 'warning'}`}
                    style={{ width: `${matchRate ?? 0}%` }}
                  ></div>
                </div>
              </div>

              <div className="gauge-item">
                <div className="gauge-header">
                  <span>Time on screen</span>
                  <span className="gauge-val">
                    {isCalibratingGaze ? 'Calibrating…' : gazeAttentionRate !== null ? `${gazeAttentionRate}%` : '—'}
                  </span>
                </div>
                <div className="gauge-track">
                  <div
                    className={`gauge-fill ${gazeAttentionRate !== null && gazeAttentionRate >= 70 ? 'good' : 'warning'}`}
                    style={{ width: `${gazeAttentionRate ?? 0}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>

          {/* Activity Log */}
          <div className="live-card events-card">
            <div className="events-card-header">
              <h4>Activity</h4>
              <span className="event-count-badge">{events.length} {events.length === 1 ? 'event' : 'events'}</span>
            </div>
            <div className="events-stream-list">
              {events.length === 0 ? (
                <div className="events-empty">
                  <span>No issues so far</span>
                </div>
              ) : (
                events.map((evt) => (
                  <div key={evt.id} className={`event-row level-${evt.level}`}>
                    <span className="event-time">{evt.time}</span>
                    <div className="event-info">
                      <strong className="event-title">{evt.title}</strong>
                      <span className="event-detail">{evt.detail}</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Technical Details */}
          <div className="live-card">
            <Disclosure title="Technical details">
              <div style={{ fontSize: '0.82rem', display: 'grid', gap: '6px', marginTop: '8px' }}>
                <div><span style={{ color: '#94a3b8' }}>Session ID:</span> <span style={{ fontFamily: 'monospace' }}>{sessionId}</span></div>
              </div>
            </Disclosure>
          </div>
        </div>
      </div>
    </div>
  );
}
