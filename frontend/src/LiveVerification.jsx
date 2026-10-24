import React, { useState, useRef, useEffect, useCallback } from 'react';
import './LiveVerification.css';

const API_BASE = 'http://localhost:8000';

export default function LiveVerification({ sessionId, referencePhoto, onStopVerification, onBackToCapture }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const isProcessingRef = useRef(false);
  const intervalRef = useRef(null);

  const [isCameraActive, setIsCameraActive] = useState(false);
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
    setIsCameraActive(false);
  }, []);

  // Send a live video frame to backend
  const captureAndSendFrame = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || isProcessingRef.current || video.readyState < 2) return;

    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;

    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Draw video frame to off-screen canvas (mirrored to match user perspective)
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, width, height);
    ctx.setTransform(1, 0, 0, 1, 0, 0); // reset transform

    const base64Image = canvas.toDataURL('image/jpeg', 0.85);

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
        throw new Error(`Frame analysis failed (${response.status})`);
      }

      const data = await response.json();
      setLatestFrame(data);

      // Update local rolling stats
      setStats((prev) => {
        const next = { ...prev, total: prev.total + 1 };
        if (data.verification_status === 'MATCH') next.matched += 1;
        else if (data.verification_status === 'NO_MATCH') next.unmatched += 1;
        else if (data.verification_status === 'NO_FACE') next.noFace += 1;

        if (data.is_looking_away) next.lookingAway += 1;
        return next;
      });

      // Add audit events to the real-time feed
      if (data.verification_status === 'NO_FACE') {
        addLocalEvent('⚠️ Face Lost', 'No face detected in camera viewport', 'warning');
      } else if (data.verification_status === 'MULTIPLE_FACES') {
        addLocalEvent('🚨 Multiple Faces', `${data.faces_detected} faces detected in frame`, 'danger');
      } else if (data.verification_status === 'NO_MATCH') {
        addLocalEvent('❌ Identity Mismatch', `Similarity: ${(data.similarity * 100).toFixed(1)}%`, 'danger');
      } else if (data.verification_status === 'UNKNOWN') {
        addLocalEvent('⚠️ Angle Too Steep', 'Face turned far from camera (>35° yaw/pitch)', 'warning');
      } else if (data.is_looking_away) {
        addLocalEvent('👀 Looking Away', `Direction: ${data.gaze_direction.replace('_', ' ')}`, 'warning');
      }

      // Draw bounding box & landmark indicators on overlay canvas
      drawOverlay(data, width, height);
    } catch (err) {
      console.warn('Frame processing glitch:', err.message);
    } finally {
      isProcessingRef.current = false;
    }
  }, [sessionId]);

  const addLocalEvent = (title, detail, level) => {
    setEvents((prev) => {
      const timeStr = new Date().toLocaleTimeString();
      const last = prev[0];
      if (last && last.title === title && Date.now() - last.ts < 2500) {
        return prev; // Debounce repetitive frame events in UI
      }
      const newEvt = { id: Math.random().toString(), time: timeStr, title, detail, level, ts: Date.now() };
      return [newEvt, ...prev.slice(0, 40)];
    });
  };

  // Draw overlay bounding box on top of the live stream
  const drawOverlay = (frameData, vidW, vidH) => {
    const overlay = overlayCanvasRef.current;
    if (!overlay) return;

    overlay.width = vidW;
    overlay.height = vidH;
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, vidW, vidH);

    if (frameData.face_box) {
      const { x, y, width, height } = frameData.face_box;

      // Because video is mirrored horizontally:
      const mirrorX = vidW - (x + width);

      let strokeColor = '#10B981'; // Green for match & center
      let statusText = 'VERIFIED MATCH';

      if (frameData.verification_status === 'NO_MATCH') {
        strokeColor = '#EF4444';
        statusText = 'NO MATCH';
      } else if (frameData.verification_status === 'UNKNOWN') {
        strokeColor = '#F59E0B';
        statusText = 'HEAD TURNED • UNCERTAIN';
      } else if (frameData.verification_status === 'MULTIPLE_FACES') {
        strokeColor = '#DC2626';
        statusText = 'MULTIPLE FACES';
      } else if (frameData.is_looking_away) {
        strokeColor = '#F59E0B';
        statusText = `MATCH • ${frameData.gaze_direction.replace('_', ' ')}`;
      }

      // Rounded Bounding Box
      ctx.lineWidth = 3;
      ctx.strokeStyle = strokeColor;
      ctx.strokeRect(mirrorX, y, width, height);

      // Corner Accents
      const cLen = Math.min(24, width * 0.2);
      ctx.lineWidth = 5;
      ctx.beginPath();
      // Top-Left
      ctx.moveTo(mirrorX, y + cLen);
      ctx.lineTo(mirrorX, y);
      ctx.lineTo(mirrorX + cLen, y);
      // Top-Right
      ctx.moveTo(mirrorX + width - cLen, y);
      ctx.lineTo(mirrorX + width, y);
      ctx.lineTo(mirrorX + width, y + cLen);
      // Bottom-Left
      ctx.moveTo(mirrorX, y + height - cLen);
      ctx.lineTo(mirrorX, y + height);
      ctx.lineTo(mirrorX + cLen, y + height);
      // Bottom-Right
      ctx.moveTo(mirrorX + width - cLen, y + height);
      ctx.lineTo(mirrorX + width, y + height);
      ctx.lineTo(mirrorX + width, y + height - cLen);
      ctx.stroke();

      // Label background & text
      const simText = frameData.similarity !== null ? ` ${(frameData.similarity * 100).toFixed(0)}%` : '';
      const label = `${statusText}${simText}`;
      ctx.font = 'bold 13px Inter, sans-serif';
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = strokeColor;
      ctx.fillRect(mirrorX, Math.max(0, y - 26), textWidth + 16, 26);

      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(label, mirrorX + 8, Math.max(16, y - 8));
    }
  };

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
              setIsCameraActive(true);
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
      // Fallback local summary if backend call fails
      onStopVerification({
        session_id: sessionId,
        status: 'completed',
        duration_seconds: 15.0,
        total_frames: stats.total,
        matched_frames: stats.matched,
        unmatched_frames: stats.unmatched,
        no_face_frames: stats.noFace,
        multiple_faces_frames: 0,
        match_percentage: stats.total > 0 ? (stats.matched / stats.total) * 100 : 0,
        looking_away_percentage: stats.total > 0 ? (stats.lookingAway / stats.total) * 100 : 0,
        overall_verdict: stats.matched > stats.unmatched ? 'PASSED' : 'SUSPICIOUS',
        events: events.map((e) => ({ id: e.id, timestamp: Date.now() / 1000, type: e.title, details: { message: e.detail } })),
      });
    }
  };

  const matchRate = stats.total > 0 ? Math.round((stats.matched / stats.total) * 100) : 100;
  const gazeAttentionRate = stats.total > 0 ? Math.round(((stats.total - stats.lookingAway) / stats.total) * 100) : 100;

  return (
    <div className="live-verification-container">
      {/* Hidden off-screen canvas used for capturing jpeg frame */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {/* Top Header & HUD */}
      <div className="live-header">
        <div className="header-left">
          <div className="pulse-indicator">
            <span className="live-dot"></span>
            <span className="live-label">LIVE VERIFICATION ACTIVE</span>
          </div>
          <span className="session-id-pill">Session: {sessionId}</span>
        </div>

        <div className="header-right">
          <button
            type="button"
            className="btn btn-danger-stop"
            onClick={handleStop}
            id="btn-stop-verification"
            disabled={isStopping}
          >
            {isStopping ? 'Finalizing Session...' : '⏹ Stop Verification & View Results'}
          </button>
        </div>
      </div>

      {error && (
        <div className="live-error-banner">
          ⚠️ <strong>Error:</strong> {error}
          <button className="btn-small-link" onClick={onBackToCapture}>
            Return to Reference Capture
          </button>
        </div>
      )}

      {/* Main Verification Workspace: Live Viewport + Real-Time Telemetry */}
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

            {/* Live Status Overlay Ribbon */}
            <div className="hud-overlay-top">
              {latestFrame ? (
                <div className={`status-tag ${latestFrame.verification_status.toLowerCase()}`}>
                  <span className="tag-icon">
                    {latestFrame.verification_status === 'MATCH'
                      ? '✓'
                      : latestFrame.verification_status === 'NO_MATCH'
                      ? '✕'
                      : latestFrame.verification_status === 'UNKNOWN'
                      ? '❓'
                      : '⚠️'}
                  </span>
                  <span className="tag-text">
                    {latestFrame.verification_status === 'MATCH'
                      ? 'IDENTITY MATCH'
                      : latestFrame.verification_status === 'NO_MATCH'
                      ? 'IDENTITY MISMATCH'
                      : latestFrame.verification_status === 'UNKNOWN'
                      ? 'IDENTITY UNCERTAIN (HEAD TURNED)'
                      : latestFrame.verification_status === 'NO_FACE'
                      ? 'NO FACE DETECTED'
                      : 'MULTIPLE FACES DETECTED'}
                  </span>
                </div>
              ) : (
                <div className="status-tag initializing">
                  <span className="spinner-dot"></span> Ingesting frames...
                </div>
              )}

              {latestFrame && (
                <div className={`gaze-tag ${latestFrame.is_looking_away ? 'away' : 'center'}`}>
                  <span className="gaze-icon">
                    {latestFrame.gaze_direction === 'CENTER'
                      ? '🎯'
                      : latestFrame.gaze_direction === 'LOOKING_LEFT'
                      ? '⬅️'
                      : latestFrame.gaze_direction === 'LOOKING_RIGHT'
                      ? '➡️'
                      : latestFrame.gaze_direction === 'LOOKING_UP'
                      ? '⬆️'
                      : '⬇️'}
                  </span>
                  <span>{latestFrame.gaze_direction.replace('_', ' ')}</span>
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
                <span className="metric-title">Head Pose</span>
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

        {/* Real-time Telemetry & Live Event Log Sidebar */}
        <div className="telemetry-sidebar">
          {/* Reference Photo Comparison Card */}
          <div className="telemetry-card">
            <h4>Reference Identity</h4>
            <div className="reference-compare-row">
              <img src={referencePhoto} alt="Reference face" className="ref-compare-thumb" />
              <div className="ref-compare-details">
                <div className="detail-item">
                  <span className="detail-label">Status</span>
                  <span className="detail-value text-success">Active Session</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Algorithm</span>
                  <span className="detail-value">SFace embedding (OpenCV)</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Gaze Sensor</span>
                  <span className="detail-value">Iris & SolvePnP 6DoF</span>
                </div>
              </div>
            </div>
          </div>

          {/* Running Compliance Gauges */}
          <div className="telemetry-card">
            <h4>Live Compliance Metrics</h4>
            <div className="gauge-group">
              <div className="gauge-item">
                <div className="gauge-header">
                  <span>Match Consistency</span>
                  <span className="gauge-val">{matchRate}%</span>
                </div>
                <div className="gauge-track">
                  <div
                    className={`gauge-fill ${matchRate >= 80 ? 'good' : 'warning'}`}
                    style={{ width: `${matchRate}%` }}
                  ></div>
                </div>
              </div>

              <div className="gauge-item">
                <div className="gauge-header">
                  <span>Gaze Attention (Center)</span>
                  <span className="gauge-val">{gazeAttentionRate}%</span>
                </div>
                <div className="gauge-track">
                  <div
                    className={`gauge-fill ${gazeAttentionRate >= 70 ? 'good' : 'warning'}`}
                    style={{ width: `${gazeAttentionRate}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>

          {/* Live Timestamped Event Log (Phases 4 & 5) */}
          <div className="telemetry-card events-card">
            <div className="events-card-header">
              <h4>Real-Time Event Stream</h4>
              <span className="event-count-badge">{events.length} events</span>
            </div>
            <div className="events-stream-list">
              {events.length === 0 ? (
                <div className="events-empty">
                  <span>🟢 Continuous valid face and center gaze stream</span>
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
        </div>
      </div>
    </div>
  );
}
