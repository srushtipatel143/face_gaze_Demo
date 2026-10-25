import { useState, useEffect } from 'react';
import {
  Camera,
  RotateCcw,
  Check,
  Play,
  Circle,
  CheckCircle2,
  XCircle,
  MinusCircle,
} from 'lucide-react';
import { useCamera } from '../hooks/useCamera';
import { Button, Badge, Alert } from './ui';
import { API_BASE } from '../config';
import '../css/CameraCapture.css';

const DEFAULT_CHECKS = [
  { id: 'face_detected', label: 'Face detected', passed: null, detail: null },
  { id: 'single_face', label: 'Only one person', passed: null, detail: null },
  { id: 'face_size', label: 'Close enough to the camera', passed: null, detail: null },
  { id: 'sharpness', label: 'Image is sharp', passed: null, detail: null },
  { id: 'lighting', label: 'Good lighting', passed: null, detail: null },
  { id: 'facing_camera', label: 'Facing the camera', passed: null, detail: null },
  { id: 'eyes_open', label: 'Eyes open', passed: null, detail: null },
];

export default function CameraCapture({
  onAccepted,
  onStartLive,
  onConfirmPhoto,
  confirmedPhoto = null,
  initialReference = null,
}) {
  const { videoRef, status: cameraStatus, error: cameraError, start: startCamera, stop: stopCamera } = useCamera();

  const [capturedPhoto, setCapturedPhoto] = useState(
    initialReference?.photo || confirmedPhoto || null
  );
  const [stepState, setStepState] = useState(
    initialReference?.photo || confirmedPhoto ? 'accepted' : 'live'
  );
  const [checks, setChecks] = useState(
    initialReference?.checks || DEFAULT_CHECKS
  );
  const [networkError, setNetworkError] = useState(null);
  const [activeSessionId, setActiveSessionId] = useState(
    initialReference?.sessionId || null
  );

  // Auto-start camera when landing on reference screen without photo
  useEffect(() => {
    if (!initialReference?.photo && !confirmedPhoto) {
      startCamera();
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleTurnOnCamera = async () => {
    setCapturedPhoto(null);
    setChecks(DEFAULT_CHECKS);
    setNetworkError(null);
    setStepState('live');
    await startCamera();
  };

  const verifyPhoto = async (photoUrl) => {
    if (!photoUrl) return false;

    setStepState('checking');
    setNetworkError(null);

    try {
      const res = await fetch(`${API_BASE}/api/reference`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: photoUrl }),
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }

      const data = await res.json();
      setChecks(data.checks || DEFAULT_CHECKS);

      if (data.session_id) {
        setActiveSessionId(data.session_id);
        if (onAccepted) {
          onAccepted({
            photo: photoUrl,
            sessionId: data.session_id,
            checks: data.checks,
          });
        }
        if (onConfirmPhoto) {
          onConfirmPhoto(photoUrl, data.session_id, data);
        }
      }

      if (data.success) {
        setStepState('accepted');
        return true;
      } else {
        setStepState('rejected');
        return Boolean(data.session_id);
      }
    } catch (err) {
      console.error('Reference photo check failed:', err);
      setNetworkError('Could not reach the verification service. Try again.');
      setStepState('preview');
      return false;
    }
  };

  const handleCapture = async () => {
    const video = videoRef.current;
    if (!video) return;

    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Draw mirrored video frame to canvas so still image matches what user saw
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, width, height);

    const dataUrl = canvas.toDataURL('image/jpeg', 0.95);
    setCapturedPhoto(dataUrl);
    stopCamera();
    setChecks(DEFAULT_CHECKS);
    setNetworkError(null);

    // Immediately verify the photo so Start live check appears without extra clicks
    await verifyPhoto(dataUrl);
  };

  const handleRetake = async () => {
    setCapturedPhoto(null);
    setChecks(DEFAULT_CHECKS);
    setNetworkError(null);
    setActiveSessionId(null);
    setStepState('live');
    await startCamera();
  };

  const handleStartLiveClick = async () => {
    if (activeSessionId) {
      if (onStartLive) onStartLive();
      return;
    }
    if (capturedPhoto) {
      await verifyPhoto(capturedPhoto);
      if (onStartLive) onStartLive();
    }
  };

  const isLive = stepState === 'live';
  const isCameraOff = (stepState === 'off' || cameraStatus === 'off') && !capturedPhoto && stepState !== 'live';

  return (
    <div className="reference-screen">
      {/* Left Column: Reference Photo Viewport & Controls */}
      <div className="reference-main">
        <div className="reference-header">
          <h2>Reference photo</h2>
          <p>
            Take a clear photo of your face. It is used to confirm it is you during the live check.
          </p>
        </div>

        {cameraError && (
          <Alert
            tone="danger"
            action={
              <Button variant="secondary" size="sm" onClick={startCamera}>
                Try again
              </Button>
            }
          >
            {cameraError}
          </Alert>
        )}

        {networkError && (
          <Alert tone="danger">{networkError}</Alert>
        )}

        {/* 16:9 Black Viewport Area */}
        <div className="camera-viewport">
          {capturedPhoto ? (
            <>
              <img
                src={capturedPhoto}
                alt="Captured reference face"
                className="camera-still"
              />
              {stepState === 'checking' && (
                <div className="checking-overlay">
                  <div className="camera-spinner" aria-label="Checking photo" />
                </div>
              )}
              {stepState === 'accepted' && (
                <div className="accepted-badge-container">
                  <Badge tone="success">Photo accepted</Badge>
                </div>
              )}
            </>
          ) : isLive ? (
            <>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="camera-video"
              />
              {cameraStatus === 'starting' && (
                <div className="checking-overlay">
                  <div className="camera-spinner" aria-label="Starting camera" />
                </div>
              )}
              <div className="face-oval-outline" aria-hidden="true" />
            </>
          ) : (
            <div className="camera-off-state">
              <Camera size={32} aria-hidden="true" />
              <p>Camera is off</p>
            </div>
          )}
        </div>

        {/* Action Buttons Below Video Area */}
        <div className="camera-actions">
          {isCameraOff && (
            <Button
              variant="primary"
              icon={Camera}
              onClick={handleTurnOnCamera}
              id="btn-turn-on-camera"
            >
              Turn on camera
            </Button>
          )}

          {isLive && (
            <Button
              variant="primary"
              icon={Camera}
              onClick={handleCapture}
              disabled={cameraStatus !== 'on'}
              id="btn-take-photo"
            >
              {cameraStatus === 'starting' ? 'Starting camera…' : 'Take photo'}
            </Button>
          )}

          {stepState === 'preview' && (
            <>
              <Button
                variant="secondary"
                icon={RotateCcw}
                onClick={handleRetake}
                id="btn-retake-photo"
              >
                Retake
              </Button>
              <Button
                variant="primary"
                icon={Play}
                onClick={handleUsePhoto}
                id="btn-use-photo"
              >
                Start live check
              </Button>
            </>
          )}

          {stepState === 'checking' && (
            <>
              <Button variant="secondary" icon={RotateCcw} disabled>
                Retake
              </Button>
              <Button variant="primary" disabled id="btn-use-photo">
                <span className="spinner-inline" aria-hidden="true" />
                Checking photo…
              </Button>
            </>
          )}

          {stepState === 'rejected' && (
            <>
              <Button
                variant="secondary"
                icon={RotateCcw}
                onClick={handleRetake}
                id="btn-retake-photo"
              >
                Retake
              </Button>
              <Button
                variant="primary"
                icon={Play}
                onClick={handleStartLiveClick}
                id="btn-start-live-check-rejected"
              >
                Start live check
              </Button>
            </>
          )}

          {stepState === 'accepted' && (
            <>
              <Button
                variant="secondary"
                icon={RotateCcw}
                onClick={handleRetake}
                id="btn-retake-photo"
              >
                Retake
              </Button>
              <Button
                variant="primary"
                icon={Play}
                onClick={handleStartLiveClick}
                id="btn-start-live-check-main"
              >
                Start live check
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Right Column: Structured Checklist & Tips */}
      <div className="checks-card">
        <h3>Photo checks</h3>

        <ul className="checklist">
          {checks.map((item) => {
            let icon = <Circle size={16} className="check-icon check-pending" aria-hidden="true" />;
            let detailText = null;

            if (stepState === 'rejected' || stepState === 'accepted') {
              if (item.passed === true) {
                icon = <CheckCircle2 size={16} className="check-icon check-passed" aria-hidden="true" />;
              } else if (item.passed === false) {
                icon = <XCircle size={16} className="check-icon check-failed" aria-hidden="true" />;
                detailText = item.detail;
              } else if (item.passed === null) {
                icon = <MinusCircle size={16} className="check-icon check-null" aria-hidden="true" />;
                detailText = 'Not checked';
              }
            }

            return (
              <li key={item.id} className="checklist-item">
                {icon}
                <div className="check-content">
                  <span className="check-label">{item.label}</span>
                  {detailText && (
                    <span className={`check-detail ${item.passed === false ? 'error-detail' : ''}`}>
                      {detailText}
                    </span>
                  )}
                </div>
              </li>
            );
          })}
        </ul>

        {/* Tips: Only shown before a photo is taken */}
        {!capturedPhoto && stepState !== 'accepted' && stepState !== 'rejected' && (
          <div className="photo-tips">
            <h4 className="tips-title">Tips</h4>
            <ul className="tips-list">
              <li>Face a window or lamp.</li>
              <li>Remove sunglasses.</li>
            </ul>
          </div>
        )}

        {/* Start Live Check Button */}
        {(stepState === 'accepted' || stepState === 'rejected' || stepState === 'preview') && (
          <Button
            variant="primary"
            size="lg"
            icon={Play}
            onClick={handleStartLiveClick}
            id="btn-start-live-check"
            className="btn-start-live"
          >
            Start live check
          </Button>
        )}
      </div>
    </div>
  );
}
