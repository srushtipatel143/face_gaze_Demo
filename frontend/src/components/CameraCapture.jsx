import { useState, useRef, useEffect, useCallback } from 'react';
import { API_BASE } from '../config';
import '../css/CameraCapture.css';

/**
 * CameraCapture component for Reference Photo Capture
 * Handles webcam stream initialization, mirror preview, outline face guide,
 * snapshot capture to canvas, track cleanup, retake, and confirmation.
 */
export default function CameraCapture({ onConfirmPhoto, confirmedPhoto = null }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [isLoadingCamera, setIsLoadingCamera] = useState(false);
  const [capturedImage, setCapturedImage] = useState(confirmedPhoto);
  const [isConfirmed, setIsConfirmed] = useState(Boolean(confirmedPhoto));
  const [error, setError] = useState(null);
  const [showGuide, setShowGuide] = useState(true);

  const [isValidating, setIsValidating] = useState(false);
  const [validationResult, setValidationResult] = useState(null);
  const [validationIssues, setValidationIssues] = useState([]);

  // Stop all active tracks on the camera stream
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setIsLoadingCamera(false);
  }, []);

  // Request camera permission and start video stream
  const startCamera = useCallback(async () => {
    setError(null);
    setValidationResult(null);
    setIsLoadingCamera(true);

    // Stop any existing stream first
    stopCamera();

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Your browser does not support webcam video capture (navigator.mediaDevices.getUserMedia is not available).');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280, min: 640 },
          height: { ideal: 720, min: 480 },
          facingMode: 'user',
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current
            .play()
            .then(() => {
              setCameraActive(true);
              setIsLoadingCamera(false);
            })
            .catch((err) => {
              console.error('Video play error:', err);
              setIsLoadingCamera(false);
            });
        };
      }
    } catch (err) {
      console.error('Webcam access error:', err);
      setIsLoadingCamera(false);
      setCameraActive(false);

      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setError("Camera access is blocked. Allow it in your browser's address bar, then try again.");
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setError('No camera found. Connect a camera and try again.');
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        setError('The camera is in use by another app. Close it and try again.');
      } else {
        setError(err.message || 'Could not access the camera. Check your browser permissions and try again.');
      }
    }
  }, [stopCamera]);

  // Clean up stream when component unmounts
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  // Capture still image from the video feed onto a canvas
  const handleCapture = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas) return;

    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;

    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Draw raw unmirrored video frame directly to canvas for backend face analysis
    ctx.drawImage(video, 0, 0, width, height);

    // Export high-quality JPEG
    const imageDataUrl = canvas.toDataURL('image/jpeg', 0.95);

    // Stop webcam tracks immediately after capture so webcam light turns off
    stopCamera();

    setCapturedImage(imageDataUrl);
    setIsConfirmed(false);
    setError(null);
    setValidationResult(null);
    setValidationIssues([]);
  };

  // Reset captured image and reactivate webcam
  const handleRetake = () => {
    setCapturedImage(null);
    setIsConfirmed(false);
    setError(null);
    setValidationResult(null);
    setValidationIssues([]);
    startCamera();
  };

  // Confirm photo and validate via reference endpoint
  const handleConfirm = async () => {
    if (!capturedImage || isValidating) return;

    setIsValidating(true);
    setError(null);
    setValidationIssues([]);

    try {
      const response = await fetch(`${API_BASE}/api/reference`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: capturedImage }),
      });

      if (!response.ok) {
        throw new Error(`Server returned error status ${response.status}`);
      }

      const data = await response.json();

      if (!data.success) {
        setError(data.message || 'Photo not accepted. Try again.');
        setValidationIssues(data.issues || []);
        setIsValidating(false);
        return;
      }

      // Valid single face detected and session created!
      setIsConfirmed(true);
      setValidationResult(data);
      setValidationIssues([]);
      setIsValidating(false);

      if (onConfirmPhoto) {
        onConfirmPhoto(capturedImage, data.session_id, data);
      }
    } catch (err) {
      console.error('Validation request error:', err);
      setError(`Connection error: ${err.message}. Check that the backend is running.`);
      setIsValidating(false);
    }
  };

  return (
    <div className="camera-capture-container">
      {/* Hidden canvas for snapshot rasterization */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {/* Header section */}
      <div className="camera-header">
        <div className="camera-badge">
          <span className={`status-dot ${isConfirmed ? 'confirmed' : cameraActive ? 'active' : 'idle'}`}></span>
          {isConfirmed
            ? 'Photo accepted'
            : capturedImage
            ? 'Review photo'
            : cameraActive
            ? 'Camera on'
            : 'Camera ready'}
        </div>
        <h2 className="camera-title">Reference photo</h2>
        <p className="camera-subtitle">
          Take a clear photo of your face. It is used to confirm it is you during the live check.
        </p>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="camera-error-banner" role="alert">
          <div className="error-content">
            <strong>Photo not accepted:</strong>
            <p>{error}</p>
            {validationIssues && validationIssues.length > 0 && (
              <ul className="error-issues-list" style={{ marginTop: '0.45rem', paddingLeft: '1.2rem', fontSize: '0.85rem', color: '#fca5a5' }}>
                {validationIssues.map((issue, idx) => (
                  <li key={idx}>{issue}</li>
                ))}
              </ul>
            )}
          </div>
          <button className="error-retry-btn" onClick={handleRetake}>
            Retake
          </button>
        </div>
      )}

      {/* Viewport Frame */}
      <div className="viewport-wrapper">
        {capturedImage ? (
          // Still Captured Preview
          <div className="preview-container">
            <img src={capturedImage} alt="Captured reference face" className="preview-image" />
            <div className="preview-overlay-tag">
              {isConfirmed ? 'Photo accepted' : 'Photo taken'}
            </div>
          </div>
        ) : (
          // Live Video Container
          <div className="video-container">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`webcam-video ${cameraActive ? 'visible' : 'hidden'}`}
            />

            {!cameraActive && !isLoadingCamera && (
              <div className="camera-placeholder">
                <p className="placeholder-text">Camera is off</p>
                <button className="btn btn-primary" onClick={startCamera}>
                  Turn on camera
                </button>
              </div>
            )}

            {isLoadingCamera && (
              <div className="camera-loading">
                <div className="spinner"></div>
                <p>Starting camera…</p>
              </div>
            )}

            {/* Face Guide Overlay */}
            {cameraActive && showGuide && (
              <div className="face-guide-overlay" title="Fit your face inside the outline">
                <div className="face-oval">
                  <div className="crosshair-h"></div>
                  <div className="crosshair-v"></div>
                </div>
                <span className="guide-hint">Fit your face inside the outline</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Toolbar Controls */}
      <div className="camera-controls">
        {!capturedImage ? (
          // Camera active state controls
          cameraActive ? (
            <div className="button-group">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setShowGuide((prev) => !prev)}
                title="Toggle alignment guide"
              >
                {showGuide ? 'Hide guide' : 'Show guide'}
              </button>
              <button
                type="button"
                className="btn btn-capture"
                onClick={handleCapture}
                id="btn-capture-photo"
              >
                <span className="shutter-circle"></span>
                Take photo
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={stopCamera}
                title="Turn off camera"
              >
                Turn off camera
              </button>
            </div>
          ) : (
            <button
              type="button"
              className="btn btn-primary btn-large"
              onClick={startCamera}
              id="btn-open-camera"
            >
              Turn on camera
            </button>
          )
        ) : (
          // Post-capture controls: Retake & Confirm
          <div className="button-group">
            <button
              type="button"
              className="btn btn-outline"
              onClick={handleRetake}
              id="btn-retake-photo"
            >
              Retake
            </button>
            <button
              type="button"
              className={`btn ${isConfirmed ? 'btn-confirmed' : 'btn-success'}`}
              onClick={handleConfirm}
              id="btn-confirm-photo"
              disabled={isConfirmed || isValidating}
            >
              {isValidating
                ? 'Checking photo…'
                : isConfirmed
                ? 'Photo accepted'
                : 'Use this photo'}
            </button>
          </div>
        )}
      </div>

      {/* Confirmed info alert */}
      {isConfirmed && (
        <div className="confirmation-banner">
          <div className="banner-text">
            <strong>Photo accepted</strong>
            <p>Your photo is ready for the live check.</p>
          </div>
          <button className="btn-small-link" onClick={handleRetake}>
            Retake
          </button>
        </div>
      )}
    </div>
  );
}
