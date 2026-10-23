import React, { useState, useRef, useEffect, useCallback } from 'react';
import './CameraCapture.css';

/**
 * CameraCapture component for Phase 1: Reference Photo Capture
 * Handles webcam stream initialization, mirror preview, oval face guide,
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
        setError('Camera permission denied. Please allow camera access in your browser address bar and click "Open Camera".');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setError('No camera device was found. Please ensure a webcam is connected and enabled.');
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        setError('Webcam is currently in use by another application. Please close other camera apps and retry.');
      } else {
        setError(err.message || 'Unable to access camera.');
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

    // Mirror horizontal to match user mirror view
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, width, height);

    // Export high-quality JPEG
    const imageDataUrl = canvas.toDataURL('image/jpeg', 0.95);

    // Stop webcam tracks immediately after capture so webcam light turns off
    stopCamera();

    setCapturedImage(imageDataUrl);
    setIsConfirmed(false);
    setError(null);
    setValidationResult(null);
  };

  // Reset captured image and reactivate webcam
  const handleRetake = () => {
    setCapturedImage(null);
    setIsConfirmed(false);
    setError(null);
    setValidationResult(null);
    startCamera();
  };

  // Confirm photo and validate via Phase 2 backend
  const handleConfirm = async () => {
    if (!capturedImage || isValidating) return;

    setIsValidating(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/reference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: capturedImage }),
      });

      if (!response.ok) {
        throw new Error(`Server returned error status ${response.status}`);
      }

      const data = await response.json();

      if (!data.success) {
        setError(data.message || 'Face validation failed. Please retake the photo.');
        setIsValidating(false);
        return;
      }

      // Valid single face detected and session created!
      setIsConfirmed(true);
      setValidationResult(data);
      setIsValidating(false);

      if (onConfirmPhoto) {
        onConfirmPhoto(capturedImage, data.session_id, data);
      }
    } catch (err) {
      console.error('Validation request error:', err);
      setError(`Backend error: ${err.message}. Ensure the FastAPI server is running on port 8000.`);
      setIsValidating(false);
    }
  };

  return (
    <div className="camera-capture-container">
      {/* Hidden canvas for snapshot rasterization */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {/* Header section with phase badge */}
      <div className="camera-header">
        <div className="camera-badge">
          <span className={`status-dot ${isConfirmed ? 'confirmed' : cameraActive ? 'active' : 'idle'}`}></span>
          {isConfirmed
            ? 'Reference Photo Confirmed'
            : capturedImage
            ? 'Photo Captured — Review & Confirm'
            : cameraActive
            ? 'Live Webcam Active'
            : 'Camera Ready'}
        </div>
        <h2 className="camera-title">Capture Reference Photo</h2>
        <p className="camera-subtitle">
          Position your face clearly in the center under good lighting. This image establishes your reference identity.
        </p>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="camera-error-banner" role="alert">
          <div className="error-icon">⚠️</div>
          <div className="error-content">
            <strong>Camera Error:</strong>
            <p>{error}</p>
          </div>
          <button className="error-retry-btn" onClick={startCamera}>
            Retry
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
              {isConfirmed ? '✅ Confirmed Reference' : '📷 Captured Snapshot'}
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
                <div className="placeholder-icon">📷</div>
                <p className="placeholder-text">Camera is currently paused or inactive</p>
                <button className="btn btn-primary" onClick={startCamera}>
                  Start Webcam
                </button>
              </div>
            )}

            {isLoadingCamera && (
              <div className="camera-loading">
                <div className="spinner"></div>
                <p>Initializing camera sensor...</p>
              </div>
            )}

            {/* Oval Face Guide Overlay */}
            {cameraActive && showGuide && (
              <div className="face-guide-overlay" title="Align your face within the oval">
                <div className="face-oval">
                  <div className="crosshair-h"></div>
                  <div className="crosshair-v"></div>
                </div>
                <span className="guide-hint">Align your face inside the oval</span>
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
                {showGuide ? 'Hide Guide' : 'Show Guide'}
              </button>
              <button
                type="button"
                className="btn btn-capture"
                onClick={handleCapture}
                id="btn-capture-photo"
              >
                <span className="shutter-circle"></span>
                Capture Photo
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={stopCamera}
                title="Turn off webcam"
              >
                Stop Camera
              </button>
            </div>
          ) : (
            <button
              type="button"
              className="btn btn-primary btn-large"
              onClick={startCamera}
              id="btn-open-camera"
            >
              Open Camera
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
              🔄 Retake Photo
            </button>
            <button
              type="button"
              className={`btn ${isConfirmed ? 'btn-confirmed' : 'btn-success'}`}
              onClick={handleConfirm}
              id="btn-confirm-photo"
              disabled={isConfirmed || isValidating}
            >
              {isValidating
                ? '⏳ Validating Face...'
                : isConfirmed
                ? '✓ Reference Confirmed'
                : '✓ Confirm Photo'}
            </button>
          </div>
        )}
      </div>

      {/* Confirmed info alert */}
      {isConfirmed && (
        <div className="confirmation-banner">
          <div className="banner-icon">🎯</div>
          <div className="banner-text">
            <strong>Reference identity verified!</strong>
            <p>
              Phase 1 & Phase 2 complete. Face validated
              {validationResult?.quality_score ? ` (Quality: ${validationResult.quality_score}%)` : ''}.
              {validationResult?.session_id ? ` Session ID: ${validationResult.session_id}` : ''}
            </p>
          </div>
          <button className="btn-small-link" onClick={handleRetake}>
            Change Photo
          </button>
        </div>
      )}
    </div>
  );
}
