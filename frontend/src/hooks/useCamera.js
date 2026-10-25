import { useCallback, useEffect, useRef, useState } from 'react';

const ERRORS = {
  NotAllowedError: "Camera access is blocked. Allow it in your browser's address bar, then try again.",
  NotFoundError: 'No camera found. Connect a camera and try again.',
  NotReadableError: 'The camera is in use by another app. Close it and try again.',
};

export function useCamera() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [status, setStatus] = useState('off'); // off | starting | on | error
  const [error, setError] = useState(null);

  const stop = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStatus('off');
  }, []);

  const attachStream = useCallback(() => {
    const video = videoRef.current;
    const stream = streamRef.current;
    if (video && stream) {
      if (video.srcObject !== stream) {
        video.srcObject = stream;
      }
      video.play().catch((err) => {
        // AbortError is common when play() is called rapidly or interrupted; ignore it
        if (err.name !== 'AbortError') {
          console.warn('Video play warning:', err);
        }
      });
    }
  }, []);

  const start = useCallback(async () => {
    setError(null);
    setStatus('starting');
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      });

      streamRef.current = stream;
      setStatus('on');

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        try {
          await videoRef.current.play();
        } catch (playErr) {
          if (playErr.name !== 'AbortError') {
            console.warn('Initial play warning:', playErr);
          }
        }
      }
    } catch (err) {
      console.error('Camera start error:', err);
      setError(ERRORS[err.name] || 'Could not start the camera.');
      setStatus('error');
    }
  }, []);

  // Ensure stream is bound whenever video element mounts or status turns on
  useEffect(() => {
    if (status === 'on') {
      attachStream();
    }
  });

  // Stop media tracks when the hook unmounts from DOM
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
    };
  }, []);

  return { videoRef, status, error, start, stop };
}
