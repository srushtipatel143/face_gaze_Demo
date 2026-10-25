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
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setStatus('off');
  }, []);

  const start = useCallback(async () => {
    setError(null);
    setStatus('starting');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setStatus('on');
    } catch (err) {
      setError(ERRORS[err.name] || 'Could not start the camera.');
      setStatus('error');
    }
  }, []);

  useEffect(() => stop, [stop]);
  return { videoRef, status, error, start, stop };
}
