import React from 'react';
import '../css/SessionResults.css';

export default function SessionResults({ summary, referencePhoto, onNewSession }) {
  if (!summary) {
    return (
      <div className="results-container empty">
        <p>No session summary data available.</p>
        <button className="btn btn-primary" onClick={onNewSession}>
          Start New Verification
        </button>
      </div>
    );
  }

  const {
    session_id,
    duration_seconds = 0,
    total_frames = 0,
    matched_frames = 0,
    unmatched_frames = 0,
    no_face_frames = 0,
    multiple_faces_frames = 0,
    match_percentage = 0,
    looking_away_percentage = 0,
    overall_verdict = 'PASSED',
    face,
    gaze,
    metrics = {},
    events = [],
    intervals = [],
  } = summary;

  const formatTimeRange = (startS, endS, durationS) => {
    const pad = (n) => String(Math.floor(n)).padStart(2, '0');
    const formatS = (s) => {
      const totalSecs = Math.max(0, s);
      const mins = Math.floor(totalSecs / 60);
      const secs = Math.floor(totalSecs % 60);
      return `${pad(mins)}:${pad(secs)}`;
    };
    const dur = durationS !== undefined && durationS !== null ? Number(durationS).toFixed(1) : (endS - startS).toFixed(1);
    return `${formatS(startS)} – ${formatS(endS)} (${dur}s)`;
  };

  const faceVerdict = face?.verdict || (overall_verdict === 'PASSED' ? 'PASS' : overall_verdict === 'FAILED' ? 'FAIL' : 'REVIEW');
  const gazeVerdict = gaze?.verdict || (looking_away_percentage <= 20 ? 'PASS' : looking_away_percentage <= 35 ? 'REVIEW' : 'FAIL');
  const faceReasons = face?.reasons || [];
  const gazeReasons = gaze?.reasons || [];

  const formatFaceReason = (r) => {
    switch (r) {
      case 'no_data':
        return 'No face data evaluated';
      case 'different_person':
        return `Different person visible for ${metrics.different_person_s ?? 0} s`;
      case 'multiple_faces':
        return `A second person was visible for ${metrics.multiple_faces_s ?? 0} s`;
      case 'no_face':
        return `Face missing from camera view for ${metrics.no_face_pct ?? 0}% of session`;
      case 'identity_unknown':
        return `Face angle too steep / uncertain for ${metrics.identity_unknown_pct ?? 0}% of evaluations`;
      case 'static_face_detected':
        return 'Possible static photo: no natural blinks or head micro-movements detected in 30+ seconds';
      default:
        return r.replace(/_/g, ' ');
    }
  };

  const formatGazeReason = (r) => {
    switch (r) {
      case 'no_data':
        return 'No valid gaze data evaluated';
      case 'no_stable_baseline':
        return 'Unstable gaze baseline: user was not looking at the screen during calibration';
      case 'looking_away':
        return `Looked away for ${metrics.away_pct ?? 0}% of evaluated gaze time`;
      case 'long_away_intervals':
        return `${metrics.long_away_intervals ?? 0} extended gaze deviation(s) (≥10s)`;
      case 'gaze_unknown':
        return `Gaze reading unavailable for ${metrics.gaze_unknown_pct ?? 0}% of session`;
      default:
        return r.replace(/_/g, ' ');
    }
  };

  const faceHeadline = faceReasons.length > 0
    ? `Face: ${faceVerdict.toLowerCase()} – ${formatFaceReason(faceReasons[0])}`
    : `Face: ${faceVerdict.toLowerCase()} – Identity confirmed with high biometric similarity`;

  const gazeHeadline = gazeReasons.length > 0
    ? `Gaze: ${gazeVerdict.toLowerCase()} – ${formatGazeReason(gazeReasons[0])}`
    : `Gaze: ${gazeVerdict.toLowerCase()} – Continuous center gaze attention maintained`;

  const attentionPercentage = Math.max(0, Math.round(100 - looking_away_percentage));

  const averageFps = (metrics?.fps !== undefined && metrics?.fps !== null)
    ? Number(metrics.fps).toFixed(1)
    : duration_seconds > 0
    ? (total_frames / duration_seconds).toFixed(1)
    : '0.0';

  const exportSummaryJSON = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(summary, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `verification_report_${session_id}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div className="session-results-container">
      {/* Top Session Actions Header */}
      <div className="results-header-bar">
        <div className="results-header-info">
          <span className="results-sub">Session Audit Report</span>
          <span className="results-session-pill">Session: {session_id}</span>
        </div>
        <div className="verdict-actions">
          <button className="btn btn-outline-light" onClick={exportSummaryJSON} id="btn-export-report">
            📥 Export Audit JSON
          </button>
          <button className="btn btn-primary-new" onClick={onNewSession} id="btn-new-session">
            🔄 New Session
          </button>
        </div>
      </div>

      {/* Independent Dual Verdicts: Face and Gaze */}
      <div className="dual-verdict-grid">
        {/* Face Identity Verdict Card */}
        <div className={`verdict-card ${faceVerdict.toLowerCase()}`}>
          <div className="verdict-card-header">
            <div className="verdict-card-title-group">
              <span className="verdict-card-icon">
                {faceVerdict === 'PASS' ? '🛡️' : faceVerdict === 'FAIL' ? '🚨' : '⚠️'}
              </span>
              <div>
                <span className="verdict-card-category">Identity Verification</span>
                <h3 className="verdict-card-name">Face Assessment</h3>
              </div>
            </div>
            <span className={`verdict-badge ${faceVerdict.toLowerCase()}`}>
              {faceVerdict}
            </span>
          </div>

          <div className="verdict-card-body">
            <p className="verdict-headline">{faceHeadline}</p>
            {faceReasons.length > 1 && (
              <ul className="verdict-reasons-list">
                {faceReasons.slice(1).map((r, i) => (
                  <li key={i}>{formatFaceReason(r)}</li>
                ))}
              </ul>
            )}
          </div>
        </div>

        {/* Eye-Gaze Verdict Card */}
        <div className={`verdict-card ${gazeVerdict.toLowerCase()}`}>
          <div className="verdict-card-header">
            <div className="verdict-card-title-group">
              <span className="verdict-card-icon">
                {gazeVerdict === 'PASS' ? '🎯' : gazeVerdict === 'FAIL' ? '🚨' : '⚠️'}
              </span>
              <div>
                <span className="verdict-card-category">Attention & Focus</span>
                <h3 className="verdict-card-name">Gaze Assessment</h3>
              </div>
            </div>
            <span className={`verdict-badge ${gazeVerdict.toLowerCase()}`}>
              {gazeVerdict}
            </span>
          </div>

          <div className="verdict-card-body">
            <p className="verdict-headline">{gazeHeadline}</p>
            {gazeReasons.length > 1 && (
              <ul className="verdict-reasons-list">
                {gazeReasons.slice(1).map((r, i) => (
                  <li key={i}>{formatGazeReason(r)}</li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>

      {/* Demo Limitation & Anti-Spoofing Notice (M8) */}
      <div className="demo-limitations-notice">
        <div className="notice-icon">ℹ️</div>
        <div className="notice-content">
          <strong>Anti-Spoofing / Liveness Note:</strong>
          <span>
            {' '}This demonstration does not include passive 3D anti-spoofing or active challenge-response verification. A high-resolution printed photo or phone screen replay of the enrolled subject may match identity. A heuristic check monitors for blinks and head micro-movements over 30+ seconds.
          </span>
        </div>
      </div>

      {/* Metric Cards Grid */}
      <div className="metrics-summary-grid">
        <div className="metric-card">
          <span className="metric-label">Biometric Match Rate</span>
          <div className="metric-large-val text-emerald">
            {match_percentage.toFixed(1)}%
          </div>
          <span className="metric-footnote">{matched_frames} of {total_frames} frames matched</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Attention (Center Gaze)</span>
          <div className={`metric-large-val ${attentionPercentage >= 75 ? 'text-emerald' : 'text-amber'}`}>
            {attentionPercentage}%
          </div>
          <span className="metric-footnote">{looking_away_percentage.toFixed(1)}% looking away</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Session Duration</span>
          <div className="metric-large-val text-sky">
            {duration_seconds}s
          </div>
          <span className="metric-footnote">Average {averageFps} FPS</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Integrity Flags</span>
          <div className={`metric-large-val ${multiple_faces_frames === 0 && no_face_frames === 0 ? 'text-emerald' : 'text-rose'}`}>
            {no_face_frames + multiple_faces_frames}
          </div>
          <span className="metric-footnote">
            {no_face_frames} missing, {multiple_faces_frames} multi-face
          </span>
        </div>
      </div>

      {/* Detailed Breakdown & Audit Trail */}
      <div className="results-breakdown-grid">
        {/* Left Side: Biometric Comparison Snapshot */}
        <div className="breakdown-card">
          <h3 className="section-title">Verified Identity Reference</h3>
          <div className="reference-profile-box">
            {referencePhoto && (
              <img src={referencePhoto} alt="Reference face snapshot" className="summary-reference-img" />
            )}
            <div className="profile-details">
              <div className="profile-row">
                <span className="row-k">Session Token</span>
                <span className="row-v font-mono">{session_id}</span>
              </div>
              <div className="profile-row">
                <span className="row-k">Total Evaluated Frames</span>
                <span className="row-v">{total_frames}</span>
              </div>
              <div className="profile-row">
                <span className="row-k">Unmatched Frames</span>
                <span className="row-v">{unmatched_frames}</span>
              </div>
              <div className="profile-row">
                <span className="row-k">Face Continuity</span>
                <span className="row-v text-emerald">
                  {total_frames > 0 ? ((1 - no_face_frames / total_frames) * 100).toFixed(1) : 100}%
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Side: Chronological Interval Audit Trail (L7) */}
        <div className="breakdown-card">
          <div className="trail-header">
            <h3 className="section-title">Interval Audit Trail</h3>
            <span className="trail-count">{intervals.length > 0 ? intervals.length : events.length} episodes</span>
          </div>

          <div className="audit-timeline">
            {intervals.length > 0 ? (
              intervals.map((iv, idx) => (
                <div key={idx} className="timeline-item">
                  <div className={`timeline-marker ${iv.type}`}></div>
                  <div className="timeline-content">
                    <div className="timeline-top">
                      <span className={`interval-badge ${iv.type}`}>
                        {iv.type.replace(/_/g, ' ')}
                      </span>
                      <span className="timeline-time">
                        {formatTimeRange(iv.start_s, iv.end_s, iv.duration_s)}
                      </span>
                    </div>
                    <p className="timeline-desc">{iv.details}</p>
                    {iv.thumbnail && (
                      <img src={iv.thumbnail} alt="Evidence thumbnail" className="interval-thumb" />
                    )}
                  </div>
                </div>
              ))
            ) : events.length === 0 ? (
              <div className="timeline-empty">
                No anomaly episodes were recorded during this session.
              </div>
            ) : (
              events.map((evt, idx) => (
                <div key={evt.id || idx} className="timeline-item">
                  <div className="timeline-marker"></div>
                  <div className="timeline-content">
                    <div className="timeline-top">
                      <strong className="timeline-title">{evt.type}</strong>
                      <span className="timeline-time">
                        {evt.timestamp
                          ? (typeof evt.timestamp === 'number' && evt.timestamp < 10000
                              ? `${evt.timestamp.toFixed(1)}s`
                              : new Date(evt.timestamp * 1000).toLocaleTimeString())
                          : 'T+event'}
                      </span>
                    </div>
                    <p className="timeline-desc">
                      {evt.message || evt.details?.message || JSON.stringify(evt.details || '')}
                    </p>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
