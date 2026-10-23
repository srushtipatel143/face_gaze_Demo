import React from 'react';
import './SessionResults.css';

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
    events = [],
  } = summary;

  const attentionPercentage = Math.max(0, Math.round(100 - looking_away_percentage));

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
      {/* Top Banner / Verdict Card */}
      <div className={`verdict-banner ${overall_verdict.toLowerCase()}`}>
        <div className="verdict-left">
          <div className="verdict-icon">
            {overall_verdict === 'PASSED' ? '🛡️' : overall_verdict === 'SUSPICIOUS' ? '⚠️' : '🚨'}
          </div>
          <div className="verdict-text">
            <span className="verdict-sub">Audit Result • Phase 5 Summary</span>
            <h2 className="verdict-title">Verification {overall_verdict}</h2>
            <p className="verdict-desc">
              {overall_verdict === 'PASSED'
                ? 'Identity confirmed with continuous high facial similarity and attentiveness.'
                : overall_verdict === 'SUSPICIOUS'
                ? 'Identity verification flagged anomalous gaze deviations or intermittent face loss.'
                : 'Session failed biometric thresholds or detected unverified faces.'}
            </p>
          </div>
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
          <span className="metric-footnote">Average 5.0 FPS</span>
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

        {/* Right Side: Chronological Audit Trail */}
        <div className="breakdown-card">
          <div className="trail-header">
            <h3 className="section-title">Timestamped Event Audit Trail</h3>
            <span className="trail-count">{events.length} records</span>
          </div>

          <div className="audit-timeline">
            {events.length === 0 ? (
              <div className="timeline-empty">
                No anomaly events were triggered during this session.
              </div>
            ) : (
              events.map((evt, idx) => (
                <div key={evt.id || idx} className="timeline-item">
                  <div className="timeline-marker"></div>
                  <div className="timeline-content">
                    <div className="timeline-top">
                      <strong className="timeline-title">{evt.type}</strong>
                      <span className="timeline-time">
                        {evt.timestamp ? new Date(evt.timestamp * 1000).toLocaleTimeString() : 'T+event'}
                      </span>
                    </div>
                    <p className="timeline-desc">
                      {evt.details?.message || JSON.stringify(evt.details)}
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
