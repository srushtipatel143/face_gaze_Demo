import { Download, Plus } from 'lucide-react';
import { Disclosure } from './ui';
import { VERDICT, INTERVAL_TYPE, reasonText, passText, formatPct } from '../lib/format';
import '../css/SessionResults.css';

export default function SessionResults({ summary, referencePhoto, onNewSession }) {
  if (!summary) {
    return (
      <div className="results-container empty">
        <p>No session summary data available.</p>
        <button className="btn btn-primary" onClick={onNewSession}>
          New session
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

  const faceHeadline = faceVerdict === 'PASS'
    ? passText('face', metrics)
    : faceReasons.length > 0
    ? reasonText(faceReasons[0], metrics)
    : 'Identity check needs review.';

  const gazeHeadline = gazeVerdict === 'PASS'
    ? passText('gaze', metrics)
    : gazeReasons.length > 0
    ? reasonText(gazeReasons[0], metrics)
    : 'Attention check needs review.';

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
    downloadAnchor.setAttribute('download', `session_report_${session_id}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div className="session-results-container">
      {/* Top Session Actions Header */}
      <div className="results-header-bar">
        <div className="results-header-info">
          <span className="results-sub">Session report</span>
        </div>
        <div className="verdict-actions">
          <button className="btn btn-outline-light" onClick={exportSummaryJSON} id="btn-export-report">
            <Download size={16} aria-hidden="true" />
            <span>Download report</span>
          </button>
          <button className="btn btn-primary-new" onClick={onNewSession} id="btn-new-session">
            <Plus size={16} aria-hidden="true" />
            <span>New session</span>
          </button>
        </div>
      </div>

      {/* Independent Dual Verdicts: Face and Gaze */}
      <div className="dual-verdict-grid">
        {/* Face Identity Verdict Card */}
        <div className={`verdict-card ${faceVerdict.toLowerCase()}`}>
          <div className="verdict-card-header">
            <div className="verdict-card-title-group">
              <div>
                <h3 className="verdict-card-name">Identity</h3>
              </div>
            </div>
            <span className={`verdict-badge ${faceVerdict.toLowerCase()}`}>
              {VERDICT[faceVerdict]?.label || faceVerdict}
            </span>
          </div>

          <div className="verdict-card-body">
            <p className="verdict-headline">{faceHeadline}</p>
            {faceReasons.length > 1 && (
              <ul className="verdict-reasons-list">
                {faceReasons.slice(1).map((r, i) => (
                  <li key={i}>{reasonText(r, metrics)}</li>
                ))}
              </ul>
            )}
          </div>
        </div>

        {/* Eye-Gaze Verdict Card */}
        <div className={`verdict-card ${gazeVerdict.toLowerCase()}`}>
          <div className="verdict-card-header">
            <div className="verdict-card-title-group">
              <div>
                <h3 className="verdict-card-name">Attention</h3>
              </div>
            </div>
            <span className={`verdict-badge ${gazeVerdict.toLowerCase()}`}>
              {VERDICT[gazeVerdict]?.label || gazeVerdict}
            </span>
          </div>

          <div className="verdict-card-body">
            <p className="verdict-headline">{gazeHeadline}</p>
            {gazeReasons.length > 1 && (
              <ul className="verdict-reasons-list">
                {gazeReasons.slice(1).map((r, i) => (
                  <li key={i}>{reasonText(r, metrics)}</li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>

      {/* Liveness Note */}
      <div className="demo-limitations-notice">
        <div className="notice-content">
          <strong>Liveness note:</strong>
          <span>
            {' '}This check monitors for natural blinking and head movement. It does not include 3D depth sensing, so a printed photo or screen replay might match.
          </span>
        </div>
      </div>

      {/* Metric Cards Grid */}
      <div className="metrics-summary-grid">
        <div className="metric-card">
          <span className="metric-label">Identity match</span>
          <div className="metric-large-val text-emerald">
            {formatPct(metrics.match_pct ?? match_percentage)}
          </div>
          <span className="metric-footnote">{matched_frames} of {total_frames} frames matched</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Time on screen</span>
          <div className={`metric-large-val ${attentionPercentage >= 75 ? 'text-emerald' : 'text-amber'}`}>
            {attentionPercentage}%
          </div>
          <span className="metric-footnote">{looking_away_percentage.toFixed(1)}% looked away</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Session duration</span>
          <div className="metric-large-val text-sky">
            {duration_seconds}s
          </div>
          <span className="metric-footnote">Average {averageFps} FPS</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Issues</span>
          <div className={`metric-large-val ${multiple_faces_frames === 0 && no_face_frames === 0 ? 'text-emerald' : 'text-rose'}`}>
            {no_face_frames + multiple_faces_frames}
          </div>
          <span className="metric-footnote">
            {no_face_frames} missing, {multiple_faces_frames} second person
          </span>
        </div>
      </div>

      {/* Detailed Breakdown */}
      <div className="results-breakdown-grid">
        {/* Left Side: Reference Snapshot and Technical Details */}
        <div className="breakdown-card">
          <h3 className="section-title">Reference photo</h3>
          <div className="reference-profile-box">
            {referencePhoto && (
              <img src={referencePhoto} alt="Reference face snapshot" className="summary-reference-img" />
            )}
            <Disclosure title="Technical details">
              <div className="profile-details" style={{ marginTop: '8px' }}>
                <div className="profile-row">
                  <span className="row-k">Session ID</span>
                  <span className="row-v font-mono">{session_id}</span>
                </div>
                <div className="profile-row">
                  <span className="row-k">Total frames</span>
                  <span className="row-v">{total_frames}</span>
                </div>
                <div className="profile-row">
                  <span className="row-k">Unmatched frames</span>
                  <span className="row-v">{unmatched_frames}</span>
                </div>
                <div className="profile-row">
                  <span className="row-k">Face visibility</span>
                  <span className="row-v text-emerald">
                    {total_frames > 0 ? ((1 - no_face_frames / total_frames) * 100).toFixed(1) : 100}%
                  </span>
                </div>
                {metrics?.embedding_model && (
                  <div className="profile-row">
                    <span className="row-k">Embedding model</span>
                    <span className="row-v">{metrics.embedding_model}</span>
                  </div>
                )}
              </div>
            </Disclosure>
          </div>
        </div>

        {/* Right Side: Chronological Issues Trail */}
        <div className="breakdown-card">
          <div className="trail-header">
            <h3 className="section-title">Issues</h3>
            <span className="trail-count">{intervals.length > 0 ? intervals.length : events.length} episodes</span>
          </div>

          <div className="issues-timeline">
            {intervals.length > 0 ? (
              intervals.map((iv, idx) => (
                <div key={idx} className="timeline-item">
                  <div className={`timeline-marker ${iv.type}`}></div>
                  <div className="timeline-content">
                    <div className="timeline-top">
                      <span className={`interval-badge ${iv.type}`}>
                        {INTERVAL_TYPE[iv.type]?.label || iv.type.replace(/_/g, ' ')}
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
                No issues recorded during this session.
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
