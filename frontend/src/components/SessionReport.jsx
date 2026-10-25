import { useState, useRef, useMemo } from 'react';
import { Download, Plus, ArrowLeft, CheckCircle2 } from 'lucide-react';
import { Button, Badge, Disclosure, KeyValue } from './ui';
import {
  VERDICT,
  INTERVAL_TYPE,
  formatClock,
  formatSeconds,
  formatPct,
  formatDateTime,
  passText,
  reasonText,
} from '../lib/format';
import '../css/SessionReport.css';

const TRACKS = [
  { key: 'identity', label: 'Identity' },
  { key: 'attention', label: 'Attention' },
];

export function Timeline({ duration, intervals, selectedId, onSelect }) {
  const safe = duration > 0 ? duration : 1;
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  return (
    <div className="timeline">
      {TRACKS.map((track) => (
        <div key={track.key} className="timeline-row">
          <span className="timeline-label">{track.label}</span>
          <div className="timeline-track">
            {intervals
              .filter((iv) => INTERVAL_TYPE[iv.type] && INTERVAL_TYPE[iv.type].track === track.key)
              .map((iv) => {
                const left = (iv.start_s / safe) * 100;
                const width = Math.max(((iv.duration_s || 0) / safe) * 100, 0.4);
                const tone = INTERVAL_TYPE[iv.type].tone;
                return (
                  <button
                    key={iv.id}
                    type="button"
                    className={'timeline-seg seg-' + tone + (iv.id === selectedId ? ' is-selected' : '')}
                    style={{ left: left + '%', width: width + '%' }}
                    title={
                      iv.message +
                      ' · ' +
                      formatClock(iv.start_s) +
                      '–' +
                      formatClock(iv.end_s) +
                      ' · ' +
                      formatSeconds(iv.duration_s)
                    }
                    aria-label={iv.message + ' at ' + formatClock(iv.start_s)}
                    onClick={() => onSelect(iv.id)}
                  />
                );
              })}
          </div>
        </div>
      ))}
      <div className="timeline-axis">
        {ticks.map((f) => (
          <span key={f} style={{ left: f * 100 + '%' }}>
            {formatClock(f * duration)}
          </span>
        ))}
      </div>
    </div>
  );
}

function VerdictCard({ kind, verdict, reasons = [], metrics = {} }) {
  const verdictInfo = VERDICT[verdict] || { label: verdict || 'Unknown', tone: 'neutral' };
  const sentences =
    reasons.length === 0
      ? [passText(kind, metrics)]
      : reasons.slice(0, 2).map((r) => reasonText(r, metrics));
  const extraCount = reasons.length > 2 ? reasons.length - 2 : 0;
  const tone = verdictInfo.tone;

  return (
    <div
      className={`verdict-card tone-${tone}`}
      style={{ borderLeft: `3px solid var(--${tone})` }}
    >
      <div className="verdict-card-top">
        <h3 className="verdict-card-title">{kind === 'face' ? 'Identity' : 'Attention'}</h3>
        <Badge tone={tone}>{verdictInfo.label}</Badge>
      </div>
      <div className="verdict-card-body">
        {sentences.map((s, idx) => (
          <p key={idx} className="verdict-sentence">
            {s}
          </p>
        ))}
        {extraCount > 0 && <p className="verdict-extra">+{extraCount} more</p>}
      </div>
    </div>
  );
}

export default function SessionReport({
  report,
  summary,
  reportSource = 'session',
  onNewSession,
  onBackToHistory,
}) {
  const activeReport = report || summary;
  const timelineRef = useRef(null);

  const duration_s = activeReport?.duration_s ?? 0;
  const intervals = activeReport?.intervals || [];
  const metrics = activeReport?.metrics || {};

  const [selectedId, setSelectedId] = useState(() => (intervals.length > 0 ? intervals[0].id : null));

  // Sync selectedId if intervals change and selectedId is invalid
  const selectedInterval = useMemo(() => {
    if (!intervals.length) return null;
    return intervals.find((iv) => iv.id === selectedId) || intervals[0];
  }, [intervals, selectedId]);

  if (!activeReport) {
    return (
      <div className="report-screen">
        <div className="issues-empty">
          <p>No report available.</p>
        </div>
      </div>
    );
  }

  const handleSelect = (id, shouldScroll = false) => {
    setSelectedId(id);
    if (shouldScroll && timelineRef.current) {
      timelineRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  };

  const handleDownloadReport = () => {
    const blob = new Blob([JSON.stringify(activeReport, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `report-${activeReport.session_id || 'session'}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Verdict details
  const faceVerdict = activeReport.face?.verdict || 'REVIEW';
  const faceReasons = activeReport.face?.reasons || [];
  const gazeVerdict = activeReport.gaze?.verdict || 'REVIEW';
  const gazeReasons = activeReport.gaze?.reasons || [];

  // Key numbers
  const hideTimeOnScreen =
    gazeReasons.includes('calibration_incomplete') || gazeReasons.includes('no_data');
  const timeOnScreenValue = hideTimeOnScreen ? '—' : formatPct(metrics.time_on_screen_pct);

  // Direction breakdown
  const awayByDir = metrics.away_by_direction_s || {};
  const dirRows = [
    { key: 'LEFT', label: 'Left', val: awayByDir.LEFT ?? awayByDir.left ?? 0 },
    { key: 'RIGHT', label: 'Right', val: awayByDir.RIGHT ?? awayByDir.right ?? 0 },
    { key: 'UP', label: 'Up', val: awayByDir.UP ?? awayByDir.up ?? 0 },
    { key: 'DOWN', label: 'Down', val: awayByDir.DOWN ?? awayByDir.down ?? 0 },
  ];
  const maxDirVal = Math.max(...dirRows.map((d) => d.val), 0);
  const showDirectionCard = dirRows.some((d) => d.val > 0);

  // Technical details
  const techItems = [
    { label: 'Session ID', value: activeReport.session_id || '—' },
    { label: 'Rules version', value: activeReport.rules_version || '1.0' },
    { label: 'Frames analysed', value: String(activeReport.total_frames ?? metrics.total_frames ?? 0) },
    { label: 'Frames per second', value: String(metrics.fps ?? 0) },
    { label: 'Similarity threshold', value: String(metrics.similarity?.threshold ?? '—') },
    {
      label: 'Similarity min / 5th % / median',
      value: `${metrics.similarity?.min ?? '—'} / ${metrics.similarity?.p05 ?? '—'} / ${metrics.similarity?.median ?? '—'}`,
    },
    { label: 'Frames near threshold', value: String(metrics.similarity?.near_threshold_frames ?? 0) },
    {
      label: 'Calibration completed / unstable',
      value: `${metrics.calibration?.completed ? 'Yes' : 'No'} / ${metrics.calibration?.unstable ? 'Yes' : 'No'}`,
    },
    { label: 'Blink count', value: String(metrics.blink_count ?? 0) },
    { label: 'Static face check', value: metrics.static_face ? 'Suspicious' : 'Normal' },
  ];

  const evidenceSrc = (img) => {
    if (!img) return null;
    return img.startsWith('data:') ? img : `data:image/jpeg;base64,${img}`;
  };

  return (
    <div className="report-screen">
      {/* Header */}
      <header className="report-header">
        <div className="report-header-titles">
          <h1>Session report</h1>
          <p>
            {formatDateTime(activeReport.created_at)} · {formatSeconds(duration_s)}
          </p>
        </div>
        <div className="report-header-actions">
          {reportSource === 'history' && onBackToHistory && (
            <Button variant="ghost" icon={ArrowLeft} onClick={onBackToHistory}>
              Back to history
            </Button>
          )}
          <Button variant="secondary" icon={Download} onClick={handleDownloadReport}>
            Download report
          </Button>
          {onNewSession && (
            <Button variant="primary" icon={Plus} onClick={onNewSession}>
              New session
            </Button>
          )}
        </div>
      </header>

      {/* Dual Verdict Cards */}
      <section className="verdict-grid" aria-label="Verdicts">
        <VerdictCard
          kind="face"
          verdict={faceVerdict}
          reasons={faceReasons}
          metrics={metrics}
        />
        <VerdictCard
          kind="gaze"
          verdict={gazeVerdict}
          reasons={gazeReasons}
          metrics={metrics}
        />
      </section>

      {/* Key Numbers Row */}
      <section className="key-numbers-row" aria-label="Key numbers">
        <div className="key-number-item">
          <span className="key-number-label">Duration</span>
          <span className="key-number-value num">{formatClock(duration_s)}</span>
        </div>
        <div className="key-number-item">
          <span className="key-number-label">Identity match</span>
          <span className="key-number-value num">{formatPct(metrics.match_pct)}</span>
        </div>
        <div className="key-number-item">
          <span className="key-number-label">Time on screen</span>
          <span className="key-number-value num">{timeOnScreenValue}</span>
        </div>
        <div className="key-number-item">
          <span className="key-number-label">Issues</span>
          <span className="key-number-value num">{intervals.length}</span>
        </div>
      </section>

      {/* Timeline Card */}
      <section className="timeline-card" ref={timelineRef} aria-label="Timeline">
        <h3>Timeline</h3>
        <Timeline
          duration={duration_s}
          intervals={intervals}
          selectedId={selectedInterval?.id}
          onSelect={(id) => handleSelect(id, false)}
        />

        {selectedInterval && (
          <div className="timeline-selected-box">
            <div className="selected-info">
              <span className="selected-title">
                Selected: {selectedInterval.message} · {formatClock(selectedInterval.start_s)}–
                {formatClock(selectedInterval.end_s)} · {formatSeconds(selectedInterval.duration_s)}
              </span>
            </div>
            <div className="selected-evidence">
              {(selectedInterval.evidence_image || selectedInterval.thumbnail || selectedInterval.proof_image) ? (
                <img
                  src={evidenceSrc(selectedInterval.evidence_image || selectedInterval.thumbnail || selectedInterval.proof_image)}
                  alt={selectedInterval.message}
                  className="selected-evidence-thumb"
                />
              ) : (
                <span className="selected-no-evidence">No image</span>
              )}
            </div>
          </div>
        )}
      </section>

      {/* Issues Table & Direction Breakdown */}
      <section
        className={`report-bottom-grid ${!showDirectionCard ? 'full-width' : ''}`}
        aria-label="Issues and direction details"
      >
        <div className="issues-card">
          <h3>Issues</h3>
          {intervals.length === 0 ? (
            <div className="issues-empty">
              <CheckCircle2 size={20} className="issues-empty-icon" aria-hidden="true" />
              <span>No issues detected.</span>
            </div>
          ) : (
            <div className="issues-table-wrap">
              <table className="issues-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Duration</th>
                    <th>Issue</th>
                    <th>Evidence</th>
                  </tr>
                </thead>
                <tbody>
                  {intervals.map((iv) => {
                    const isSelected = iv.id === selectedInterval?.id;
                    const timeRange = `${formatClock(iv.start_s)}–${formatClock(iv.end_s)}`;
                    return (
                      <tr
                        key={iv.id}
                        className={isSelected ? 'is-selected' : ''}
                        onClick={() => handleSelect(iv.id, false)}
                      >
                        <td className="num">{timeRange}</td>
                        <td className="num">{formatSeconds(iv.duration_s)}</td>
                        <td>{iv.message}</td>
                        <td>
                          {(iv.evidence_image || iv.thumbnail || iv.proof_image) ? (
                            <img
                              src={evidenceSrc(iv.evidence_image || iv.thumbnail || iv.proof_image)}
                              alt={iv.message}
                              className="issue-thumb"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleSelect(iv.id, true);
                              }}
                            />
                          ) : (
                            <span className="issue-no-thumb">No image</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {showDirectionCard && (
          <div className="direction-card">
            <h3>Looking away by direction</h3>
            <div className="direction-list">
              {dirRows.map((dir) => {
                const widthPct = maxDirVal > 0 ? (dir.val / maxDirVal) * 100 : 0;
                return (
                  <div key={dir.key} className="direction-row">
                    <span className="direction-label">{dir.label}</span>
                    <div className="direction-bar-track">
                      <div
                        className="direction-bar-fill"
                        style={{ width: `${widthPct}%` }}
                      />
                    </div>
                    <span className="direction-val num">{formatSeconds(dir.val)}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </section>

      {/* Technical Details Disclosure */}
      <section className="tech-details-section">
        <Disclosure title="Technical details">
          <KeyValue items={techItems} />
        </Disclosure>
      </section>
    </div>
  );
}
