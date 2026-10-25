import { useState, useEffect, useCallback } from 'react';
import { History as HistoryIcon, Plus } from 'lucide-react';
import { Button, Badge, Spinner, Alert } from './ui';
import { VERDICT, formatDateTime, formatClock } from '../lib/format';
import { API_BASE } from '../config';
import '../css/SessionHistory.css';

export default function SessionHistory({ onOpenReport, onNewSession }) {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(false);
  const [loadingSessionId, setLoadingSessionId] = useState(null);
  const [reportError, setReportError] = useState(null);

  const fetchSessions = useCallback(async () => {
    setLoading(true);
    setLoadError(false);
    try {
      const res = await fetch(`${API_BASE}/api/sessions?limit=50`);
      if (!res.ok) throw new Error('Failed to fetch');
      const data = await res.json();
      setSessions(data.sessions || []);
    } catch {
      setLoadError(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  const handleViewReport = async (sessionId) => {
    if (loadingSessionId) return;
    setLoadingSessionId(sessionId);
    setReportError(null);
    try {
      const res = await fetch(`${API_BASE}/api/session/${sessionId}/report`);
      if (res.status === 404) {
        setReportError('This report is no longer available.');
        return;
      }
      if (!res.ok) {
        setReportError('Could not load report.');
        return;
      }
      const report = await res.json();
      if (onOpenReport) {
        onOpenReport(report);
      }
    } catch {
      setReportError('Could not load report.');
    } finally {
      setLoadingSessionId(null);
    }
  };

  return (
    <div className="history-screen">
      <header className="history-header">
        <div className="history-header-titles">
          <h1>History</h1>
          <p>Past sessions saved on this machine.</p>
        </div>
        <div className="history-header-actions">
          <Button variant="primary" icon={Plus} onClick={onNewSession}>
            New session
          </Button>
        </div>
      </header>

      {reportError && (
        <Alert tone="danger">
          {reportError}
        </Alert>
      )}

      {loadError && (
        <Alert
          tone="danger"
          title="Could not load sessions."
          action={
            <Button variant="secondary" size="sm" onClick={fetchSessions}>
              Try again
            </Button>
          }
        />
      )}

      <div className="history-card">
        {loading ? (
          <div className="history-loading">
            <Spinner />
            <span>Loading sessions…</span>
          </div>
        ) : sessions.length === 0 ? (
          <div className="history-empty">
            <HistoryIcon size={20} className="history-empty-icon" aria-hidden="true" />
            <p>No sessions yet.</p>
            <Button variant="primary" icon={Plus} onClick={onNewSession}>
              New session
            </Button>
          </div>
        ) : (
          <div className="history-table-wrap">
            <table className="history-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Duration</th>
                  <th>Identity</th>
                  <th>Attention</th>
                  <th>Session ID</th>
                  <th aria-label="Actions" />
                </tr>
              </thead>
              <tbody>
                {sessions.map((s) => {
                  const faceV = VERDICT[s.face_verdict] || { label: s.face_verdict, tone: 'neutral' };
                  const gazeV = VERDICT[s.gaze_verdict] || { label: s.gaze_verdict, tone: 'neutral' };
                  const isCurrentLoading = loadingSessionId === s.session_id;

                  return (
                    <tr
                      key={s.session_id}
                      onClick={() => handleViewReport(s.session_id)}
                    >
                      <td>{formatDateTime(s.created_at)}</td>
                      <td className="num">{formatClock(s.duration_seconds)}</td>
                      <td>
                        <Badge tone={faceV.tone}>{faceV.label}</Badge>
                      </td>
                      <td>
                        <Badge tone={gazeV.tone}>{gazeV.label}</Badge>
                      </td>
                      <td>
                        <span className="mono muted">{s.session_id}</span>
                      </td>
                      <td className="history-action-cell">
                        <Button
                          variant="secondary"
                          size="sm"
                          disabled={isCurrentLoading}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleViewReport(s.session_id);
                          }}
                        >
                          {isCurrentLoading ? <Spinner /> : 'View'}
                        </Button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
