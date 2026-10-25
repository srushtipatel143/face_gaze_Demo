export function formatClock(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return '--:--';
  const s = Math.max(0, Math.floor(seconds));
  return String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0');
}

export function formatSeconds(seconds) {
  if (seconds === null || seconds === undefined) return '—';
  if (seconds < 60) return seconds.toFixed(1) + ' s';
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m + ' min ' + String(s).padStart(2, '0') + ' s';
}

export function formatPct(value) {
  return value === null || value === undefined ? '—' : Math.round(value) + '%';
}

// Backend stores naive UTC timestamps without "Z"; treat them as UTC.
export function parseUtc(iso) {
  if (!iso) return null;
  return new Date(/Z$|[+-]\d\d:\d\d$/.test(iso) ? iso : iso + 'Z');
}

export function formatDateTime(iso) {
  const d = parseUtc(iso);
  return d ? d.toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }) : '—';
}

export const VERDICT = {
  PASS: { label: 'Pass', tone: 'success' },
  REVIEW: { label: 'Needs review', tone: 'warning' },
  FAIL: { label: 'Fail', tone: 'danger' },
};

export const IDENTITY_STATUS = {
  VERIFIED: { label: 'Verified', tone: 'success' },
  NOT_MATCHED: { label: 'Not matched', tone: 'danger' },
  NO_FACE: { label: 'No face', tone: 'warning' },
  MULTIPLE_FACES: { label: 'Another person', tone: 'danger' },
  UNCERTAIN: { label: 'Face turned away', tone: 'warning' },
  CHECKING: { label: 'Checking', tone: 'neutral' },
};

export const ATTENTION_STATUS = {
  ON_SCREEN: { label: 'On screen', tone: 'success' },
  AWAY: { label: 'Looking away', tone: 'warning' },
  UNKNOWN: { label: 'Not measured', tone: 'neutral' },
  CALIBRATING: { label: 'Calibrating', tone: 'info' },
};

export const INTERVAL_TYPE = {
  looking_away: { label: 'Looked away', tone: 'warning', track: 'attention' },
  no_face: { label: 'Face not visible', tone: 'neutral', track: 'identity' },
  multiple_faces: { label: 'Another person', tone: 'danger', track: 'identity' },
  different_person: { label: 'Different person', tone: 'danger', track: 'identity' },
};

// Sentences for verdict reasons. See Appendix B.
export function reasonText(code, m) {
  switch (code) {
    case 'different_person': return 'A different person was on camera for ' + formatSeconds(m.different_person_s) + '.';
    case 'multiple_faces': return 'Another person was visible for ' + formatSeconds(m.multiple_faces_s) + '.';
    case 'no_face': return 'The face was not visible for ' + formatPct(m.no_face_pct) + ' of the session.';
    case 'identity_unknown': return 'The face was turned too far to check identity for ' + formatPct(m.identity_unknown_pct) + ' of the time.';
    case 'static_face': return 'No blinking or head movement for a long time. The camera may be showing a photo.';
    case 'no_data': return 'No frames were analysed.';
    case 'calibration_incomplete': return 'Calibration did not finish, so attention could not be measured.';
    case 'no_stable_baseline': return 'Calibration was unreliable, so attention needs a manual check.';
    case 'long_away': return m.long_away_intervals + ' look-away(s) lasted ' + m.long_away_threshold_s + ' s or longer.';
    case 'looking_away': return 'Looked away ' + formatPct(m.away_pct) + ' of the time.';
    case 'gaze_unknown': return 'Gaze could not be measured ' + formatPct(m.gaze_unknown_pct) + ' of the time.';
    default: return code;
  }
}

export function passText(kind, m) {
  return kind === 'face'
    ? 'Same person throughout. No one else appeared.'
    : 'Looked at the screen ' + formatPct(m.time_on_screen_pct) + ' of the time.';
}
