"""
PostgreSQL Database Models and Persistence Layer
Stores reference face data, verification session results, and failure proof images.
"""

from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    create_engine,
)
from sqlalchemy.engine import URL
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

import config

# Load environment variables from backend/.env
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
else:
    load_dotenv()


logger = logging.getLogger("face_gaze.db")

engine = None
SessionLocal = None
Base = declarative_base()
_sessionlocal_warned = False


def get_candidate_db_urls() -> List[Any]:
    """Construct candidate connection URLs with proper escaping for special characters."""
    candidates = []
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgrespassword")
    port_str = os.getenv("POSTGRES_PORT", "5432")
    try:
        port = int(port_str)
    except Exception:
        port = 5432
    dbname = os.getenv("POSTGRES_DB", "face_gaze_db")
    primary_host = os.getenv("POSTGRES_HOST", "face_gaze_postgres")

    for host in [primary_host, "face_gaze_postgres", "localhost", "127.0.0.1"]:
        if host and host not in [c.host for c in candidates if hasattr(c, "host")]:
            try:
                candidates.append(
                    URL.create(
                        drivername="postgresql+psycopg2",
                        username=user,
                        password=password,
                        host=host,
                        port=port,
                        database=dbname
                    )
                )
            except Exception:
                pass

    raw_url = os.getenv("DATABASE_URL")
    if raw_url:
        if not raw_url.startswith("postgresql+psycopg2://") and raw_url.startswith("postgresql://"):
            candidates.append(raw_url.replace("postgresql://", "postgresql+psycopg2://", 1))
        else:
            candidates.append(raw_url)

    return candidates


class VerificationSession(Base):
    """
    Persisted Verification Session
    Stores reference photo, reference biometric embedding, quality score,
    and final audit metrics/verdicts.
    """
    __tablename__ = "verification_sessions"

    session_id = Column(String(64), primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String(32), default="initialized", nullable=False)

    # Reference biometric data
    reference_image = Column(Text, nullable=True)  # Base64 data URL
    reference_embedding = Column(JSON, nullable=True)  # 128-D float vector as JSON array
    quality_score = Column(Float, nullable=True)
    reference_metadata = Column(JSON, nullable=True)

    # Live verification timings
    start_time = Column(Float, nullable=True)
    end_time = Column(Float, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    # Frame aggregates
    total_frames = Column(Integer, default=0)
    matched_frames = Column(Integer, default=0)
    unmatched_frames = Column(Integer, default=0)
    no_face_frames = Column(Integer, default=0)
    multiple_faces_frames = Column(Integer, default=0)
    match_percentage = Column(Float, default=0.0)
    looking_away_percentage = Column(Float, default=0.0)

    # Final Verdicts
    face_verdict = Column(String(32), nullable=True)  # PASS | FAIL | REVIEW
    gaze_verdict = Column(String(32), nullable=True)  # PASS | FAIL | REVIEW
    overall_verdict = Column(String(32), nullable=True)
    verdict_reasons = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    intervals = Column(JSON, nullable=True)

    # Full report and records persistence (Section 9)
    report = Column(JSON, nullable=True)
    records = Column(JSON, nullable=True)
    config_snapshot = Column(JSON, nullable=True)
    rules_version = Column(String(32), nullable=True)

    # Relationship to failure proof snapshots
    failures = relationship("VerificationFailure", back_populates="session", cascade="all, delete-orphan")


class VerificationFailure(Base):
    """
    Proof of Failure Snapshot
    Captures the visual evidence image and metadata when a violation occurs.
    """
    __tablename__ = "verification_failures"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("verification_sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    t_s = Column(Float, nullable=True)  # Timestamp offset in seconds into the session

    failure_type = Column(String(64), nullable=False)
    reason = Column(Text, nullable=True)
    similarity_score = Column(Float, nullable=True)

    # Proof of fail image (JPEG thumbnail base64 data URL)
    proof_image = Column(Text, nullable=False)

    session = relationship("VerificationSession", back_populates="failures")


def init_db() -> bool:
    """Initialize database tables. Returns True if successful, False if DB is unreachable."""
    global engine, SessionLocal
    candidates = get_candidate_db_urls()

    for url in candidates:
        if not url:
            continue
        try:
            temp_engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 3})
            with temp_engine.connect():
                pass
            Base.metadata.create_all(bind=temp_engine)
            engine = temp_engine
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            logger.info("Successfully connected to PostgreSQL database.")
            return True
        except Exception as e:
            logger.debug("Failed connecting to DB candidate %s: %s", url, e)

    logger.warning("Could not connect to PostgreSQL. Operating in resilient in-memory mode until DB is available.")
    return False


def get_db_session():
    """Context manager or generator for DB session."""
    global _sessionlocal_warned
    if SessionLocal is None:
        if not _sessionlocal_warned:
            logger.warning("Database SessionLocal is not initialized. Database operations will be skipped.")
            _sessionlocal_warned = True
        return None
    return SessionLocal()


def save_reference_session(
    session_id: str,
    reference_image: Optional[str] = None,
    reference_embedding: Any = None,
    quality_score: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Store reference photo and initial session data in PostgreSQL."""
    db = get_db_session()
    if not db:
        return False
    try:
        emb_list = None
        if reference_embedding is not None:
            if hasattr(reference_embedding, "tolist"):
                emb_list = reference_embedding.tolist()
            elif isinstance(reference_embedding, list):
                emb_list = reference_embedding

        sess_obj = db.query(VerificationSession).filter_by(session_id=session_id).first()
        if not sess_obj:
            sess_obj = VerificationSession(
                session_id=session_id,
                reference_image=reference_image,
                reference_embedding=emb_list,
                quality_score=quality_score,
                reference_metadata=metadata or {},
                status="initialized"
            )
            db.add(sess_obj)
        else:
            if reference_image:
                sess_obj.reference_image = reference_image
            if emb_list:
                sess_obj.reference_embedding = emb_list
            sess_obj.quality_score = quality_score
            sess_obj.reference_metadata = metadata or {}
        db.commit()
        return True
    except Exception as e:
        logger.error("Failed to save reference session in DB: %s", e)
        db.rollback()
        return False
    finally:
        db.close()


def save_failure_proof(
    session_id: str,
    failure_type: str,
    reason: Optional[str],
    proof_image: str,
    similarity_score: Optional[float] = None,
    t_s: Optional[float] = None
) -> bool:
    """Persist a proof image of a verification failure."""
    if not proof_image:
        return False

    db = get_db_session()
    if not db:
        return False
    try:
        fail_record = VerificationFailure(
            session_id=session_id,
            failure_type=failure_type,
            reason=reason,
            similarity_score=similarity_score,
            proof_image=proof_image,
            t_s=t_s
        )
        db.add(fail_record)
        db.commit()
        return True
    except Exception as e:
        logger.error("Failed to save failure proof in DB: %s", e)
        db.rollback()
        return False
    finally:
        db.close()


def save_report(session_id: str, report: Dict[str, Any], records: List[Any]) -> bool:
    db = get_db_session()
    if not db:
        return False
    try:
        s = db.query(VerificationSession).filter_by(session_id=session_id).first()
        if not s:
            s = VerificationSession(session_id=session_id)
            db.add(s)
        m = report["metrics"]
        s.status = "completed"
        s.duration_seconds = report["duration_s"]
        s.total_frames = report["total_frames"]
        s.matched_frames = m["matched_frames"]
        s.unmatched_frames = m["unmatched_frames"]
        s.no_face_frames = m["no_face_frames"]
        s.multiple_faces_frames = m["multiple_faces_frames"]
        s.match_percentage = m["match_pct"]
        s.looking_away_percentage = m["away_pct"]
        s.face_verdict = report["face"]["verdict"]
        s.gaze_verdict = report["gaze"]["verdict"]
        s.overall_verdict = report["overall_verdict"]
        s.verdict_reasons = report["face"]["reasons"] + report["gaze"]["reasons"]
        s.metrics = m
        s.intervals = [{k: v for k, v in iv.items() if k != "thumbnail"} for iv in report["intervals"]]
        s.report = report
        s.records = [r.to_dict(include_thumbnail=False) if hasattr(r, "to_dict") else {k: v for k, v in r.items() if k != "thumbnail"} for r in records]
        s.config_snapshot = config.snapshot()
        s.rules_version = report["rules_version"]

        db.query(VerificationFailure).filter_by(session_id=session_id).delete()
        for iv in report["intervals"]:
            if iv.get("thumbnail"):
                db.add(VerificationFailure(
                    session_id=session_id, failure_type=iv["type"].upper(),
                    reason=iv["message"], proof_image=iv["thumbnail"], t_s=iv["start_s"]))
        db.commit()
        return True
    except Exception:
        logger.exception("save_report failed for %s", session_id)
        db.rollback()
        return False
    finally:
        db.close()


def list_sessions(limit: int = 50) -> List[Dict[str, Any]]:
    db = get_db_session()
    if not db:
        return []
    try:
        rows = (
            db.query(VerificationSession)
            .filter(
                VerificationSession.status == "completed",
                VerificationSession.report.isnot(None)
            )
            .order_by(VerificationSession.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "session_id": r.session_id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "duration_seconds": r.duration_seconds,
                "face_verdict": r.face_verdict,
                "gaze_verdict": r.gaze_verdict,
                "total_frames": r.total_frames,
            }
            for r in rows
        ]
    finally:
        db.close()


def get_report(session_id: str) -> Optional[Dict[str, Any]]:
    db = get_db_session()
    if not db:
        return None
    try:
        s = db.query(VerificationSession).filter_by(session_id=session_id).first()
        if not s or not s.report:
            return None
        return s.report
    finally:
        db.close()


def get_session_from_db(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve full session record from PostgreSQL."""
    db = get_db_session()
    if not db:
        return None
    try:
        sess = db.query(VerificationSession).filter_by(session_id=session_id).first()
        if not sess:
            return None
        return {
            "session_id": sess.session_id,
            "created_at": sess.created_at.isoformat() if sess.created_at else None,
            "status": sess.status,
            "quality_score": sess.quality_score,
            "has_reference_image": bool(sess.reference_image),
            "reference_image": sess.reference_image,
            "duration_seconds": sess.duration_seconds,
            "total_frames": sess.total_frames,
            "matched_frames": sess.matched_frames,
            "unmatched_frames": sess.unmatched_frames,
            "match_percentage": sess.match_percentage,
            "face_verdict": sess.face_verdict,
            "gaze_verdict": sess.gaze_verdict,
            "overall_verdict": sess.overall_verdict,
            "verdict_reasons": sess.verdict_reasons,
            "metrics": sess.metrics,
            "intervals": sess.intervals,
            "report": sess.report,
            "records": sess.records,
            "config_snapshot": sess.config_snapshot,
            "rules_version": sess.rules_version,
        }
    finally:
        db.close()


def get_session_failures(session_id: str) -> List[Dict[str, Any]]:
    """Retrieve all proof-of-failure records for a given session."""
    db = get_db_session()
    if not db:
        return []
    try:
        records = (
            db.query(VerificationFailure)
            .filter_by(session_id=session_id)
            .order_by(VerificationFailure.id.asc())
            .all()
        )
        return [
            {
                "id": r.id,
                "session_id": r.session_id,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "t_s": r.t_s,
                "failure_type": r.failure_type,
                "reason": r.reason,
                "similarity_score": r.similarity_score,
                "proof_image": r.proof_image
            }
            for r in records
        ]
    finally:
        db.close()
