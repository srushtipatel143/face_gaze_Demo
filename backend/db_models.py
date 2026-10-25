"""
PostgreSQL Database Models and Persistence Layer
Stores reference face data, verification session results, and failure proof images.
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    Text,
    DateTime,
    JSON,
    ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

from pathlib import Path
from sqlalchemy.engine import URL

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

    # Preferred: constructed URL with proper URL escaping
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

    # Direct env DATABASE_URL if provided
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

    # Relationship to failure proof snapshots
    failures = relationship("VerificationFailure", back_populates="session", cascade="all, delete-orphan")


class VerificationFailure(Base):
    """
    Proof of Failure Snapshot
    Captures the visual evidence image and metadata when a violation occurs:
    - Different person detected (NO_MATCH)
    - Multiple faces detected
    - No face detected
    - Sustained gaze diversion (LOOKING_AWAY)
    """
    __tablename__ = "verification_failures"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("verification_sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    t_s = Column(Float, nullable=True)  # Timestamp offset in seconds into the session

    failure_type = Column(String(64), nullable=False)  # DIFFERENT_PERSON, MULTIPLE_FACES, NO_FACE, LOOKING_AWAY
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
            # Test connection
            with temp_engine.connect() as conn:
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
    if SessionLocal is None:
        init_db()
    if SessionLocal is None:
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
        # Convert numpy array to list if needed
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


def save_session_results(session_id: str, summary: Dict[str, Any]) -> bool:
    """Update PostgreSQL session with final verdicts, compliance metrics, and intervals."""
    db = get_db_session()
    if not db:
        return False
    try:
        sess_obj = db.query(VerificationSession).filter_by(session_id=session_id).first()
        if not sess_obj:
            sess_obj = VerificationSession(session_id=session_id)
            db.add(sess_obj)

        sess_obj.status = summary.get("status", "completed")
        sess_obj.duration_seconds = summary.get("duration_seconds")
        sess_obj.total_frames = summary.get("total_frames", 0)
        sess_obj.matched_frames = summary.get("matched_frames", 0)
        sess_obj.unmatched_frames = summary.get("unmatched_frames", 0)
        sess_obj.no_face_frames = summary.get("no_face_frames", 0)
        sess_obj.multiple_faces_frames = summary.get("multiple_faces_frames", 0)
        sess_obj.match_percentage = summary.get("match_percentage", 0.0)
        sess_obj.looking_away_percentage = summary.get("looking_away_percentage", 0.0)

        # Verdicts
        face_v = summary.get("face", {})
        gaze_v = summary.get("gaze", {})
        sess_obj.face_verdict = face_v.get("verdict") if isinstance(face_v, dict) else str(face_v)
        sess_obj.gaze_verdict = gaze_v.get("verdict") if isinstance(gaze_v, dict) else str(gaze_v)
        sess_obj.overall_verdict = summary.get("overall_verdict")

        reasons = []
        if isinstance(face_v, dict) and face_v.get("reasons"):
            reasons.extend(face_v["reasons"])
        if isinstance(gaze_v, dict) and gaze_v.get("reasons"):
            reasons.extend(gaze_v["reasons"])
        sess_obj.verdict_reasons = reasons

        sess_obj.metrics = summary.get("metrics")
        sess_obj.intervals = summary.get("intervals")

        # Save any interval failure thumbnails as failure proofs if not already recorded
        for iv in summary.get("intervals", []):
            thumb = iv.get("thumbnail")
            if thumb:
                iv_type = iv.get("type", "ANOMALY").upper()
                iv_details = iv.get("details", "")
                save_failure_proof(
                    session_id=session_id,
                    failure_type=iv_type,
                    reason=iv_details,
                    proof_image=thumb,
                    t_s=iv.get("start_t")
                )

        db.commit()
        return True
    except Exception as e:
        logger.error("Failed to save session results in DB: %s", e)
        db.rollback()
        return False
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
