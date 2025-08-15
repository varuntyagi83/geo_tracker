from __future__ import annotations
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session
from sqlalchemy import create_engine, ForeignKey, String, Integer, Float, JSON, Date, Text, DateTime, func
from pydantic import BaseModel
from typing import Optional
from .config import SETTINGS

class Base(DeclarativeBase):
    pass

class Prompt(Base):
    __tablename__ = "prompts"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    prompt_id: Mapped[str] = mapped_column(String(64), index=True)
    text: Mapped[str] = mapped_column(Text)
    topic: Mapped[str] = mapped_column(String(64))
    intent: Mapped[str] = mapped_column(String(64))
    language: Mapped[str] = mapped_column(String(16))
    market: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())

class Run(Base):
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    panel_version: Mapped[str] = mapped_column(String(64))
    comments: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())

class ModelCall(Base):
    __tablename__ = "model_calls"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    prompt_ref: Mapped[int] = mapped_column(ForeignKey("prompts.id"))
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[Optional[str]] = mapped_column(String(64))
    temperature: Mapped[Optional[float]] = mapped_column(Float)
    top_p: Mapped[Optional[float]] = mapped_column(Float)
    max_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    is_mock: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="ok")
    refusal_flag: Mapped[int] = mapped_column(Integer, default=0)
    raw_answer: Mapped[str] = mapped_column(Text)
    citations_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())

class ExtractedMention(Base):
    __tablename__ = "extracted_mentions"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    call_id: Mapped[int] = mapped_column(ForeignKey("model_calls.id"), index=True)
    brand: Mapped[str] = mapped_column(String(64), index=True)
    first_position: Mapped[int] = mapped_column(Integer)
    mention_count: Mapped[int] = mapped_column(Integer)
    in_final_recommendation: Mapped[int] = mapped_column(Integer, default=0)

class Sentiment(Base):
    __tablename__ = "sentiment"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    call_id: Mapped[int] = mapped_column(ForeignKey("model_calls.id"), index=True)
    sentiment_score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    rater_model: Mapped[str] = mapped_column(String(64))
    rationale: Mapped[str] = mapped_column(Text)

class Grounding(Base):
    __tablename__ = "grounding"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    call_id: Mapped[int] = mapped_column(ForeignKey("model_calls.id"), index=True)
    source_count: Mapped[int] = mapped_column(Integer)
    credible_source_count: Mapped[int] = mapped_column(Integer)
    credible_fraction: Mapped[float] = mapped_column(Float)
    credible_domains_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

# daily metrics table
class MetricsDaily(Base):
    __tablename__ = "metrics_daily"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    date: Mapped[Date] = mapped_column(Date, index=True)
    provider: Mapped[str] = mapped_column(String(32), index=True)
    model: Mapped[str] = mapped_column(String(64), index=True)
    market: Mapped[str] = mapped_column(String(16), index=True)
    topic: Mapped[str] = mapped_column(String(64), index=True)
    presence_rate: Mapped[float] = mapped_column(Float)
    share_of_voice: Mapped[float] = mapped_column(Float)
    avg_sentiment: Mapped[float] = mapped_column(Float)
    grounding_quality: Mapped[float] = mapped_column(Float)
    placement_quality: Mapped[float] = mapped_column(Float)
    refusal_rate: Mapped[float] = mapped_column(Float)
    hallucination_rate: Mapped[float] = mapped_column(Float)
    geo_score: Mapped[float] = mapped_column(Float)

_engine = create_engine(SETTINGS.database_url, future=True)

def init_db() -> None:
    Base.metadata.create_all(_engine)
    # Lightweight migration for SQLite: add is_mock if missing
    with _engine.begin() as conn:
        try:
            cols = conn.exec_driver_sql("PRAGMA table_info(model_calls)").fetchall()
            colnames = {row[1] for row in cols}  # PRAGMA columns: cid, name, type, ...
            if "is_mock" not in colnames:
                conn.exec_driver_sql("ALTER TABLE model_calls ADD COLUMN is_mock INTEGER DEFAULT 0")
        except Exception:
            # Best-effort migration; ignore if not SQLite or already present
            pass

def get_session() -> Session:
    return Session(_engine)
