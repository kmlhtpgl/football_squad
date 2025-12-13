from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Boolean,
    ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship


DB_URL = "sqlite:///./football.db"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()


class Schedule(Base):
    __tablename__ = "schedules"

    id = Column(Integer, primary_key=True)
    title = Column(String, default="Weekly Match")

    # 0=Mon ... 6=Sun
    weekday = Column(Integer, default=4)  # Friday
    time_text = Column(String, default="20:00")
    location = Column(String, default="Pitch")

    default_cost = Column(Integer, default=70)  # pounds
    max_players = Column(Integer, default=12)
    waitlist_enabled = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    matches = relationship("Match", back_populates="schedule", cascade="all, delete-orphan")


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    schedule_id = Column(Integer, ForeignKey("schedules.id"))

    # week_start is the Monday of that week (YYYY-MM-DD)
    week_start = Column(String, nullable=False)

    start_time_text = Column(String, default="Fri 20:00")
    location = Column(String, default="Pitch")
    total_cost = Column(Integer, default=70)  # pounds

    is_open = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    schedule = relationship("Schedule", back_populates="matches")
    signups = relationship("Signup", back_populates="match", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("schedule_id", "week_start", name="uq_schedule_week"),
    )


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    display_name = Column(String, nullable=False, unique=True)
    pin_hash = Column(String, nullable=False)

    # NEW: for team balancing + lineup positions
    level = Column(Integer, default=5)      # 1..10
    position = Column(String, default="")   # GK, CB, CM, LW, ST, etc.

    created_at = Column(DateTime, default=datetime.utcnow)

    payments = relationship("Payment", back_populates="user", cascade="all, delete-orphan")


class Signup(Base):
    __tablename__ = "signups"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # keep the name for display + legacy
    name = Column(String, nullable=False)

    # device token (legacy / optional now, but kept)
    token = Column(String, nullable=False)

    # confirmed | waitlist
    status = Column(String, default="confirmed")

    created_at = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match", back_populates="signups")

    __table_args__ = (
        UniqueConstraint("match_id", "name", name="uq_match_name"),
    )


class Payment(Base):
    __tablename__ = "payments"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))

    # store in pennies (pence)
    amount_pence = Column(Integer, nullable=False)

    paid_at = Column(DateTime, default=datetime.utcnow)
    note = Column(String, default="")

    user = relationship("User", back_populates="payments")


def init_db():
    Base.metadata.create_all(bind=engine)

    # ensure we always have one default schedule
    db = SessionLocal()
    try:
        sched = db.query(Schedule).order_by(Schedule.id.asc()).first()
        if not sched:
            db.add(Schedule())
            db.commit()
    finally:
        db.close()
