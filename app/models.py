# app/models.py  –  SQLAlchemy ORM models
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import relationship

from app.database import Base


def _utcnow():
    return datetime.now(timezone.utc)


# ── Empresas ──────────────────────────────────────────────
class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    rut = Column(String, nullable=True, unique=True, index=True)
    codigo = Column(String, nullable=True, unique=True, index=True)
    is_active = Column(Boolean, nullable=False, default=True)

    users = relationship("User", back_populates="company")
    cameras = relationship("Camera", back_populates="company")


# ── Catálogo de roles ─────────────────────────────────────
class Role(Base):
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)

    users = relationship("User", back_populates="role")


# ── Usuarios ──────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, nullable=False)
    apellido = Column(String, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)  # correo
    password = Column(String, nullable=False)  # hash
    salt = Column(String, nullable=False, default="")  # salt para PBKDF2
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=False, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True, index=True)
    is_active = Column(Boolean, nullable=False, default=True)

    role = relationship("Role", back_populates="users")
    company = relationship("Company", back_populates="users")
    access_logs = relationship("AccessLog", back_populates="user", cascade="all, delete-orphan")
    camera_actions = relationship("CameraAction", back_populates="user", cascade="all, delete-orphan")
    reviewed_alerts = relationship("AlertLog", back_populates="reviewer")


# ── Cámaras registradas ──────────────────────────────────
class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True)
    serial_number = Column(String, unique=True, nullable=False, index=True)
    src = Column(String, nullable=False)
    location_description = Column(String, nullable=False, default="")
    is_active = Column(Boolean, nullable=False, default=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True, index=True)

    company = relationship("Company", back_populates="cameras")
    alert_logs = relationship("AlertLog", back_populates="camera")
    camera_actions = relationship("CameraAction", back_populates="camera")


# ── Alertas ───────────────────────────────────────────────
class AlertLog(Base):
    __tablename__ = "alert_logs"

    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    prob = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=_utcnow, nullable=False)
    status = Column(String, nullable=False, default="pending")
    evidence_path = Column(String, nullable=True)
    reviewed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    review_timestamp = Column(DateTime, nullable=True)

    camera = relationship("Camera", back_populates="alert_logs")
    reviewer = relationship("User", back_populates="reviewed_alerts", foreign_keys=[reviewed_by])

    __table_args__ = (
        Index("ix_alert_ts", "timestamp"),
        Index("ix_alert_camera_status", "camera_id", "status"),
    )


# ── Accesos ───────────────────────────────────────────────
class AccessLog(Base):
    __tablename__ = "access_logs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp = Column(DateTime, default=_utcnow, nullable=False)

    user = relationship("User", back_populates="access_logs")

    __table_args__ = (
        Index("ix_access_ts", "timestamp"),
    )


# ── Acciones sobre cámaras ────────────────────────────────
class CameraAction(Base):
    __tablename__ = "camera_actions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    action = Column(String, nullable=False)
    timestamp = Column(DateTime, default=_utcnow, nullable=False)

    user = relationship("User", back_populates="camera_actions")
    camera = relationship("Camera", back_populates="camera_actions")

    __table_args__ = (
        Index("ix_camaction_ts", "timestamp"),
    )
