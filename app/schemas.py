# app/schemas.py  –  Pydantic schemas (request / response)
from typing import Literal, Optional

from pydantic import BaseModel


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    nombre: str
    apellido: str
    email: str
    password: str
    charge: str
    codigo: Optional[str] = None
    company_id: Optional[int] = None
    empresa_nombre: Optional[str] = None


class CameraConfig(BaseModel):
    serial_number: str
    src: str
    location_description: Optional[str] = ""
    is_active: Optional[bool] = True
    company_id: Optional[int] = None


class CompanyCreate(BaseModel):
    name: str
    rut: Optional[str] = None


class AlertDecision(BaseModel):
    serial_number: str
    prob: float
    timestamp: float
    accepted: Optional[bool] = None
    status: Optional[Literal["pending", "true_positive", "false_positive"]] = None
    evidence_path: Optional[str] = None
    reviewer_email: Optional[str] = None


class CameraStreamStatus(BaseModel):
    status: Literal["online", "offline", "error"]
    detail: Optional[str] = None
