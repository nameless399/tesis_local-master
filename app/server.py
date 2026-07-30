# uvicorn app.server:app --host 0.0.0.0 --port 8000
#
# Backend ligero (Render/Supabase):
#   - Auth, usuarios, camaras (metadatos), alertas, admin
#   - Sirve el frontend estatico
#   - NO corre inferencia (eso vive en app/server_inference.py, desplegado en RunPod)
#
# El navegador abre el WebSocket directamente contra la URL de RunPod,
# configurada via INFERENCE_BASE_URL y expuesta al frontend en /config.js.

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.crud import (
    add_access_log,
    add_camera_action,
    create_company,
    create_user,
    ensure_roles,
    get_company_by_codigo,
    get_company_by_id,
    get_role_by_name,
    get_user_by_email,
)
from app.database import Base, SessionLocal, engine
from app.models import AccessLog, AlertLog, Camera, CameraAction, Company, User
from app.schemas import AlertDecision, CameraConfig, CameraStreamStatus, CompanyCreate, LoginRequest, RegisterRequest
from app.utils import verify_password

load_dotenv()
log = logging.getLogger("server")

# ==========================================================
# CONFIGURACION
# ==========================================================
ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = (ROOT_DIR / "static").resolve()

BOSS_CODE = os.getenv("BOSS_CODE", "20261")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# URL publica del servidor de inferencia (RunPod). Ej: https://abc-8000.proxy.runpod.net
INFERENCE_BASE_URL = os.getenv("INFERENCE_BASE_URL", "").rstrip("/")

LOG_RETENTION_DAYS = 7

FAILED_ATTEMPTS: dict[str, dict] = {}
MAX_ATTEMPTS = 3
BLOCK_TIME_MINUTES = 5

ROLE_OPERATOR = "operador"
ROLE_BOSS = "jefe"
DEFAULT_ROLES = [ROLE_OPERATOR, ROLE_BOSS]
ALERT_STATUSES = {"pending", "true_positive", "false_positive"}


# ==========================================================
# LIMPIEZA PERIODICA DE LOGS
# ==========================================================
def _cleanup_logs():
    while True:
        try:
            db = SessionLocal()
            limit = datetime.now(timezone.utc) - timedelta(days=LOG_RETENTION_DAYS)
            db.query(AccessLog).filter(AccessLog.timestamp < limit).delete()
            db.query(CameraAction).filter(CameraAction.timestamp < limit).delete()
            db.commit()
            db.close()
        except Exception as e:
            log.warning("Error al borrar logs: %s", e)
        time.sleep(24 * 3600)


threading.Thread(target=_cleanup_logs, daemon=True).start()


# ==========================================================
# BASE DE DATOS
# ==========================================================
Base.metadata.create_all(bind=engine)


def _seed_default_roles():
    db = SessionLocal()
    try:
        ensure_roles(db, DEFAULT_ROLES)
    except Exception as e:
        log.warning("No se pudo sembrar roles base: %s", e)
    finally:
        db.close()


_seed_default_roles()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==========================================================
# HELPERS
# ==========================================================
def _role_name(user: User | None) -> str:
    if user and user.role:
        return user.role.name
    return ROLE_OPERATOR


def _epoch_or_none(dt_value: datetime | None) -> float | None:
    if dt_value is None:
        return None
    return dt_value.timestamp()


def _resolve_status(decision: AlertDecision) -> str:
    if decision.accepted is True:
        return "true_positive"
    if decision.accepted is False:
        return "false_positive"
    if decision.status in ALERT_STATUSES:
        return decision.status
    return "pending"


def _resolve_alert_timestamp(ts: float) -> datetime:
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return datetime.now(timezone.utc)


def _resolve_user_from_request(
    db: Session,
    request: Request,
    explicit_email: str | None = None,
) -> User | None:
    header_user_id = (request.headers.get("X-User-Id") or "").strip()
    if header_user_id.isdigit():
        by_id = db.query(User).filter(User.id == int(header_user_id)).first()
        if by_id:
            return by_id

    email = (explicit_email or request.headers.get("X-User-Email") or "").strip().lower()
    if email:
        return get_user_by_email(db, email)

    return None


# ==========================================================
# APP
# ==========================================================
app = FastAPI(title="Sistema de Videovigilancia - Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -- Paginas ------------------------------------------------
@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "login.html")


@app.get("/dashboard")
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/admin")
def admin_page():
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/login_fail")
def login_fail_page():
    return FileResponse(STATIC_DIR / "login_fail.html")


# -- Configuracion inyectada al frontend -------------------
@app.get("/config.js")
def frontend_config():
    """
    Expone al navegador:
      - API_BASE_URL       -> este backend (Render, auth + DB + static)
      - INFERENCE_BASE_URL -> servidor de inferencia (RunPod, /ws, /overlay)
    """
    cfg = {
        "API_BASE_URL": "",
        "INFERENCE_BASE_URL": INFERENCE_BASE_URL,
    }
    js = f"""
window.SICHER_CONFIG = {json.dumps(cfg)};

window.sicherApiUrl = function(path) {{
  const base = (window.SICHER_CONFIG && window.SICHER_CONFIG.API_BASE_URL) || "";
  return base ? base.replace(/\\/$/, "") + path : path;
}};

window.sicherInferenceUrl = function(path) {{
  const base = (window.SICHER_CONFIG && window.SICHER_CONFIG.INFERENCE_BASE_URL) || "";
  return base ? base.replace(/\\/$/, "") + path : path;
}};

window.sicherWsUrl = function(path) {{
  const base = (window.SICHER_CONFIG && window.SICHER_CONFIG.INFERENCE_BASE_URL) || "";
  if (!base) {{
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    return `${{proto}}://${{window.location.host}}${{path}}`;
  }}
  const url = new URL(base);
  const proto = url.protocol === "https:" ? "wss:" : "ws:";
  return `${{proto}}//${{url.host}}${{path}}`;
}};
""".strip()
    return Response(content=js, media_type="application/javascript")


# -- Auth ---------------------------------------------------
@app.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    email = req.email.lower().strip()
    now = datetime.now(timezone.utc)

    info = FAILED_ATTEMPTS.get(email, {"count": 0, "blocked_until": None})
    if info["blocked_until"] and now < info["blocked_until"]:
        return JSONResponse({"error": "Bloqueado", "blocked": True, "blocked_until": info["blocked_until"].isoformat()}, status_code=403)

    user = get_user_by_email(db, email)
    if not user or not verify_password(req.password, user.salt, user.password):
        info["count"] += 1
        if info["count"] >= MAX_ATTEMPTS:
            info["blocked_until"] = now + timedelta(minutes=BLOCK_TIME_MINUTES)
        FAILED_ATTEMPTS[email] = info
        return JSONResponse({"error": "Credenciales invalidas"}, status_code=401)

    if not user.is_active:
        return JSONResponse({"error": "Usuario inactivo"}, status_code=403)

    FAILED_ATTEMPTS[email] = {"count": 0, "blocked_until": None}
    add_access_log(db, user.id)

    role_name = _role_name(user)

    return {
        "access_token": "token_valido",
        "user_id": user.id,
        "email": email,
        "nombre": user.nombre,
        "apellido": user.apellido,
        "role": role_name,
        "charge": role_name,  # compatibilidad con frontend antiguo
        "is_active": user.is_active,
    }


@app.post("/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    email = req.email.lower().strip()
    role_name = (req.charge or "").strip().lower()

    if role_name not in DEFAULT_ROLES:
        return JSONResponse({"error": "Cargo invalido"}, status_code=400)

    if get_user_by_email(db, email):
        return JSONResponse({"error": "Usuario existente"}, status_code=409)

    role = get_role_by_name(db, role_name)
    if role is None:
        ensure_roles(db, DEFAULT_ROLES)
        role = get_role_by_name(db, role_name)

    if role is None:
        return JSONResponse({"error": "No se pudo resolver el rol"}, status_code=500)

    company_id = req.company_id

    # Jefe: create company with code chosen by jefe
    if role_name == ROLE_BOSS:
        empresa_nombre = (req.empresa_nombre or "").strip()
        codigo = (req.codigo or "").strip()
        if not empresa_nombre:
            return JSONResponse({"error": "Nombre de empresa obligatorio"}, status_code=400)
        if not codigo:
            return JSONResponse({"error": "Debe crear un codigo para su empresa"}, status_code=400)
        if get_company_by_codigo(db, codigo):
            return JSONResponse({"error": "Ese codigo ya esta en uso, elija otro"}, status_code=409)
        company = create_company(db, empresa_nombre, codigo=codigo)
        company_id = company.id

    # Operador: find company by codigo
    if role_name == ROLE_OPERATOR:
        codigo = (req.codigo or "").strip()
        if not codigo:
            return JSONResponse({"error": "Codigo de empresa obligatorio"}, status_code=400)
        company = get_company_by_codigo(db, codigo)
        if not company:
            return JSONResponse({"error": "Codigo de empresa invalido"}, status_code=404)
        company_id = company.id

    create_user(db, req.nombre, req.apellido, email, req.password, role.id, company_id)
    return {"ok": True}


# -- Admin endpoints ---------------------------------------
@app.get("/admin/users")
def admin_users(request: Request, db: Session = Depends(get_db)):
    requester = _resolve_user_from_request(db, request)
    if requester is None:
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    if _role_name(requester) != ROLE_BOSS:
        return JSONResponse({"error": "No autorizado"}, status_code=403)

    # Un jefe solo puede ver usuarios de su propia empresa.
    users_query = db.query(User)
    if requester.company_id is None:
        users_query = users_query.filter(User.id == requester.id)
    else:
        users_query = users_query.filter(User.company_id == requester.company_id)

    users = users_query.order_by(User.id.asc()).all()
    return [
        {
            "id": u.id,
            "nombre": u.nombre,
            "apellido": u.apellido,
            "email": u.username,
            "role": _role_name(u),
            "charge": _role_name(u),
            "is_active": u.is_active,
            "company_id": u.company_id,
            "company_name": u.company.name if u.company else None,
        }
        for u in users
    ]


@app.get("/admin/logs/access")
def admin_access_logs(request: Request, db: Session = Depends(get_db)):
    requester = _resolve_user_from_request(db, request)
    if requester is None:
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    if _role_name(requester) != ROLE_BOSS:
        return JSONResponse({"error": "No autorizado"}, status_code=403)
    if requester.company_id is None:
        return []

    logs = (
        db.query(AccessLog)
        .join(User, AccessLog.user_id == User.id)
        .filter(User.company_id == requester.company_id)
        .order_by(AccessLog.timestamp.desc())
        .all()
    )
    return [
        {
            "usuario": f"{log.user.nombre} {log.user.apellido}" if log.user else "-",
            "cargo": _role_name(log.user) if log.user else "-",
            "timestamp": log.timestamp.timestamp(),
        }
        for log in logs
    ]


@app.get("/admin/logs/cameras")
def admin_camera_logs(request: Request, db: Session = Depends(get_db)):
    requester = _resolve_user_from_request(db, request)
    if requester is None:
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    if _role_name(requester) != ROLE_BOSS:
        return JSONResponse({"error": "No autorizado"}, status_code=403)
    if requester.company_id is None:
        return []

    logs = (
        db.query(CameraAction)
        .join(Camera, CameraAction.camera_id == Camera.id)
        .filter(Camera.company_id == requester.company_id)
        .order_by(CameraAction.timestamp.desc())
        .all()
    )
    return [
        {
            "usuario": f"{log.user.nombre} {log.user.apellido}" if log.user else "-",
            "camera_id": log.camera_id,
            "serial_number": log.camera.serial_number if log.camera else "-",
            "action": log.action,
            "timestamp": log.timestamp.timestamp(),
        }
        for log in logs
    ]


@app.get("/admin/logs/alerts")
def admin_alert_logs(request: Request, db: Session = Depends(get_db)):
    requester = _resolve_user_from_request(db, request)
    if requester is None:
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    if _role_name(requester) != ROLE_BOSS:
        return JSONResponse({"error": "No autorizado"}, status_code=403)
    if requester.company_id is None:
        return []

    logs = (
        db.query(AlertLog)
        .join(Camera, AlertLog.camera_id == Camera.id)
        .filter(Camera.company_id == requester.company_id)
        .order_by(AlertLog.timestamp.desc())
        .all()
    )
    return [
        {
            "id": l.id,
            "camera_id": l.camera_id,
            "serial_number": l.camera.serial_number if l.camera else "-",
            "prob": l.prob,
            "status": l.status,
            "evidence_path": l.evidence_path,
            "timestamp": l.timestamp.timestamp(),
            "reviewed_by": l.reviewed_by,
            "reviewed_by_name": (
                f"{l.reviewer.nombre} {l.reviewer.apellido}" if l.reviewer else "-"
            ),
            "reviewed_by_email": l.reviewer.username if l.reviewer else None,
            "review_timestamp": _epoch_or_none(l.review_timestamp),
        }
        for l in logs
    ]


@app.delete("/admin/logs/access/clear")
def clear_access_logs(request: Request, db: Session = Depends(get_db)):
    requester = _resolve_user_from_request(db, request)
    if requester is None:
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    if _role_name(requester) != ROLE_BOSS:
        return JSONResponse({"error": "No autorizado"}, status_code=403)
    if requester.company_id is None:
        return JSONResponse({"ok": True, "msg": "Sin datos para borrar"}, status_code=200)

    user_ids = db.query(User.id).filter(User.company_id == requester.company_id)
    db.query(AccessLog).filter(AccessLog.user_id.in_(user_ids)).delete(synchronize_session=False)
    db.commit()
    return {"ok": True, "msg": "Historial de accesos eliminado"}


@app.delete("/admin/logs/actions/clear")
def clear_action_logs(request: Request, db: Session = Depends(get_db)):
    requester = _resolve_user_from_request(db, request)
    if requester is None:
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    if _role_name(requester) != ROLE_BOSS:
        return JSONResponse({"error": "No autorizado"}, status_code=403)
    if requester.company_id is None:
        return JSONResponse({"ok": True, "msg": "Sin datos para borrar"}, status_code=200)

    camera_ids = db.query(Camera.id).filter(Camera.company_id == requester.company_id)
    db.query(CameraAction).filter(CameraAction.camera_id.in_(camera_ids)).delete(synchronize_session=False)
    db.commit()
    return {"ok": True, "msg": "Historial de acciones eliminado"}


@app.delete("/admin/logs/alerts/clear")
def clear_alert_logs(request: Request, db: Session = Depends(get_db)):
    requester = _resolve_user_from_request(db, request)
    if requester is None:
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    if _role_name(requester) != ROLE_BOSS:
        return JSONResponse({"error": "No autorizado"}, status_code=403)
    if requester.company_id is None:
        return JSONResponse({"ok": True, "msg": "Sin datos para borrar"}, status_code=200)

    camera_ids = db.query(Camera.id).filter(Camera.company_id == requester.company_id)
    db.query(AlertLog).filter(AlertLog.camera_id.in_(camera_ids)).delete(synchronize_session=False)
    db.commit()
    return {"ok": True, "msg": "Alertas eliminadas"}


# ==========================================================
# EMPRESAS
# ==========================================================
@app.get("/companies")
def list_companies(request: Request, db: Session = Depends(get_db)):
    requester = _resolve_user_from_request(db, request)
    if requester is None:
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    if _role_name(requester) != ROLE_BOSS:
        return JSONResponse({"error": "No autorizado"}, status_code=403)
    if requester.company_id is None:
        return []

    rows = db.query(Company).filter(Company.id == requester.company_id).order_by(Company.id.asc()).all()
    return [
        {
            "id": c.id,
            "name": c.name,
            "rut": c.rut,
            "codigo": c.codigo,
            "is_active": c.is_active,
            "user_count": len(c.users),
            "camera_count": len(c.cameras),
        }
        for c in rows
    ]


@app.post("/companies")
def add_company(payload: CompanyCreate, request: Request, db: Session = Depends(get_db)):
    requester = _resolve_user_from_request(db, request)
    if requester is None:
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    if _role_name(requester) != ROLE_BOSS:
        return JSONResponse({"error": "No autorizado"}, status_code=403)
    if requester.company_id is not None:
        return JSONResponse({"error": "No autorizado para crear otra empresa"}, status_code=403)

    name = payload.name.strip()
    if not name:
        return JSONResponse({"error": "Nombre obligatorio"}, status_code=400)

    rut = (payload.rut or "").strip() or None
    if rut:
        existing = db.query(Company).filter(Company.rut == rut).first()
        if existing:
            return JSONResponse({"error": "RUT ya registrado"}, status_code=409)

    company = create_company(db, name, rut)
    return {"ok": True, "company": {"id": company.id, "name": company.name, "rut": company.rut, "codigo": company.codigo}}


@app.patch("/companies/{company_id}")
def update_company(company_id: int, payload: CompanyCreate, request: Request, db: Session = Depends(get_db)):
    requester = _resolve_user_from_request(db, request)
    if requester is None:
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    if _role_name(requester) != ROLE_BOSS:
        return JSONResponse({"error": "No autorizado"}, status_code=403)
    if requester.company_id != company_id:
        return JSONResponse({"error": "No autorizado"}, status_code=403)

    company = get_company_by_id(db, company_id)
    if not company:
        return JSONResponse({"error": "Empresa no encontrada"}, status_code=404)

    name = payload.name.strip()
    if name:
        company.name = name
    rut = (payload.rut or "").strip() or None
    if rut and rut != company.rut:
        existing = db.query(Company).filter(Company.rut == rut, Company.id != company_id).first()
        if existing:
            return JSONResponse({"error": "RUT ya registrado"}, status_code=409)
        company.rut = rut

    db.commit()
    db.refresh(company)
    return {"ok": True, "company": {"id": company.id, "name": company.name, "rut": company.rut, "codigo": company.codigo}}


@app.delete("/companies/{company_id}")
def delete_company(company_id: int, request: Request, db: Session = Depends(get_db)):
    requester = _resolve_user_from_request(db, request)
    if requester is None:
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    if _role_name(requester) != ROLE_BOSS:
        return JSONResponse({"error": "No autorizado"}, status_code=403)
    if requester.company_id != company_id:
        return JSONResponse({"error": "No autorizado"}, status_code=403)

    company = get_company_by_id(db, company_id)
    if not company:
        return JSONResponse({"error": "Empresa no encontrada"}, status_code=404)

    company.is_active = False
    db.commit()
    return {"ok": True}


# ==========================================================
# CAMARAS  -  persistidas en DB (Supabase)
# ==========================================================
@app.get("/cameras")
def list_cameras(request: Request, include_inactive: bool = False, db: Session = Depends(get_db)):
    requester = _resolve_user_from_request(db, request)
    if requester is None:
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    if requester.company_id is None:
        return []

    query = db.query(Camera)
    query = query.filter(Camera.company_id == requester.company_id)
    if not include_inactive:
        query = query.filter(Camera.is_active.is_(True))

    rows = query.order_by(Camera.id.asc()).all()
    return [
        {
            "id": r.id,
            "serial_number": r.serial_number,
            "cam_id": r.serial_number,  # compatibilidad frontend legado
            "src": r.src,
            "location_description": r.location_description,
            "is_active": r.is_active,
            "company_id": r.company_id,
            "company_name": r.company.name if r.company else None,
        }
        for r in rows
    ]


@app.post("/cameras")
def add_camera(cfg: CameraConfig, request: Request, db: Session = Depends(get_db)):
    actor = _resolve_user_from_request(db, request)
    if actor is None:
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    if actor.company_id is None:
        return JSONResponse({"error": "No autorizado"}, status_code=403)

    serial_number = cfg.serial_number.strip()
    src = cfg.src.strip()

    if not serial_number or not src:
        return JSONResponse({"error": "serial_number y src son obligatorios"}, status_code=400)

    location_description = (cfg.location_description or "").strip()
    desired_active = True if cfg.is_active is None else bool(cfg.is_active)

    camera = db.query(Camera).filter(Camera.serial_number == serial_number).first()
    action = "add"

    if camera:
        if camera.company_id is None:
            camera.company_id = actor.company_id
        elif camera.company_id != actor.company_id:
            return JSONResponse({"error": "No autorizado para esta camara"}, status_code=403)

        action = "update"
        camera.src = src
        camera.location_description = location_description
        camera.is_active = desired_active
    else:
        camera = Camera(
            serial_number=serial_number,
            src=src,
            location_description=location_description,
            is_active=desired_active,
            company_id=actor.company_id,
        )
        db.add(camera)

    db.commit()
    db.refresh(camera)

    add_camera_action(db, actor.id, camera.id, action)

    return {
        "ok": True,
        "camera": {
            "id": camera.id,
            "serial_number": camera.serial_number,
            "src": camera.src,
            "location_description": camera.location_description,
            "is_active": camera.is_active,
        },
    }


@app.delete("/cameras/{serial_number}")
def delete_camera(serial_number: str, request: Request, db: Session = Depends(get_db)):
    actor = _resolve_user_from_request(db, request)
    if actor is None:
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    if actor.company_id is None:
        return JSONResponse({"error": "No autorizado"}, status_code=403)

    camera = db.query(Camera).filter(Camera.serial_number == serial_number).first()
    if camera is None:
        return JSONResponse({"error": "Camara no encontrada"}, status_code=404)
    if camera.company_id != actor.company_id:
        return JSONResponse({"error": "No autorizado para esta camara"}, status_code=403)

    camera.is_active = False
    db.commit()

    add_camera_action(db, actor.id, camera.id, "delete")

    return {"ok": True}


@app.post("/cameras/{serial_number}/status")
def set_camera_stream_status(serial_number: str, payload: CameraStreamStatus, request: Request, db: Session = Depends(get_db)):
    actor = _resolve_user_from_request(db, request)
    if actor is None:
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    if actor.company_id is None:
        return JSONResponse({"error": "No autorizado"}, status_code=403)

    camera = db.query(Camera).filter(Camera.serial_number == serial_number).first()
    if camera is None:
        return JSONResponse({"error": "Camara no encontrada"}, status_code=404)
    if camera.company_id != actor.company_id:
        return JSONResponse({"error": "No autorizado para esta camara"}, status_code=403)

    detail = (payload.detail or "").strip()
    action = f"status:{payload.status}"
    if detail:
        action = f"{action}:{detail[:120]}"

    add_camera_action(db, actor.id, camera.id, action)
    return {"ok": True}


# -- Alertas (aceptar/rechazar) ----------------------------
@app.post("/alerts/save")
def save_alert(decision: AlertDecision, request: Request, db: Session = Depends(get_db)):
    serial_number = decision.serial_number.strip()
    reviewer = _resolve_user_from_request(db, request, decision.reviewer_email)
    if reviewer is None:
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    if reviewer.company_id is None:
        return JSONResponse({"error": "No autorizado"}, status_code=403)

    camera = db.query(Camera).filter(Camera.serial_number == serial_number).first()
    if camera is None:
        return JSONResponse({"error": "Camara no encontrada"}, status_code=404)
    if camera.company_id != reviewer.company_id:
        return JSONResponse({"error": "No autorizado para esta camara"}, status_code=403)

    status = _resolve_status(decision)
    alert_ts = _resolve_alert_timestamp(decision.timestamp)

    pending_log = (
        db.query(AlertLog)
        .filter(AlertLog.camera_id == camera.id, AlertLog.status == "pending")
        .order_by(AlertLog.timestamp.desc())
        .first()
    )

    alert_log = None
    if pending_log and abs((pending_log.timestamp - alert_ts).total_seconds()) <= 300:
        alert_log = pending_log
    else:
        alert_log = AlertLog(
            camera_id=camera.id,
            prob=decision.prob,
            timestamp=alert_ts,
        )
        db.add(alert_log)

    alert_log.prob = decision.prob
    alert_log.timestamp = alert_ts
    alert_log.status = status
    alert_log.evidence_path = decision.evidence_path

    if status == "pending":
        alert_log.reviewed_by = None
        alert_log.review_timestamp = None
    else:
        alert_log.reviewed_by = reviewer.id if reviewer else None
        alert_log.review_timestamp = datetime.now(timezone.utc)

    db.commit()
    db.refresh(alert_log)

    return {
        "ok": True,
        "msg": "Alerta actualizada",
        "alert": {
            "id": alert_log.id,
            "camera_id": alert_log.camera_id,
            "serial_number": camera.serial_number,
            "status": alert_log.status,
            "reviewed_by": alert_log.reviewed_by,
            "review_timestamp": _epoch_or_none(alert_log.review_timestamp),
        },
    }
