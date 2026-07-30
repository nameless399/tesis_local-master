# uvicorn app.server_inference:app --host 0.0.0.0 --port 8000
#
# Servidor de INFERENCIA (se despliega en RunPod con GPU).
#   - YOLO + LSTM + LGBM
#   - WebSocket /ws/stream/{cam_id}?src=... para streaming + detección
#   - /overlay/get y /overlay/set para togglear el render de poses
#   - SIN base de datos, SIN auth, SIN HTML
#
# El src de la cámara se recibe como query param (lo envía el frontend),
# así este servidor no necesita acceso a Supabase.

import os
import time
import base64
import asyncio
import logging
import threading
import multiprocessing
from typing import Set
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

from app.pipeline import (
    SEQ_LEN, pool_frame_to_110, pool_scores,
    predict_window, load_artifacts, smooth_window
)

try:
    cv2.setNumThreads(1)
except Exception:
    pass

# Keras/TF en CPU (modelo LSTM pequeno). YOLO sigue en GPU via torch.
# Evita pelear con cuDNN (TF 2.21 bundled libs chocan con OpenCV ffmpeg)
# y libera VRAM para YOLO. IMPORTANTE: cv2 debe importarse ANTES que TF
# o cv2.VideoCapture segfaultea por conflicto de abseil/protobuf.

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline import (
    SEQ_LEN, pool_frame_to_110, pool_scores,
    predict_window, load_artifacts,
)

load_dotenv()
log = logging.getLogger("inference")

# ==========================================================
# CONFIGURACIÓN
# ==========================================================
MODELS_DIR   = Path(os.getenv("MODELS_DIR", "models_mix3"))
POSE_WEIGHTS = os.getenv("POSE_WEIGHTS", "yolo11s-pose.pt")

IMGSZ, CONF_POSE, IOU_POSE = 640, 0.25, 0.50
TOPK = 4

CONF_MIN     = 0.10
POOL_METHOD  = "topk"
TOPK_FRAC    = 0.20
FUSION_W = 0.0
# FUSION_W     = float(os.getenv("FUSION_W", "0.50"))

VIDEO_MAX_SCORES = 900
DRAW_OVERLAY     = True

# Override en runtime de los umbrales del modelo. None => usar el valor
# cargado desde el JSON del modelo (o THR_ON/THR_OFF env var). Se puede
# modificar en caliente con POST /threshold/set.
THR_ON_RUNTIME: float | None = None
THR_OFF_RUNTIME: float | None = None

# Predicciones de warmup tras llenar la ventana inicial: se descartan del
# pooling y no pueden disparar alertas (evita "spike" inicial del LSTM).
WARMUP_PREDS = int(os.getenv("WARMUP_PREDS", "32"))

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Thread pool para operaciones bloqueantes (cv2, YOLO, Keras)
_executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2)

# Modelos pesados: se cargan bajo demanda.
ARTIFACTS = None
ARTIFACTS_LOCK = threading.Lock()


def get_artifacts():
    global ARTIFACTS
    if ARTIFACTS is not None:
        return ARTIFACTS

    with ARTIFACTS_LOCK:
        if ARTIFACTS is None:
            log.info("Cargando modelos bajo demanda desde %s", MODELS_DIR)
            ARTIFACTS = load_artifacts(MODELS_DIR, POSE_WEIGHTS)
    return ARTIFACTS


# ==========================================================
# APP
# ==========================================================
app = FastAPI(title="Inference Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"service": "inference", "ok": True}


@app.get("/health")
def health():
    return {"ok": True, "models_loaded": ARTIFACTS is not None}


# ── Overlay toggle ────────────────────────────────────────
@app.get("/overlay/get")
def overlay_get():
    return {"overlay": DRAW_OVERLAY}

@app.post("/overlay/set")
def overlay_set(payload: dict = Body(...)):
    global DRAW_OVERLAY
    DRAW_OVERLAY = bool(payload.get("overlay", True))
    return {"ok": True, "overlay": DRAW_OVERLAY}


# ── Thresholds (runtime override) ─────────────────────────
@app.get("/threshold/get")
def threshold_get():
    """
    Devuelve umbrales efectivos (override si existe, sino los del modelo).
    """
    thr_on_model = None
    thr_off_model = None
    if ARTIFACTS is not None:
        _, _, _, thr_on_model, thr_off_model, *_ = ARTIFACTS
    return {
        "thr_on": THR_ON_RUNTIME if THR_ON_RUNTIME is not None else thr_on_model,
        "thr_off": THR_OFF_RUNTIME if THR_OFF_RUNTIME is not None else thr_off_model,
        "thr_on_override": THR_ON_RUNTIME,
        "thr_off_override": THR_OFF_RUNTIME,
        "thr_on_model": thr_on_model,
        "thr_off_model": thr_off_model,
    }


@app.post("/threshold/set")
def threshold_set(payload: dict = Body(...)):
    """
    Setea override de THR_ON / THR_OFF. Manda null para limpiar override.
    Body: { "thr_on": 0.70, "thr_off": 0.55 }
    """
    global THR_ON_RUNTIME, THR_OFF_RUNTIME
    if "thr_on" in payload:
        v = payload["thr_on"]
        THR_ON_RUNTIME = float(v) if v is not None else None
    if "thr_off" in payload:
        v = payload["thr_off"]
        THR_OFF_RUNTIME = float(v) if v is not None else None
    return {"ok": True, "thr_on": THR_ON_RUNTIME, "thr_off": THR_OFF_RUNTIME}


# ==========================================================
# CAMERA WORKER
# ==========================================================
class CameraWorker:
    def __init__(self, cam_id: str, src: str):
        self.cam_id = cam_id
        self.src = src
        self.clients: Set[WebSocket] = set()
        self.running = False
        self.task = None

        self.win_feats = deque(maxlen=SEQ_LEN)
        self.video_scores = deque(maxlen=VIDEO_MAX_SCORES)
        self.on_state = False
        self.n_predictions = 0
        self.W: int = 0
        self.H: int = 0

        # [NUEVO] Contadores para suavizar las alertas
        self.trigger_count = 0
        self.cooldown_count = 0

        # [NUEVO] Memoria de Tracking para 2 personas principales
        self.target_ids = [None, None]       # Guardará los IDs de tracking de YOLO
        self.missing_frames = [0, 0]         # Tolerancia: frames que llevan desaparecidos

    async def start(self):
        if not self.running:
            self.running = True
            self.task = asyncio.create_task(self._loop())

    def _open_capture(self):
        src = int(self.src) if self.src.isdigit() else self.src
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            return None
        self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cap

    def _grab_and_infer(self, cap) -> dict | None:
        try:
            ok, frame = cap.read()
            if not ok:
                # ¡TRUCO MAGICO! Si el video se acaba, lo reiniciamos desde el frame 0 (Bucle infinito)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    return None

            # Cargamos los modelos (solo lo hace la primera vez)
            keras_model, mu, sd, thr_on, thr_off, lgbm, pose, stacker = get_artifacts()

            # Override en caliente desde /threshold/set
            if THR_ON_RUNTIME is not None:
                thr_on = THR_ON_RUNTIME
            if THR_OFF_RUNTIME is not None:
                thr_off = THR_OFF_RUNTIME

            res = pose.track(frame, imgsz=IMGSZ, conf=CONF_POSE, iou=IOU_POSE, 
                             persist=True, tracker="bytetrack.yaml", verbose=False)[0]

            # Inicializamos un array vacío para exactamente 2 personas (Top-2)
            kps_f = np.zeros((2, 17, 3), dtype=np.float32)

            if res.keypoints is not None and res.boxes is not None and res.boxes.id is not None:
                xy = res.keypoints.xy.cpu().numpy()
                c  = res.keypoints.conf.cpu().numpy()
                ids = res.boxes.id.int().cpu().numpy()   # IDs únicos que da el tracker
                confs = c.mean(axis=1)                   # Confianza media del esqueleto
                
                # 2. Verificar si nuestros objetivos actuales siguen en pantalla
                for i in range(2):
                    if self.target_ids[i] is not None:
                        if self.target_ids[i] not in ids:
                            # Se perdió de vista. Le damos 15 frames (~0.5s) de tolerancia antes de olvidarlo
                            self.missing_frames[i] += 1
                            if self.missing_frames[i] > 15:
                                self.target_ids[i] = None
                        else:
                            self.missing_frames[i] = 0 # Lo vimos, reseteamos tolerancia
                
                # 3. Asignar nuevos objetivos si hay "huecos" libres
                # Priorizamos a las personas con mayor confianza (ej: > 0.70)
                available_indices = np.argsort(-confs) 
                for idx in available_indices:
                    current_id = ids[idx]
                    current_conf = confs[idx]

                    if current_id in self.target_ids:
                        continue # Ya lo estamos trackeando
                    
                    # Umbral del 0.70 que mencionaste para decir "esto sí es una persona firme"
                    if current_conf > 0.70:
                        if self.target_ids[0] is None:
                            self.target_ids[0] = current_id
                            self.missing_frames[0] = 0
                        elif self.target_ids[1] is None:
                            self.target_ids[1] = current_id
                            self.missing_frames[1] = 0

                # 4. Extraer los keypoints en el ORDEN ESTRICTO de target_ids
                for i in range(2):
                    t_id = self.target_ids[i]
                    if t_id is not None and t_id in ids:
                        # Buscamos en qué índice de la salida actual de YOLO está nuestro actor
                        idx_yolo = np.where(ids == t_id)[0][0]
                        kps_f[i] = np.concatenate([xy[idx_yolo], c[idx_yolo, ..., None]], axis=-1)
            
            feat = pool_frame_to_110(kps_f, self.W, self.H, is_tracked=True)
            self.win_feats.append(feat)

            p_win, p_vid = 0.0, 0.0

            if len(self.win_feats) == SEQ_LEN:
                Xw = smooth_window(np.stack(self.win_feats))
                
                p_win = predict_window(Xw, keras_model, mu, sd,
                        lgbm=lgbm, fusion_w=FUSION_W, stacker=stacker)
                p_win = predict_window(Xw, keras_model, mu, sd,
                        lgbm=lgbm, fusion_w=FUSION_W, stacker=stacker)
                self.n_predictions += 1
                if self.n_predictions > WARMUP_PREDS:
                    self.video_scores.append(p_win)
                    p_vid = pool_scores(list(self.video_scores), pool=POOL_METHOD, topk_frac=TOPK_FRAC)

            fired_alert = False
            if self.n_predictions > WARMUP_PREDS:
                
                # 1. Si el score supera el umbral, empezamos a contar
                if p_vid >= thr_on:
                    self.trigger_count += 1
                    self.cooldown_count = 0 # Se cancela el enfriamiento
                    
                    # Dispara la alerta solo si lleva 15 frames seguidos (~0.5 segundos) por encima del umbral
                    if not self.on_state and self.trigger_count >= 15:
                        self.on_state = True
                        fired_alert = True
                
                # 2. Si el score baja del umbral de apagado, empezamos a enfriar
                elif p_vid <= thr_off:
                    self.cooldown_count += 1
                    self.trigger_count = 0 # Se cancela el disparo
                    
                    # Apaga la alerta solo si lleva 30 frames seguidos (~1 segundo) en calma
                    if self.on_state and self.cooldown_count >= 30:
                        self.on_state = False
                
                # 3. Zona muerta (entre thr_off y thr_on)
                else:
                    # En la zona muerta no disparamos ni apagamos, pero reiniciamos contadores
                    self.trigger_count = 0
                    self.cooldown_count = 0

            try:
                shown = res.plot() if DRAW_OVERLAY else frame
            except Exception:
                shown = frame
            
            _, buf = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 65])
            jpg = base64.b64encode(buf).decode()

            now = time.time()
            return {
                "p_win": p_win,
                "p_vid": p_vid,
                "on": self.on_state,
                "ts": now,
                "jpg_b64": jpg,
                "fired_alert": fired_alert,
            }
        
        except Exception as e:
            # Si hay un error (ej. falta un modelo), ahora sí nos lo dirá en la terminal
            print(f"\n[ERROR CRITICO] Algo falló procesando el frame: {e}\n")
            return None

    async def _loop(self):
        loop = asyncio.get_event_loop()

        cap = await loop.run_in_executor(_executor, self._open_capture)
        if cap is None:
            await self._broadcast({"type": "error", "msg": "No se pudo abrir la fuente de video"})
            self.running = False
            return

        try:
            while self.running and self.clients:
                result = await loop.run_in_executor(_executor, self._grab_and_infer, cap)
                
                if result is None:
                    # MAGIA PARA LA TESIS: Si el video termina y es un archivo local, ¡lo rebobinamos!
                    if not str(self.src).isdigit() and not str(self.src).startswith("rtsp"):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rebobinar al frame 0
                        continue # Volver a empezar
                    else:
                        await self._broadcast({"type": "error", "msg": "Fin del stream"})
                        break

                if result["fired_alert"]:
                    await self._broadcast({
                        "type": "alert",
                        "cam_id": self.cam_id,
                        "prob": result["p_vid"],
                        "ts": result["ts"],
                    })

                await self._broadcast({
                    "type": "frame",
                    "cam_id": self.cam_id,
                    "p_win": result["p_win"],
                    "p_vid": result["p_vid"],
                    "on": result["on"],
                    "ts": result["ts"],
                    "jpg_b64": result["jpg_b64"],
                })

                # MAGIA 2: Frenamos el servidor 0.03 segundos para que vaya a 30 FPS.
                # ¡Esto evita que el navegador colapse y quite la pantalla negra!
                await asyncio.sleep(0.033)
        finally:
            await loop.run_in_executor(_executor, cap.release)
            self.running = False

    async def _broadcast(self, msg: dict):
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


# ── WebSocket endpoint ────────────────────────────────────
WORKERS: dict[str, CameraWorker] = {}


@app.websocket("/ws/stream/{cam_id}")
async def ws_stream(ws: WebSocket, cam_id: str, src: str = ""):
    """
    El frontend pasa el src como query param:
      wss://<runpod>/ws/stream/lobby?src=rtsp%3A%2F%2F...
    """
    await ws.accept()

    if not src:
        await ws.send_json({"type": "error", "msg": "missing src query param"})
        await ws.close()
        return

    worker = WORKERS.get(cam_id)
    if not worker or not worker.running:
        worker = CameraWorker(cam_id, src)
        WORKERS[cam_id] = worker
        await worker.start()

    worker.clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        worker.clients.discard(ws)
