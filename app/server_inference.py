# uvicorn app.server_inference:app --host 0.0.0.0 --port 8000
#
# Servidor de INFERENCIA (se despliega en RunPod con GPU).
#   - YOLO (batcheado entre camaras) + LSTM + LGBM
#   - WebSocket /ws/stream/{cam_id}?src=... para streaming + deteccion
#   - /overlay/get y /overlay/set para togglear el render de poses
#   - SIN base de datos, SIN auth, SIN HTML
#
# El src de la camara se recibe como query param (lo envia el frontend),
# asi este servidor no necesita acceso a Supabase.

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_THREADS", "8")
os.environ.setdefault("TORCH_NUM_THREADS", "4")
import time
import base64
import yt_dlp
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
    predict_window, load_artifacts, smooth_window,
)

load_dotenv()
log = logging.getLogger("inference")

# ==========================================================
# CONFIGURACION
# ==========================================================
MODELS_DIR   = Path(os.getenv("MODELS_DIR", "models_mix3"))
POSE_WEIGHTS = os.getenv("POSE_WEIGHTS", "yolo11s-pose.pt")

IMGSZ, CONF_POSE, IOU_POSE = 640, 0.25, 0.50

CONF_MIN     = 0.10
POOL_METHOD  = "topk"
TOPK_FRAC    = 0.20
FUSION_W = 0.0
# FUSION_W     = float(os.getenv("FUSION_W", "0.50"))

VIDEO_MAX_SCORES = 900
DRAW_OVERLAY     = True

# Umbral de distancia (px) para considerar que una deteccion en el frame
# actual es la misma persona que ocupaba un slot en el frame anterior.
TRACK_DIST_PX = 300
# Confianza minima para "reclamar" un slot libre (persona nueva en escena).
TRACK_CONF_MIN = 0.70
TRACK_MISSING_TOLERANCE = 15  # frames de tolerancia antes de soltar el slot

# Override en runtime de los umbrales del modelo. None => usar el valor
# cargado desde el JSON del modelo (o THR_ON/THR_OFF env var). Se puede
# modificar en caliente con POST /threshold/set.
THR_ON_RUNTIME: float | None = None
THR_OFF_RUNTIME: float | None = None
LAST_TICK_MS: float = 0.0

# Predicciones de warmup tras llenar la ventana inicial: se descartan del
# pooling y no pueden disparar alertas (evita "spike" inicial del LSTM).
WARMUP_PREDS = int(os.getenv("WARMUP_PREDS", "32"))

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# TARGET_FPS ahora es la cadencia del scheduler GLOBAL (un tick procesa
# TODAS las camaras activas en un solo batch), no un loop por camara.
TARGET_FPS  = float(os.getenv("TARGET_FPS", "24"))
INFER_SLEEP = max(0.0, 1.0 / TARGET_FPS)

MAX_CAMERA_WORKERS = int(os.getenv("MAX_CAMERA_WORKERS", str(multiprocessing.cpu_count())))

# Thread pool para operaciones bloqueantes (cv2, YOLO, Keras)
_executor = ThreadPoolExecutor(max_workers=MAX_CAMERA_WORKERS)

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
    active = sum(1 for w in WORKERS.values() if w.running and w.clients)
    return {
        "ok": True,
        "models_loaded": ARTIFACTS is not None,
        "target_fps": TARGET_FPS,
        "max_camera_workers": MAX_CAMERA_WORKERS,
        "active_cameras": active,
        "registered_cameras": len(WORKERS),
        "last_tick_ms": round(LAST_TICK_MS, 1),
    }


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
    global THR_ON_RUNTIME, THR_OFF_RUNTIME
    if "thr_on" in payload:
        v = payload["thr_on"]
        THR_ON_RUNTIME = float(v) if v is not None else None
    if "thr_off" in payload:
        v = payload["thr_off"]
        THR_OFF_RUNTIME = float(v) if v is not None else None
    return {"ok": True, "thr_on": THR_ON_RUNTIME, "thr_off": THR_OFF_RUNTIME}


# ==========================================================
# LECTOR DE VIDEO SIN LAG (TIEMPO REAL)
# ==========================================================
class RealTimeVideoReader:
    """
    Hilo en segundo plano que siempre lee el frame mas reciente.
    Evita que el buffer de OpenCV se acumule cuando el scheduler
    no alcanza a procesar tan rapido como llegan los frames de red.
    """
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        if self.cap.isOpened():
            self.ret, self.frame = self.cap.read()
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                if ret:
                    self.frame = frame
            if not ret:
                self.running = False
                break

    def read(self):
        with self.lock:
            if self.ret and self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, propId):
        return self.cap.get(propId)

    def set(self, propId, value):
        self.cap.set(propId, value)

    def release(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()


# ==========================================================
# CAMERA WORKER
# ==========================================================
class CameraWorker:
    """
    Ya NO corre su propio loop de inferencia. Solo abre la fuente de video,
    sabe leer un frame cuando el scheduler se lo pide, y postprocesa el
    resultado de YOLO que le llega ya calculado (en batch) desde afuera.
    """
    def __init__(self, cam_id: str, src: str):
        self.cam_id = cam_id
        self.src = src
        self.clients: Set[WebSocket] = set()
        self.running = False
        self.cap = None
        self.is_live_stream = False
        self.W: int = 0
        self.H: int = 0

        self.win_feats = deque(maxlen=SEQ_LEN)
        self.video_scores = deque(maxlen=VIDEO_MAX_SCORES)
        self.on_state = False
        self.n_predictions = 0
        self.trigger_count = 0
        self.cooldown_count = 0

        # Tracking online simple por centroide (reemplaza a ByteTrack,
        # que no se puede batchear entre camaras con persist=True).
        self.slot_centroids = [None, None]   # ultimo centroide (x,y) por slot
        self.missing_frames = [0, 0]

    # -- ciclo de vida ---------------------------------------------------
    async def start(self) -> bool:
        if self.running:
            return True
        loop = asyncio.get_event_loop()
        self.cap = await loop.run_in_executor(_executor, self._open_capture)
        if self.cap is None:
            await self._broadcast({"type": "error", "msg": "No se pudo abrir la fuente de video"})
            return False
        self.running = True
        return True

    async def stop(self):
        self.running = False
        cap = self.cap
        self.cap = None
        if cap is not None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(_executor, cap.release)

    def _open_capture(self):
        src = self.src
        is_network_stream = False
        self.is_live_stream = False

        # --- LOGICA DE YOUTUBE ---
        if isinstance(src, str) and ("youtube.com" in src or "youtu.be" in src):
            log.info(f"Detectado YouTube: {src}. Extrayendo stream...")
            try:
                ydl_opts = {
                    'format': 'bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]/best[height<=720]/best',
                    'quiet': True,
                    'no_warnings': True,
                    'noplaylist': True,
                }
                cookies_path = Path("/workspace/app/cookies.txt")
                if cookies_path.exists():
                    ydl_opts['cookiefile'] = str(cookies_path)
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(src, download=False)
                    src = info['url']
                    self.is_live_stream = bool(info.get('is_live'))
                    is_network_stream = True
                    log.info(f"URL extraida OK. is_live={self.is_live_stream}")
            except Exception as e:
                log.error(f"Error extrayendo URL de YouTube: {e}")
                return None

        target_src = int(src) if str(src).isdigit() else src
        if isinstance(target_src, str) and (target_src.startswith("http") or target_src.startswith("rtsp")):
            is_network_stream = True
            os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "10000"

        if is_network_stream:
            log.info("Usando RealTimeVideoReader (0 lag).")
            cap = RealTimeVideoReader(target_src)
        else:
            log.info("Usando cv2.VideoCapture (archivo local).")
            cap = cv2.VideoCapture(target_src)

        if not cap.isOpened():
            log.error("No se pudo abrir la fuente de video.")
            return None

        self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cap

    # -- lectura (llamada por el scheduler, dentro del executor) --------
    def _read_frame_blocking(self):
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok:
            # Live real (YouTube live / RTSP): si se corta, se acabo de verdad.
            if self.is_live_stream:
                return None
            # Video pregrabado: loop infinito.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                return None
        return frame

    # -- postprocesamiento de UN resultado de YOLO -----------------------
    def process_result(self, res, frame) -> dict:
        keras_model, mu, sd, thr_on, thr_off, lgbm, _pose, stacker = get_artifacts()

        if THR_ON_RUNTIME is not None:
            thr_on = THR_ON_RUNTIME
        if THR_OFF_RUNTIME is not None:
            thr_off = THR_OFF_RUNTIME

        kps_f = np.zeros((2, 17, 3), dtype=np.float32)

        if res.keypoints is not None and res.keypoints.xy is not None and res.keypoints.xy.shape[0] > 0:
            xy = res.keypoints.xy.cpu().numpy()  # (N,17,2)
            c = res.keypoints.conf
            c = c.cpu().numpy() if c is not None else np.ones(xy.shape[:2], dtype=np.float32)
            confs = c.mean(axis=1)

            def centroid(i):
                valid = c[i] > 0.1
                if not valid.any():
                    return None
                return xy[i, valid].mean(axis=0)

            det_centroids = [centroid(i) for i in range(xy.shape[0])]
            used = set()

            # 1) Mantener cada slot con la deteccion mas cercana a su ultimo centroide
            for slot in range(2):
                if self.slot_centroids[slot] is None:
                    continue
                best_i, best_d = -1, 1e9
                for i, cen in enumerate(det_centroids):
                    if i in used or cen is None:
                        continue
                    d = float(np.linalg.norm(cen - self.slot_centroids[slot]))
                    if d < best_d:
                        best_d, best_i = d, i
                if best_i != -1 and best_d < TRACK_DIST_PX:
                    kps_f[slot] = np.concatenate([xy[best_i], c[best_i, ..., None]], axis=-1)
                    self.slot_centroids[slot] = det_centroids[best_i]
                    self.missing_frames[slot] = 0
                    used.add(best_i)
                else:
                    self.missing_frames[slot] += 1
                    if self.missing_frames[slot] > TRACK_MISSING_TOLERANCE:
                        self.slot_centroids[slot] = None

            # 2) Ocupar slots libres con detecciones nuevas de alta confianza
            free_slots = [s for s in range(2) if self.slot_centroids[s] is None]
            if free_slots:
                candidates = sorted(
                    (i for i in range(xy.shape[0]) if i not in used and confs[i] > TRACK_CONF_MIN),
                    key=lambda i: -confs[i],
                )
                for slot, i in zip(free_slots, candidates):
                    kps_f[slot] = np.concatenate([xy[i], c[i, ..., None]], axis=-1)
                    self.slot_centroids[slot] = det_centroids[i]
                    self.missing_frames[slot] = 0

        feat = pool_frame_to_110(kps_f, self.W, self.H, is_tracked=True)
        self.win_feats.append(feat)

        p_win, p_vid = 0.0, 0.0
        if len(self.win_feats) == SEQ_LEN:
            Xw = smooth_window(np.stack(self.win_feats))
            p_win = predict_window(Xw, keras_model, mu, sd,
                                   lgbm=lgbm, fusion_w=FUSION_W, stacker=stacker)
            self.n_predictions += 1
            if self.n_predictions > WARMUP_PREDS:
                self.video_scores.append(p_win)
                p_vid = pool_scores(list(self.video_scores), pool=POOL_METHOD, topk_frac=TOPK_FRAC)

        fired_alert = False
        if self.n_predictions > WARMUP_PREDS:
            if p_vid >= thr_on:
                self.trigger_count += 1
                self.cooldown_count = 0
                if not self.on_state and self.trigger_count >= 15:
                    self.on_state = True
                    fired_alert = True
            elif p_vid <= thr_off:
                self.cooldown_count += 1
                self.trigger_count = 0
                if self.on_state and self.cooldown_count >= 30:
                    self.on_state = False
            else:
                self.trigger_count = 0
                self.cooldown_count = 0

        try:
            shown = res.plot() if DRAW_OVERLAY else frame
        except Exception:
            shown = frame

        _, buf = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 65])
        jpg = base64.b64encode(buf).decode()

        return {
            "p_win": p_win,
            "p_vid": p_vid,
            "on": self.on_state,
            "ts": time.time(),
            "jpg_b64": jpg,
            "fired_alert": fired_alert,
        }

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

# ==========================================================
# SCHEDULER GLOBAL (un solo loop batchea TODAS las camaras)
# ==========================================================
class InferenceScheduler:
    def __init__(self):
        self.running = False
        self.task = None

    async def start(self):
        if not self.running:
            self.running = True
            self.task = asyncio.create_task(self._loop())

    async def _loop(self):
        global LAST_TICK_MS
        loop = asyncio.get_event_loop()
        while self.running:
            t0 = time.time()
            try:
                await self._tick(loop)
            except Exception as e:
                log.error(f"[scheduler] tick fallo: {e}")

            elapsed = time.time() - t0
            LAST_TICK_MS = elapsed * 1000
            if elapsed > INFER_SLEEP * 1.5:
                log.warning(
                    f"[scheduler] tick={elapsed*1000:.0f}ms > "
                    f"budget={INFER_SLEEP*1000:.0f}ms (bajale camaras o subile TARGET_FPS)"
                )
            await asyncio.sleep(max(0.0, INFER_SLEEP - elapsed))

    async def _tick(self, loop):
        active = [(cid, w) for cid, w in list(WORKERS.items())
                 if w.running and w.clients and w.cap is not None]
        if not active:
            return

        # ── Fase 1: lectura de frames EN PARALELO ──
        t_read0 = time.time()
        read_tasks = [loop.run_in_executor(_executor, w._read_frame_blocking)
                     for _, w in active]
        frames_raw = await asyncio.gather(*read_tasks)
        t_read = time.time() - t_read0

        dead  = [(cid, w) for (cid, w), f in zip(active, frames_raw) if f is None]
        valid = [(cid, w, f) for (cid, w), f in zip(active, frames_raw) if f is not None]

        for cam_id, w in dead:
            await w._broadcast({"type": "error", "msg": "Fin del stream"})
            await w.stop()

        if not valid:
            return

        # ── Fase 2: UNA sola llamada batcheada a la GPU ──
        _, _, _, _, _, _, pose_model, _ = get_artifacts()
        frames = [f for _, _, f in valid]

        def run_batch():
            use_half = next(pose_model.model.parameters()).is_cuda
            return pose_model.predict(
                frames, imgsz=IMGSZ, conf=CONF_POSE, iou=IOU_POSE,
                verbose=False, half=use_half,
            )

        t_yolo0 = time.time()
        results = await loop.run_in_executor(_executor, run_batch)
        t_yolo = time.time() - t_yolo0

        # ── Fase 3: postprocesamiento (LSTM + JPEG) EN PARALELO ──
        t_post0 = time.time()
        post_tasks = [
            loop.run_in_executor(_executor, w.process_result, res, frame)
            for (cam_id, w, frame), res in zip(valid, results)
        ]
        post_results = await asyncio.gather(*post_tasks)
        t_post = time.time() - t_post0

        log.warning(
            f"[tick-breakdown] read={t_read*1000:.0f}ms  yolo={t_yolo*1000:.0f}ms  "
            f"post={t_post*1000:.0f}ms  (n_cams={len(valid)})"
        )

        # ── Fase 4: broadcast ──
        for (cam_id, w, _), result in zip(valid, post_results):
            if result["fired_alert"]:
                await w._broadcast({
                    "type": "alert",
                    "cam_id": cam_id,
                    "prob": result["p_vid"],
                    "ts": result["ts"],
                })

            await w._broadcast({
                "type": "frame",
                "cam_id": cam_id,
                "p_win": result["p_win"],
                "p_vid": result["p_vid"],
                "on": result["on"],
                "ts": result["ts"],
                "jpg_b64": result["jpg_b64"],
            })

SCHEDULER = InferenceScheduler()


@app.on_event("startup")
async def _on_startup():
    await SCHEDULER.start()

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
        ok = await worker.start()
        if not ok:
            await ws.close()
            return

    worker.clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        worker.clients.discard(ws)
        if not worker.clients:
            await worker.stop()
