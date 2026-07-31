# app/pipeline.py  –  Funciones compartidas del pipeline de pose-estimation
#
# Centraliza: pool_frame_to_51, frame_visible, pool_scores, featurize_T51,
#             norm_apply y predict_window.  Usado por server.py, inferencia_video.py
#             y train_lgbm.py para evitar divergencias.

import json
import os
from pathlib import Path
from typing import List

import numpy as np
import joblib

def smooth_window(Xw: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Aplica una media móvil para quitar el temblor del esqueleto de YOLO"""
    pad_size = window_size // 2
    # Rellenamos los bordes para no perder frames
    Xw_padded = np.pad(Xw, ((pad_size, pad_size), (0, 0)), mode='edge')
    smoothed = np.zeros_like(Xw)
    for i in range(Xw.shape[0]):
        smoothed[i] = np.mean(Xw_padded[i : i+window_size], axis=0)
    return smoothed
# ══════════════════════════════════════════════════════════
# Constantes por defecto
# ══════════════════════════════════════════════════════════
SEQ_LEN      = 32
CONF_MIN     = 0.10
MIN_VIS_FRAC = 0.30
HYST_GAP     = 0.10
N_FEATURES_V2 = 110   # 2 personas × 51 + 8 interacción

# --- Constantes para V2 ---
N_PERSONS    = 2        
N_JOINTS     = 17
N_INTER      = 8        
N_FEATURES   = N_PERSONS * N_JOINTS * 3 + N_INTER 

_NOSE                    = 0
_WRIST_L, _WRIST_R       = 9, 10
_HIP_L,   _HIP_R         = 11, 12
_SHOULDER_L, _SHOULDER_R = 5, 6

def frame_visible_v2(kps_f: np.ndarray, conf_min: float = 0.10) -> bool:
    if kps_f is None or kps_f.size == 0:
        return False
    return bool((np.nan_to_num(kps_f[..., 2], nan=0.0) >= conf_min).any())

def pool_frame_to_110(kps_f: np.ndarray | None, W: int, H: int,  is_tracked: bool = False) -> np.ndarray:
    """Extrae features de 2 personas + distancias de interacción para el modelo V2"""
    out = np.zeros(N_FEATURES, dtype=np.float32)
    if kps_f is None or kps_f.size == 0:
        return out

    K = kps_f.shape[0]

    # --- ¡EL SWITCH MÁGICO! ---
    if is_tracked:
        # Si ya vienen trackeados (por YOLO en vivo o por el Offline Tracker), mantenemos su orden real
        order = np.arange(min(K, N_PERSONS))
    else:
        # Comportamiento antiguo: Si no están trackeados, ordenamos por el que tenga más confianza
        confs = np.nan_to_num(kps_f[..., 2], nan=0.0).mean(axis=1)
        order = np.argsort(-confs)

    p_data = []
    for pi in range(N_PERSONS):
        base = pi * N_JOINTS * 3
        kp   = np.zeros((N_JOINTS, 3), dtype=np.float32)
        
        # Verificamos que el índice exista en nuestro 'order'
        if pi < len(order) and order[pi] < K:
            raw = kps_f[order[pi]]
            c   = np.nan_to_num(raw[:, 2], nan=0.0)
            
            # --- MEJORA: CENTRADO EN EL CUERPO ---
            valid = c > 0
            cx = raw[valid, 0].mean() if valid.any() else 0.0
            cy = raw[valid, 1].mean() if valid.any() else 0.0

            for j in range(N_JOINTS):
                if c[j] > 0 and np.isfinite(raw[j, 0]) and np.isfinite(raw[j, 1]):
                    # Restamos el centro y sumamos 0.5 para mantenerlo en rango [0, 1]
                    kp[j, 0] = np.clip(((raw[j, 0] - cx) / max(W, 1)) + 0.5, 0.0, 1.0)
                    kp[j, 1] = np.clip(((raw[j, 1] - cy) / max(H, 1)) + 0.5, 0.0, 1.0)
                    kp[j, 2] = np.clip(float(c[j]), 0.0, 1.0)
        
        out[base : base + N_JOINTS * 3] = kp.reshape(-1)
        p_data.append(kp)

    off    = N_PERSONS * N_JOINTS * 3
    p1, p2 = p_data[0], p_data[1]
    c1, c2 = p1[:, 2], p2[:, 2]
    p1_ok  = c1.mean() > 0.10
    p2_ok  = c2.mean() > 0.10

    vis_total = ((c1 > 0.1).sum() + (c2 > 0.1).sum()) / (N_JOINTS * N_PERSONS)
    out[off + 0] = float(vis_total)

    if p1_ok:
        out[off + 1] = float((c1[_WRIST_L] + c1[_WRIST_R]) / 2.0)

    if p1_ok and p2_ok:
        w1  = c1 + 1e-6
        w2  = c2 + 1e-6
        cm1 = np.array([np.average(p1[:, 0], weights=w1),
                         np.average(p1[:, 1], weights=w1)])
        cm2 = np.array([np.average(p2[:, 0], weights=w2), 
                        np.average(p2[:, 1], weights=w2)])

        out[off + 2] = float(np.linalg.norm(cm1 - cm2))

        def jdist(j_a, pa, j_b, pb):
            a, b = pa[j_a], pb[j_b]
            if a[2] > 0.1 and b[2] > 0.1:
                return float(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))
            return 0.0

        out[off + 3] = jdist(_WRIST_R, p1, _NOSE,   p2)
        out[off + 4] = jdist(_WRIST_R, p2, _NOSE,   p1)
        out[off + 5] = jdist(_WRIST_L, p1, _NOSE,   p2)
        out[off + 6] = jdist(_WRIST_R, p1, _HIP_L,  p2)

        def tilt(kp):
            sx = kp[_SHOULDER_R, 0] - kp[_HIP_R, 0]
            sy = kp[_SHOULDER_R, 1] - kp[_HIP_R, 1]
            return float(np.arctan2(sy, sx + 1e-6) / np.pi)

        out[off + 7] = abs(tilt(p1) - tilt(p2))

    return out

# --- arriba Constantes para V2 arriba ---

# --- Indices COCO de joints ---
#  0=nose  1=L_eye  2=R_eye  3=L_ear  4=R_ear
#  5=L_shoulder  6=R_shoulder  7=L_elbow  8=R_elbow
#  9=L_wrist  10=R_wrist  11=L_hip  12=R_hip
# 13=L_knee  14=R_knee  15=L_ankle  16=R_ankle

# Pares de distancias
PAIR_DISTS = [
    (5, 6),    # hombro-hombro
    (9, 10),   # muñeca-muñeca
    (0, 9),    # nariz-muñeca_izq  (golpe)
    (0, 10),   # nariz-muñeca_der  (golpe)
    (11, 12),  # cadera-cadera
    (15, 16),  # tobillo-tobillo
    (9, 12),   # muñeca_izq-cadera_der  (cruce)
    (10, 11),  # muñeca_der-cadera_izq  (cruce)
    (5, 11),   # hombro_izq-cadera_izq  (torso)
    (6, 12),   # hombro_der-cadera_der  (torso)
]

# Triplets para angulos articulares: (A, vertice, B)
ANGLE_JOINTS = [
    (5, 7, 9),    # hombro_izq -> codo_izq -> muñeca_izq
    (6, 8, 10),   # hombro_der -> codo_der -> muñeca_der
    (11, 13, 15),  # cadera_izq -> rodilla_izq -> tobillo_izq
    (12, 14, 16),  # cadera_der -> rodilla_der -> tobillo_der
    (7, 5, 11),   # codo_izq -> hombro_izq -> cadera_izq
    (8, 6, 12),   # codo_der -> hombro_der -> cadera_der
    (5, 0, 6),    # hombro_izq -> nariz -> hombro_der  (inclinacion cabeza)
]


# ══════════════════════════════════════════════════════════
# Utilidades de frame
# ══════════════════════════════════════════════════════════
def pool_frame_to_51(kps_f: np.ndarray | None, W: int, H: int) -> np.ndarray:
    """(P,17,3) -> (51,)  pooling por joint con mayor confianza, normalizado por W,H."""
    out = np.zeros((17, 3), dtype=np.float32)
    if kps_f is None or kps_f.size == 0:
        return out.reshape(-1)
    conf_j = np.nan_to_num(kps_f[..., 2], nan=0.0)
    for j in range(17):
        idx = int(np.argmax(conf_j[:, j]))
        c = conf_j[idx, j]
        if c > 0:
            x, y, _ = kps_f[idx, j, :]
            if np.isfinite(x) and np.isfinite(y):
                out[j, 0] = np.clip(x / max(W, 1), 0.0, 1.0)
                out[j, 1] = np.clip(y / max(H, 1), 0.0, 1.0)
                out[j, 2] = float(np.clip(c, 0.0, 1.0))
    return out.reshape(-1)


def frame_visible(kps_f: np.ndarray | None, conf_min: float = CONF_MIN) -> bool:
    if kps_f is None or kps_f.size == 0:
        return False
    conf = np.nan_to_num(kps_f[..., 2], nan=0.0)
    return bool((conf >= conf_min).any())


# ══════════════════════════════════════════════════════════
# Pooling de scores a nivel video
# ══════════════════════════════════════════════════════════
def pool_scores(scores: List[float], pool: str = "topk", topk_frac: float = 0.2) -> float:
    if not scores:
        return 0.0
    arr = np.asarray(scores, dtype=np.float32)
    if pool == "max":
        return float(arr.max())
    if pool == "mean":
        return float(arr.mean())
    k = max(1, int(len(arr) * topk_frac))
    return float(np.partition(arr, -k)[-k:].mean())


# ══════════════════════════════════════════════════════════
# Featurizador tabular para LGBM  (DEBE coincidir con train_lgbm.py)
# ══════════════════════════════════════════════════════════
def _stats(a: np.ndarray) -> np.ndarray:
    """mean, std, min, max por columna."""
    return np.concatenate([a.mean(0), a.std(0), a.min(0), a.max(0)], axis=0)


def _angle_between(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angulo en radianes en el vertice, para cada frame.  (T,)"""
    va = a - vertex  # (T, 2)
    vb = b - vertex
    cos = np.sum(va * vb, axis=1) / (np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1) + 1e-8)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def featurize_T51(X_win: np.ndarray) -> np.ndarray:
    """
    Desde una ventana (T, 51) genera el vector tabular para LGBM.

    Bloques de features:
      1. Stats x, y, velocidad, confianza  (4 * 4stats * 17joints = 272)
      2. Aceleracion stats                  (4stats * 17joints = 68)
      3. Distancias entre pares             (10 pares * 2 = 20)
      4. Angulos articulares stats          (7 angulos * 4stats = 28)

    Total: 272 + 68 + 20 + 28 = 388 features

    Returns: (1, D) array listo para predict.
    """
    T = X_win.shape[0]
    xyz = X_win.reshape(T, 17, 3)
    x, y, c = xyz[..., 0], xyz[..., 1], xyz[..., 2]  # (T, 17) cada uno

    # --- Velocidades (1ra derivada) ---
    dx = np.diff(x, axis=0, prepend=x[0:1])
    dy = np.diff(y, axis=0, prepend=y[0:1])
    v = np.sqrt(dx * dx + dy * dy)

    # --- Aceleracion (2da derivada, magnitud) ---
    ddx = np.diff(dx, axis=0, prepend=dx[0:1])
    ddy = np.diff(dy, axis=0, prepend=dy[0:1])
    acc = np.sqrt(ddx * ddx + ddy * ddy)

    # --- Stats basicas ---
    Fx = _stats(x)
    Fy = _stats(y)
    Fv = _stats(v)
    Fc = _stats(c)
    Fa = _stats(acc)

    # --- Distancias entre pares ---
    D = []
    for (i, j) in PAIR_DISTS:
        dij = np.sqrt((x[:, i] - x[:, j]) ** 2 + (y[:, i] - y[:, j]) ** 2)
        D += [dij.mean(), dij.std()]
    D = np.array(D, dtype=np.float32)

    # --- Angulos articulares ---
    angles = []
    for (a_idx, v_idx, b_idx) in ANGLE_JOINTS:
        pt_a = np.stack([x[:, a_idx], y[:, a_idx]], axis=1)    # (T, 2)
        pt_v = np.stack([x[:, v_idx], y[:, v_idx]], axis=1)
        pt_b = np.stack([x[:, b_idx], y[:, b_idx]], axis=1)
        ang = _angle_between(pt_a, pt_v, pt_b)                 # (T,)
        angles.append(ang)
    angles = np.stack(angles, axis=1)  # (T, 7)
    Fang = _stats(angles)

    feat = np.concatenate([Fx, Fy, Fv, Fc, Fa, D, Fang], axis=0).astype(np.float32)
    return feat.reshape(1, -1)


# ══════════════════════════════════════════════════════════
# Normalización para el modelo Keras (LSTM)
# ══════════════════════════════════════════════════════════
def norm_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """X: (1, T, 51) -> normalizado."""
    T, F = X.shape[1], X.shape[2]
    X2 = X.reshape(-1, F)
    Xn = (X2 - mu) / (sd + 1e-6)
    return Xn.reshape(1, T, F).astype("float32")


# ══════════════════════════════════════════════════════════
# Predicción fusionada  LSTM + LGBM (soporta stacking)
# ══════════════════════════════════════════════════════════
def predict_window(Xw: np.ndarray, keras_model, mu, sd,
                   lgbm=None, fusion_w: float = 0.5,
                   stacker=None) -> float:
    """
    Xw: (T, 51)
    fusion_w: peso para fusion lineal (ignorado si stacker != None)
    stacker:  modelo meta-learner (LogisticRegression) entrenado sobre [p_lstm, p_lgbm]
    """
    # --- Keras (LSTM) ---
    X = Xw[np.newaxis, ...]
    X = norm_apply(X, mu, sd)
    p_keras = float(keras_model(X, training=False).numpy().ravel()[0])

    if lgbm is None or fusion_w <= 0.0:
        return np.clip(p_keras, 0.0, 1.0)

    # --- LGBM ---
    feat = featurize_T51(Xw)
    expected = getattr(lgbm, "n_features_in_", None)
    if expected is not None and feat.shape[1] != expected:
        return np.clip(p_keras, 0.0, 1.0)

    try:
        p_lgbm = float(lgbm.predict_proba(feat)[:, 1][0])
    except Exception:
        return np.clip(p_keras, 0.0, 1.0)

    # --- Fusion ---
    if stacker is not None:
        try:
            meta_feat = np.array([[p_keras, p_lgbm]], dtype=np.float32)
            p_fused = float(stacker.predict_proba(meta_feat)[:, 1][0])
            return np.clip(p_fused, 0.0, 1.0)
        except Exception:
            pass

    # Fallback: fusion lineal
    return float(np.clip((1.0 - fusion_w) * p_keras + fusion_w * p_lgbm, 0.0, 1.0))


# ══════════════════════════════════════════════════════════
# Carga de artefactos (modelos + stats + threshold + stacker)
# ══════════════════════════════════════════════════════════
def load_artifacts(models_dir: str | Path, pose_weights: str):
    """
    Retorna: (keras_model, mu, sd, thr_on, thr_off, lgbm, pose, stacker)
    """
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Deja correr el LSTM en la 4090. set_memory_growth evita que TF
        # reserve toda la VRAM de una vez y le quite espacio a YOLO.
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    else:
        tf_threads = int(os.getenv("TF_NUM_THREADS", "2"))
        try:
            tf.config.threading.set_intra_op_parallelism_threads(tf_threads)
            tf.config.threading.set_inter_op_parallelism_threads(tf_threads)
        except RuntimeError:
            pass
    
    from keras.models import load_model
    from ultralytics import YOLO
    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True   # Ada gana bastante con esto en matmuls
    torch.backends.cudnn.allow_tf32 = True
    
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))

    models_dir = Path(models_dir)

    keras_path  = models_dir / "v2_bilstm_attention.keras"
    stats_path  = models_dir / "v2_norm_stats.npz"
    thr_path    = models_dir / "v2_threshold.json"
    lgbm_path   = models_dir / "lgbm_model.pkl"
    stacker_path = models_dir / "stacker.pkl"

    keras_model = load_model(str(keras_path), compile=False)

    stats = np.load(stats_path)
    mu = stats["mean"].astype("float32")
    sd = stats["std"].astype("float32")

    thr_on = 0.5
    if thr_path.exists():
        thr_on = float(json.loads(thr_path.read_text(encoding="utf-8")).get("best_threshold", 0.5))

    # Override opcional por env var (sin tocar el JSON del modelo).
    env_thr_on = os.getenv("THR_ON")
    if env_thr_on:
        try:
            thr_on = float(env_thr_on)
            print(f"[BOOT] THR_ON sobreescrito por env var -> {thr_on:.2f}")
        except ValueError:
            print(f"[BOOT] THR_ON env var invalido: {env_thr_on!r}")

    thr_off = max(0.0, thr_on - HYST_GAP)
    env_thr_off = os.getenv("THR_OFF")
    if env_thr_off:
        try:
            thr_off = float(env_thr_off)
            print(f"[BOOT] THR_OFF sobreescrito por env var -> {thr_off:.2f}")
        except ValueError:
            print(f"[BOOT] THR_OFF env var invalido: {env_thr_off!r}")

    lgbm = None
    if lgbm_path.exists():
        try:
            lgbm = joblib.load(lgbm_path)
            print(f"[BOOT] LGBM cargado desde {lgbm_path}")
        except Exception as e:
            print(f"[BOOT] LGBM falló al cargar ({e}), continuo sin LGBM")

    stacker = None
    if stacker_path.exists():
        try:
            stacker = joblib.load(stacker_path)
            print(f"[BOOT] Stacker cargado desde {stacker_path}")
        except Exception as e:
            print(f"[BOOT] Stacker falló ({e}), usando fusion lineal")

    pose = YOLO(pose_weights)
    if torch.cuda.is_available():
        pose.to("cuda")
    print("TF ve GPU:", tf.config.list_physical_devices('GPU'))
    print("Torch ve GPU:", torch.cuda.is_available())
    print("YOLO device:", pose.device)
    tf_threads_label = "GPU" if gpus else tf_threads
    print(f"[BOOT] Keras={keras_path.name} | THR_ON={thr_on:.2f} THR_OFF={thr_off:.2f}"
          f" | LGBM={'ON' if lgbm else 'OFF'} | Stacker={'ON' if stacker else 'OFF'}"
          f" | TF_threads={tf_threads_label} | Torch_threads={os.getenv('TORCH_NUM_THREADS', '2')}")

    return keras_model, mu, sd, thr_on, thr_off, lgbm, pose, stacker
