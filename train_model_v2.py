# train_model_v2.py
# ================================================================
# Modelo v2 para detección de violencia
# Mejoras vs v1:
#   1. Top-2 personas + 8 features de interacción (110 total)
#   2. MultiHeadAttention temporal
#   3. Augmentación (ruido, jitter, flip)
#   4. LayerNorm de entrada
# ================================================================
import os, json
import numpy as np
from pathlib import Path
from app.pipeline import pool_frame_to_110, frame_visible_v2, smooth_window

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import keras
from keras import layers, callbacks
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, precision_recall_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════
NPZ_DIRS = [
    r"out_npz/rwf_train",
    r"out_npz/rwf_val"
    #,r"out_npz\ubi",
]
SEQ_LEN      = 32
STRIDE       = 8        # stride menor = más ventanas de entrenamiento
TRIM_BORDERS = 25
MIN_VIS_FRAC = 0.35
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
SEED         = 42
EPOCHS       = 120
BATCH_SIZE   = 64
MAX_RATIO    = 3.0      # neg/pos en balanceo

N_PERSONS    = 2        # top-2 personas
N_JOINTS     = 17
N_INTER      = 8        # features de interacción entre personas
N_FEATURES   = N_PERSONS * N_JOINTS * 3 + N_INTER  # 102 + 8 = 110

OUT_DIR = Path("./models_mix3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)

# Índices COCO relevantes
_NOSE                    = 0
_WRIST_L, _WRIST_R       = 9, 10
_HIP_L,   _HIP_R         = 11, 12
_SHOULDER_L, _SHOULDER_R = 5, 6


def align_skeletons_temporally(kps: np.ndarray) -> np.ndarray:
    """
    Offline Tracker: Alinea los esqueletos comparando las distancias entre
    el frame actual y el frame anterior para evitar teletransportes.
    """
    aligned = kps.copy()
    F, K, J, _ = aligned.shape
    
    for t in range(1, F):
        prev = aligned[t-1]
        curr = aligned[t].copy()
        
        # Función para sacar el centro (X, Y) de una persona
        def get_center(person):
            valid = person[:, 2] > 0
            if not valid.any(): return np.array([1e6, 1e6]) # Lejos si no es válido
            return person[valid, :2].mean(axis=0)
            
        prev_centers = [get_center(prev[i]) for i in range(K)]
        curr_centers = [get_center(curr[i]) for i in range(K)]
        
        new_curr = np.zeros_like(curr)
        used_j = set()
        
        # Emparejar a los que ya existían en el frame anterior
        for i in range(K):
            if prev_centers[i][0] == 1e6: continue # No había nadie aquí
            
            best_j, best_dist = -1, 1e5
            for j in range(K):
                if j not in used_j:
                    dist = np.linalg.norm(prev_centers[i] - curr_centers[j])
                    if dist < best_dist:
                        best_dist, best_j = dist, j
            
            # Si se movió menos de 300 píxeles, asumimos que es la misma persona
            if best_j != -1 and best_dist < 300:
                new_curr[i] = curr[best_j]
                used_j.add(best_j)
                
        # Llenar los espacios vacíos con las personas nuevas/restantes
        empty_i = [i for i in range(K) if np.all(new_curr[i] == 0)]
        unused_j = [j for j in range(K) if j not in used_j]
        for i, j in zip(empty_i, unused_j):
            new_curr[i] = curr[j]
            
        aligned[t] = new_curr
        
    return aligned


# ══════════════════════════════════════════════════════════
# AUGMENTACIÓN
# ══════════════════════════════════════════════════════════
# Pares L/R para flip horizontal
_FLIP_PAIRS = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]

def flip_sequence(X: np.ndarray) -> np.ndarray:
    """Espejo horizontal: intercambia joints L/R e invierte x."""
    aug = X.copy()
    for pi in range(N_PERSONS):
        base = pi * N_JOINTS * 3
        # Invertir x de todos los joints de esta persona
        for j in range(N_JOINTS):
            idx = base + j * 3
            aug[:, idx] = 1.0 - aug[:, idx]   # x → 1 - x
        # Intercambiar pares L/R
        for (j1, j2) in _FLIP_PAIRS:
            i1, i2 = base + j1 * 3, base + j2 * 3
            tmp = aug[:, i1:i1+3].copy()
            aug[:, i1:i1+3] = aug[:, i2:i2+3]
            aug[:, i2:i2+3] = tmp
    # Las features de interacción son distancias Euclídeas → flip-invariantes
    return aug


def augment_sequence(X: np.ndarray) -> np.ndarray:
    """Aplica augmentaciones aleatorias a una ventana (T, 110)."""
    aug = X.copy()

    # 1. Ruido gaussiano en posiciones (no en confianza)
    if np.random.random() < 0.6:
        noise = np.random.normal(0, 0.012, aug.shape).astype(np.float32)
        conf_cols = [i for i in range(N_FEATURES) if i % 3 == 2]
        noise[:, conf_cols] = 0.0
        aug = np.clip(aug + noise, 0.0, 1.0)

    # 2. Flip horizontal
    if np.random.random() < 0.5:
        aug = flip_sequence(aug)

    # 3. Jitter temporal (velocidad aleatoria 0.8× – 1.2×)
    if np.random.random() < 0.4:
        T = aug.shape[0]
        speed  = np.random.uniform(0.8, 1.2)
        idx    = np.clip(np.round(np.arange(T) * speed).astype(int), 0, T - 1)
        aug    = aug[idx]

    return aug


# ══════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════
def load_meta(z):
    m = z["meta"]
    try:
        return json.loads(m.item() if hasattr(m, "item") else m)
    except Exception:
        return {}


def video_to_windows(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    if not {"kps", "labels_aligned", "meta"}.issubset(z.files):
        return (np.zeros((0, SEQ_LEN, N_FEATURES), np.float32),
                np.zeros((0,), np.int64), npz_path.name, npz_path.parent.name)

    kps  = z["kps"]
    kps = align_skeletons_temporally(kps)

    y    = z["labels_aligned"].astype(np.int64)
    meta = load_meta(z)
    W, H = int(meta.get("width", 1920)), int(meta.get("height", 1080))
    F = kps.shape[0]

    if F != len(y) or F < SEQ_LEN:
        return (np.zeros((0, SEQ_LEN, N_FEATURES), np.float32),
                np.zeros((0,), np.int64), npz_path.name, npz_path.parent.name)

    # Recorte de bordes para videos con etiqueta global
    is_vid_level = meta.get("label_type", "video") == "video"
    if is_vid_level and meta.get("video_level_label", 0) == 1:
        if TRIM_BORDERS > 0 and F > 2 * TRIM_BORDERS:
            y = y.copy()
            y[:TRIM_BORDERS]  = 0
            y[-TRIM_BORDERS:] = 0

    # Extraer features frame a frame
    feats = np.zeros((F, N_FEATURES), np.float32)
    vis   = np.zeros(F, np.float32)
    kps = align_skeletons_temporally(kps)
    for t in range(F):
        feats[t] = pool_frame_to_110(kps[t], W, H, is_tracked=True)
        vis[t]   = 1.0 if frame_visible_v2(kps[t]) else 0.0

    # Ventanas deslizantes
    xw_list, yw_list = [], []
    for s in range(0, F - SEQ_LEN + 1, STRIDE):
        e = s + SEQ_LEN
        if vis[s:e].mean() < MIN_VIS_FRAC:
            continue
        xw_list.append(smooth_window(feats[s:e]))
        yw_list.append(int(y[e - 1]))

    if not xw_list:
        return (np.zeros((0, SEQ_LEN, N_FEATURES), np.float32),
                np.zeros((0,), np.int64), npz_path.name, npz_path.parent.name)

    return (np.stack(xw_list).astype(np.float32),
            np.array(yw_list, np.int64),
            npz_path.name, npz_path.parent.name)


def stratified_split_by_domain(vid_ids, domains, val_ratio, test_ratio, seed=SEED):
    rng = np.random.RandomState(seed)
    vid_ids, domains = np.array(vid_ids), np.array(domains)
    tr, va, te = set(), set(), set()
    for d in np.unique(domains):
        mask = domains == d
        vids = np.unique(vid_ids[mask]).copy()
        rng.shuffle(vids)
        n    = len(vids)
        n_te = int(round(n * test_ratio))
        n_va = int(round(n * val_ratio))
        te.update(vids[:n_te])
        va.update(vids[n_te : n_te + n_va])
        tr.update(vids[n_te + n_va :])
    return tr, va, te


def balance_undersample(X, y, max_ratio=MAX_RATIO, seed=SEED):
    pos = np.nonzero(y == 1)[0]
    neg = np.nonzero(y == 0)[0]
    max_neg = int(len(pos) * max_ratio)
    if len(neg) <= max_neg:
        return X, y
    rng  = np.random.default_rng(seed)
    keep = np.sort(np.concatenate([pos, rng.choice(neg, max_neg, replace=False)]))
    print(f"  Balanceo: {len(neg)} neg → {max_neg} neg  (ratio {max_ratio}:1, pos={len(pos)})")
    return X[keep], y[keep]


# ══════════════════════════════════════════════════════════
# MODELO  (BiLSTM + MultiHeadAttention)
# ══════════════════════════════════════════════════════════
def build_model(seq_len=SEQ_LEN, n_features=N_FEATURES):
    """
    Input(32, 110)
     → LayerNorm
     → Conv1D(64) + BN + ReLU + Dropout
     → BiLSTM(64, return_seq=True)
     → MultiHeadAttention(4 heads) + residual + LayerNorm
     → BiLSTM(32, return_seq=False)
     → Dense(64) → Dense(1, sigmoid)
    """
    reg = keras.regularizers.l2(1e-4)
    inp = layers.Input(shape=(seq_len, n_features))

    # Normalización de entrada (estabiliza entrenamiento con features heterogéneos)
    x = layers.LayerNormalization()(inp)

    # Bloque convolucional (captura patrones locales en T)
    x = layers.Conv1D(64, kernel_size=3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.25)(x)
    x = layers.SpatialDropout1D(0.10)(x)

    # BiLSTM 1 — contexto temporal bidireccional
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True,
                    recurrent_dropout=0.20, kernel_regularizer=reg)
    )(x)
    x = layers.Dropout(0.25)(x)

    # Atención multi-cabeza — aprende qué frames son relevantes
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.10)(x, x)
    x    = layers.Add()([x, attn])          # residual
    x    = layers.LayerNormalization()(x)

    # BiLSTM 2 — comprime la secuencia a un vector
    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=False,
                    recurrent_dropout=0.20, kernel_regularizer=reg)
    )(x)

    # Cabeza clasificadora
    x   = layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)
    x   = layers.Dropout(0.25)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inp, out)


# ══════════════════════════════════════════════════════════
# HELPERS EVALUACIÓN
# ══════════════════════════════════════════════════════════
def find_best_threshold(y_true, y_proba):
    prec, rec, thrs = precision_recall_curve(y_true, y_proba)
    f1s     = 2 * prec * rec / (prec + rec + 1e-9)
    best    = int(np.argmax(f1s))
    best_t  = float(thrs[best]) if best < len(thrs) else 0.5
    return best_t, float(f1s[best])


def plot_history(history, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.plot(history.history["loss"],     label="train")
    a1.plot(history.history["val_loss"], label="val")
    a1.set_title("Loss"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(history.history["auc"],      label="train")
    a2.plot(history.history["val_auc"],  label="val")
    a2.set_title("AUC");  a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(str(path), dpi=150); plt.close(fig)


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── 1. Cargar y construir ventanas ──────────────────
    paths = []
    for d in NPZ_DIRS:
        paths += sorted(Path(d).glob("*.npz"))
    assert paths, "No se encontraron archivos .npz"

    x_list, y_list, vids, doms = [], [], [], []
    for i, p in enumerate(paths):
        if (i + 1) % 100 == 0:
            print(f"  Cargando [{i+1}/{len(paths)}]...")
        xw, yw, vid, dom = video_to_windows(p)
        if len(xw):
            x_list.append(xw); y_list.append(yw)
            vids += [vid] * len(yw)
            doms += [dom] * len(yw)

    X = np.concatenate(x_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int64)
    vids, doms = np.array(vids), np.array(doms)
    print(f"Ventanas totales: {len(y)}  |  Pos={y.sum()} ({y.mean():.3f})")

    # ── 2. Split sin fuga de video ──────────────────────
    tr_ids, va_ids, te_ids = stratified_split_by_domain(
        vids, doms, VAL_RATIO, TEST_RATIO
    )
    m_tr = np.isin(vids, list(tr_ids))
    m_va = np.isin(vids, list(va_ids))
    m_te = np.isin(vids, list(te_ids))
    x_tr, y_tr = X[m_tr], y[m_tr]
    x_va, y_va = X[m_va], y[m_va]
    x_te, y_te = X[m_te], y[m_te]
    print(f"Split → train={len(y_tr)}  val={len(y_va)}  test={len(y_te)}")

    # ── 3. Balanceo ─────────────────────────────────────
    x_tr, y_tr = balance_undersample(x_tr, y_tr)

    # ── 4. Augmentación (solo positivos del train) ──────
    print("Augmentando ejemplos positivos...")
    pos_idx = np.where(y_tr == 1)[0]
    aug_X = [x_tr, ]
    aug_y = [y_tr, ]
    for idx in pos_idx:
        aug_X.append(augment_sequence(x_tr[idx])[np.newaxis])
        aug_y.append(np.array([1], dtype=np.int64))
    x_tr = np.concatenate(aug_X, axis=0)
    y_tr = np.concatenate(aug_y, axis=0)
    perm = np.random.permutation(len(y_tr))
    x_tr, y_tr = x_tr[perm], y_tr[perm]
    print(f"Train post-augment: {len(y_tr)}  |  Pos={y_tr.sum()} ({y_tr.mean():.3f})")

    # ── 5. Normalización z-score (fit sobre train) ──────
    mu = x_tr.reshape(-1, N_FEATURES).mean(axis=0).astype(np.float32)
    sd = x_tr.reshape(-1, N_FEATURES).std(axis=0).astype(np.float32)
    sd[sd < 1e-6] = 1.0

    def normalize(arr):
        s = arr.shape
        return ((arr.reshape(-1, N_FEATURES) - mu) / sd).reshape(s).astype(np.float32)

    x_tr_n = normalize(x_tr)
    x_va_n = normalize(x_va)
    x_te_n = normalize(x_te)
    np.savez(OUT_DIR / "v2_norm_stats.npz", mean=mu, std=sd)
    print("Norm stats guardadas.")

    # ── 6. Pesos de clase ────────────────────────────────
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    class_weight = {0: len(y_tr) / (2 * n_neg),
                    1: len(y_tr) / (2 * n_pos)}

    # ── 7. Modelo ────────────────────────────────────────
    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss=keras.losses.BinaryFocalCrossentropy(
            gamma=2.0, label_smoothing=0.05
        ),
        metrics=[keras.metrics.AUC(name="auc")],
    )
    model.summary()

    # ── 8. Callbacks ─────────────────────────────────────
    cb = [
        callbacks.EarlyStopping(
            monitor="val_auc", patience=20, mode="max",
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_auc", factor=0.5, patience=8,
            mode="max", min_lr=1e-6, verbose=1,
        ),
    ]

    # ── 9. Entrenamiento ─────────────────────────────────
    history = model.fit(
        x_tr_n, y_tr,
        validation_data=(x_va_n, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=cb,
        verbose=1,
    )

    # ── 10. Guardar modelo ───────────────────────────────
    model_path = OUT_DIR / "v2_bilstm_attention.keras"
    model.save(str(model_path))
    print(f"\nModelo guardado: {model_path}")

    # ── 11. Evaluación completa ──────────────────────────
    print("\n" + "=" * 60 + "\nEVALUACIÓN\n" + "=" * 60)
    all_metrics = {}
    for split_name, x_s, y_s in [("VAL", x_va_n, y_va), ("TEST", x_te_n, y_te)]:
        proba = model.predict(x_s, verbose=0).ravel()
        roc   = roc_auc_score(y_s, proba)
        pr    = average_precision_score(y_s, proba)
        best_thr, best_f1 = find_best_threshold(y_s, proba)
        pred  = (proba >= best_thr).astype(int)
        print(f"\n── {split_name} ({len(y_s)} ventanas, pos={y_s.sum()}) ──")
        print(f"  ROC-AUC : {roc:.4f}")
        print(f"  PR-AUC  : {pr:.4f}")
        print(f"  Best-F1 : {best_f1:.4f}  @ thr={best_thr:.3f}")
        print(classification_report(y_s, pred,
              target_names=["Normal", "Violencia"], digits=4))
        all_metrics[split_name] = {
            "roc_auc": float(roc), "pr_auc": float(pr),
            "best_threshold": float(best_thr), "best_f1": float(best_f1),
        }

    # ── 12. Guardar métricas y threshold ─────────────────
    (OUT_DIR / "v2_metrics.json").write_text(
        json.dumps(all_metrics, indent=2)
    )
    thr_data = {"best_threshold": all_metrics["TEST"]["best_threshold"],
                "best_f1":        all_metrics["TEST"]["best_f1"]}
    (OUT_DIR / "v2_threshold.json").write_text(json.dumps(thr_data, indent=2))
    plot_history(history, OUT_DIR / "v2_training_history.png")
    print("\n✓ Entrenamiento v2 completado.")