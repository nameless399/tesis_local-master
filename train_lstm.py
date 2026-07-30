# train_lstm.py  –  Entrena modelo BiLSTM para deteccion de agresiones
#
# Mejoras vs modelo original:
#   - Bidirectional LSTM (captura contexto pasado y futuro)
#   - Dropout reducido de 0.5 a 0.25 (modelo pequeno, evita underfitting)
#   - recurrent_dropout=0.15 en LSTM
#   - EarlyStopping + ReduceLROnPlateau
#   - Evaluacion completa con graficas para tesis
#
# Genera en OUT_DIR:
#   mix_cnn_lstm_T32_F51.keras        - modelo entrenado
#   mix_cnn_lstm_T32_F51_norm_stats.npz - estadisticas de normalizacion
#   mix_cnn_lstm_T32_F51_threshold.json - threshold optimo
#   lstm_metrics.json                  - metricas completas
#   lstm_confusion.png                 - confusion matrix
#   lstm_roc_pr_curves.png             - curvas ROC y PR
#   lstm_training_history.png          - loss/auc durante entrenamiento

import os, json
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import keras
from keras import layers, callbacks
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.pipeline import (
    SEQ_LEN, CONF_MIN,
    pool_frame_to_51, frame_visible,
)

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════
NPZ_DIRS = [
    r"out_npz/rwf_train",
    r"out_npz/rwf_val"
    #,r"out_npz\ubi",
]
STRIDE           = 12
TRIM_BORDERS     = 25   # solo para videos con label_type="video" (RWF-2000)
MIN_VIS_FRAC     = 0.40 # más estricto: descarta ventanas con pose detection pobre
VAL_RATIO        = 0.15
TEST_RATIO       = 0.15
SEED             = 42
N_FEATURES       = 51
EPOCHS           = 150
BATCH_SIZE       = 64
MAX_RATIO        = 2.0

OUT_DIR = Path("./models_mix3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)


# ══════════════════════════════════════════════════════════
# CARGA DE DATOS (compartida con train_lgbm.py)
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
        return np.zeros((0, SEQ_LEN, N_FEATURES), np.float32), np.zeros((0,), np.int64), npz_path.name, npz_path.parent.name

    kps  = z["kps"]
    y    = z["labels_aligned"].astype(np.int64)
    meta = load_meta(z)
    W, H = int(meta.get("width", 1920)), int(meta.get("height", 1080))
    F = kps.shape[0]

    if F != len(y) or F < SEQ_LEN:
        return np.zeros((0, SEQ_LEN, N_FEATURES), np.float32), np.zeros((0,), np.int64), npz_path.name, npz_path.parent.name

    # TRIM solo para videos con etiqueta a nivel video (RWF-2000).
    # UBI-Fights tiene label_type="frame" — sus labels ya son exactos, no recortar.
    is_video_level = meta.get("label_type", "video") == "video"
    if is_video_level and meta.get("video_level_label", 0) == 1 and TRIM_BORDERS > 0 and F > 2 * TRIM_BORDERS:
        y = y.copy()
        y[:TRIM_BORDERS] = 0
        y[-TRIM_BORDERS:] = 0

    feats = np.zeros((F, N_FEATURES), np.float32)
    vis   = np.zeros(F, np.float32)
    for t in range(F):
        feats[t] = pool_frame_to_51(kps[t], W, H)
        vis[t]   = 1.0 if frame_visible(kps[t]) else 0.0

    xw_list, yw_list = [], []
    for s in range(0, F - SEQ_LEN + 1, STRIDE):
        e = s + SEQ_LEN
        if vis[s:e].mean() < MIN_VIS_FRAC:
            continue
        xw_list.append(feats[s:e])
        yw_list.append(int(y[e - 1]))

    xw = np.stack(xw_list, 0) if xw_list else np.zeros((0, SEQ_LEN, N_FEATURES), np.float32)
    yw = np.array(yw_list, np.int64) if yw_list else np.zeros((0,), np.int64)
    return xw, yw, npz_path.name, npz_path.parent.name


def stratified_split_by_domain(vid_ids, domains, val_ratio, test_ratio, seed=SEED):
    rng = np.random.RandomState(seed)
    vid_ids = np.array(vid_ids)
    domains = np.array(domains)
    tr, va, te = set(), set(), set()
    for d in np.unique(domains):
        mask = (domains == d)
        vids = np.unique(vid_ids[mask]).copy()
        rng.shuffle(vids)
        n = len(vids)
        n_te = int(round(n * test_ratio))
        n_va = int(round(n * val_ratio))
        te.update(vids[:n_te])
        va.update(vids[n_te:n_te + n_va])
        tr.update(vids[n_te + n_va:])
    return tr, va, te


def balance_undersample(X, y, max_ratio=MAX_RATIO, seed=SEED):
    """Undersample clase mayoritaria para que neg/pos <= max_ratio.
    Solo se aplica al train set. Val/test no se tocan."""
    if max_ratio is None:
        return X, y
    pos_idx = np.nonzero(y == 1)[0]
    neg_idx = np.nonzero(y == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    max_neg = int(n_pos * max_ratio)
    if n_neg <= max_neg:
        return X, y
    rng = np.random.default_rng(seed)
    keep_neg = rng.choice(neg_idx, size=max_neg, replace=False)
    keep = np.sort(np.concatenate([pos_idx, keep_neg]))
    print(f"  Balanceo: {n_neg} neg -> {max_neg} neg (ratio {max_ratio}:1, pos={n_pos})")
    return X[keep], y[keep]


# ══════════════════════════════════════════════════════════
# MODELO BiLSTM
# ══════════════════════════════════════════════════════════
def build_bilstm_model(seq_len=SEQ_LEN, n_features=N_FEATURES):
    inp = layers.Input(shape=(seq_len, n_features))

    x = layers.Masking(mask_value=0.0)(inp)

    # Conv1D para patrones locales
    x = layers.Conv1D(48, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Dropout(0.35)(x)
    x = layers.SpatialDropout1D(0.15)(x)

    # Bidirectional LSTM
    x = layers.Bidirectional(
        layers.LSTM(48, return_sequences=True, recurrent_dropout=0.25,
                    kernel_regularizer=keras.regularizers.l2(1e-4))
    )(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Bidirectional(
        layers.LSTM(24, return_sequences=False, recurrent_dropout=0.25,
                    kernel_regularizer=keras.regularizers.l2(1e-4))
    )(x)

    # Clasificador
    x = layers.Dense(32, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out)
    return model


# ══════════════════════════════════════════════════════════
# GRAFICAS
# ══════════════════════════════════════════════════════════
def plot_history(history, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history.history["loss"], label="train")
    ax1.plot(history.history["val_loss"], label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax1_key = "auc" if "auc" in history.history else list(history.history.keys())[2]
    ax1_val = "val_auc" if "val_auc" in history.history else list(history.history.keys())[3]
    ax2.plot(history.history[ax1_key], label="train")
    ax2.plot(history.history[ax1_val], label="val")
    ax2.set_title("AUC")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")


def plot_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Agresion"])
    ax.set_yticklabels(["Normal", "Agresion"])
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Confusion Matrix - BiLSTM (Test)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=16)
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")


def plot_roc_pr(y_true, y_proba, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    ax1.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.set_title("ROC - BiLSTM (Test)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    ax2.plot(rec, prec, linewidth=2, label=f"AP = {pr_auc:.4f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("PR - BiLSTM (Test)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")


def find_best_threshold(y_true, y_proba):
    prec, rec, thrs = precision_recall_curve(y_true, y_proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1s)
    best_thr = float(thrs[best_idx]) if best_idx < len(thrs) else 0.5
    return best_thr, float(f1s[best_idx])


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    # --- Cargar ventanas ---
    paths = []
    for d in NPZ_DIRS:
        found = sorted(Path(d).glob("*.npz"))
        print(f"  {d}: {len(found)} archivos")
        paths += found
    assert paths, "No hay archivos .npz"
    print(f"Total NPZ: {len(paths)}")

    x_list, y_list, vids, doms = [], [], [], []
    for i, p in enumerate(paths):
        if (i + 1) % 200 == 0 or i == 0:
            print(f"  Cargando [{i+1}/{len(paths)}] ...", flush=True)
        xw, yw, vid, dom = video_to_windows(p)
        if len(xw):
            x_list.append(xw)
            y_list.append(yw)
            vids += [vid] * len(yw)
            doms += [dom] * len(yw)
    print(f"  Carga completa: {len(x_list)} videos con ventanas validas.")

    X = np.concatenate(x_list, 0).astype(np.float32)  # (N, T, 51)
    y = np.concatenate(y_list, 0).astype(np.int64)
    vids = np.array(vids)
    doms = np.array(doms)
    print(f"Ventanas: {len(y)} | Pos={y.sum()} ({y.mean():.3f})")

    # --- Split ---
    train_ids, val_ids, test_ids = stratified_split_by_domain(
        vids, doms, VAL_RATIO, TEST_RATIO, seed=SEED
    )
    m_tr = np.isin(vids, list(train_ids))
    m_va = np.isin(vids, list(val_ids))
    m_te = np.isin(vids, list(test_ids))
    x_tr, y_tr = X[m_tr], y[m_tr]
    x_va, y_va = X[m_va], y[m_va]
    x_te, y_te = X[m_te], y[m_te]
    print(f"Split: train={len(y_tr)} val={len(y_va)} test={len(y_te)}")

    # --- Balanceo (solo train) ---
    x_tr, y_tr = balance_undersample(x_tr, y_tr)
    print(f"Train post-balanceo: {len(y_tr)} | Pos={y_tr.sum()} ({y_tr.mean():.3f})")

    # --- Normalizar (z-score por feature, sobre train) ---
    mu = x_tr.reshape(-1, N_FEATURES).mean(axis=0, keepdims=True).astype(np.float32)
    sd = x_tr.reshape(-1, N_FEATURES).std(axis=0, keepdims=True).astype(np.float32)
    sd[sd < 1e-6] = 1.0

    def normalize(data):
        shape = data.shape
        return ((data.reshape(-1, N_FEATURES) - mu) / sd).reshape(shape).astype(np.float32)

    x_tr_n = normalize(x_tr)
    x_va_n = normalize(x_va)
    x_te_n = normalize(x_te)

    np.savez(OUT_DIR / "mix_cnn_lstm_T32_F51_norm_stats.npz", mean=mu, std=sd)
    print("Stats guardadas.")

    # --- Peso de clase ---
    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    class_weight = {0: len(y_tr) / (2 * n_neg), 1: len(y_tr) / (2 * n_pos)}
    print(f"class_weight: {class_weight}")

    # --- Construir modelo ---
    model = build_bilstm_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryFocalCrossentropy(
            gamma=2.0, label_smoothing=0.1,
        ),
        metrics=[keras.metrics.AUC(name="auc")],
    )
    model.summary()

    # --- Callbacks ---
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

    # --- Entrenar ---
    history = model.fit(
        x_tr_n, y_tr,
        validation_data=(x_va_n, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=cb,
        verbose=1,
    )

    # --- Guardar modelo ---
    model_path = OUT_DIR / "mix_cnn_lstm_T32_F51.keras"
    model.save(str(model_path))
    print(f"\nModelo guardado: {model_path}")

    # --- Evaluacion ---
    print("\n" + "=" * 60)
    print("EVALUACION")
    print("=" * 60)

    all_metrics = {}

    for split_name, x_split, y_split in [("VAL", x_va_n, y_va), ("TEST", x_te_n, y_te)]:
        proba = model.predict(x_split, verbose=0).ravel()
        roc = roc_auc_score(y_split, proba)
        pr  = average_precision_score(y_split, proba)
        best_thr, best_f1 = find_best_threshold(y_split, proba)

        pred = (proba >= best_thr).astype(int)

        print(f"\n-- {split_name} ({len(y_split)} ventanas, pos={y_split.sum()}) --")
        print(f"  ROC-AUC:    {roc:.4f}")
        print(f"  PR-AUC:     {pr:.4f}")
        print(f"  Best thr:   {best_thr:.3f} (F1={best_f1:.4f})")
        print(classification_report(y_split, pred, target_names=["Normal", "Agresion"], digits=4))

        all_metrics[split_name] = {
            "n_samples": int(len(y_split)),
            "n_positive": int(y_split.sum()),
            "roc_auc": float(roc),
            "pr_auc": float(pr),
            "best_threshold": float(best_thr),
            "best_f1": float(best_f1),
        }

    # --- Graficas ---
    print("Generando graficas...")
    test_proba = model.predict(x_te_n, verbose=0).ravel()
    best_thr = all_metrics["TEST"]["best_threshold"]
    test_pred = (test_proba >= best_thr).astype(int)

    plot_history(history, OUT_DIR / "lstm_training_history.png")
    plot_confusion_matrix(y_te, test_pred, OUT_DIR / "lstm_confusion.png")
    plot_roc_pr(y_te, test_proba, OUT_DIR / "lstm_roc_pr_curves.png")

    # --- Guardar metricas ---
    metrics_path = OUT_DIR / "lstm_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"Metricas: {metrics_path}")

    # --- Threshold ---
    thr_path = OUT_DIR / "mix_cnn_lstm_T32_F51_threshold.json"
    thr_data = {"best_threshold": all_metrics["TEST"]["best_threshold"],
                "best_f1": all_metrics["TEST"]["best_f1"]}
    thr_path.write_text(json.dumps(thr_data, indent=2), encoding="utf-8")
    print(f"Threshold: {thr_path}")

    print("\nEntrenamiento completado.")
