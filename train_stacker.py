# train_stacker.py  –  Entrena meta-learner (stacking) sobre predicciones LSTM + LGBM
#
# Ejecutar DESPUES de train_lstm.py y train_lgbm.py.
# Carga ambos modelos, genera predicciones OOF (out-of-fold) sobre el val set,
# y entrena un LogisticRegression como meta-learner.
#
# Genera:
#   models_mix/stacker.pkl              - meta-learner
#   models_mix/stacker_metrics.json     - metricas de la fusion

import os, json
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
from keras.models import load_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix, roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.pipeline import (
    SEQ_LEN, CONF_MIN,
    pool_frame_to_51, frame_visible, featurize_T51,
)

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════
NPZ_DIRS = [
    r"out_npz/rwf_train",
    r"out_npz/rwf_val"
    #,r"out_npz\ubi",
]
STRIDE       = 12
TRIM_BORDERS = 25
MIN_VIS_FRAC = 0.40
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
SEED         = 42

OUT_DIR = Path("./models_mix3")


# ══════════════════════════════════════════════════════════
# CARGA DE DATOS (misma logica que train_lgbm.py)
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
        return np.zeros((0, SEQ_LEN, 51), np.float32), np.zeros((0,), np.int64), npz_path.name, npz_path.parent.name
    kps = z["kps"]
    y = z["labels_aligned"].astype(np.int64)
    meta = load_meta(z)
    W, H = int(meta.get("width", 1920)), int(meta.get("height", 1080))
    F = kps.shape[0]
    if F != len(y) or F < SEQ_LEN:
        return np.zeros((0, SEQ_LEN, 51), np.float32), np.zeros((0,), np.int64), npz_path.name, npz_path.parent.name
    is_video_level = meta.get("label_type", "video") == "video"
    if is_video_level and meta.get("video_level_label", 0) == 1 and TRIM_BORDERS > 0 and F > 2 * TRIM_BORDERS:
        y = y.copy()
        y[:TRIM_BORDERS] = 0
        y[-TRIM_BORDERS:] = 0
    feats = np.zeros((F, 51), np.float32)
    vis = np.zeros(F, np.float32)
    for t in range(F):
        feats[t] = pool_frame_to_51(kps[t], W, H)
        vis[t] = 1.0 if frame_visible(kps[t]) else 0.0
    xw_list, yw_list = [], []
    for s in range(0, F - SEQ_LEN + 1, STRIDE):
        e = s + SEQ_LEN
        if vis[s:e].mean() < MIN_VIS_FRAC:
            continue
        xw_list.append(feats[s:e])
        yw_list.append(int(y[e - 1]))
    xw = np.stack(xw_list, 0) if xw_list else np.zeros((0, SEQ_LEN, 51), np.float32)
    yw = np.array(yw_list, np.int64) if yw_list else np.zeros((0,), np.int64)
    return xw, yw, npz_path.name, npz_path.parent.name


def stratified_split_by_domain(vid_ids, domains, val_ratio, test_ratio, seed=SEED):
    rng = np.random.RandomState(seed)
    vid_ids, domains = np.array(vid_ids), np.array(domains)
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


def find_best_threshold(y_true, y_proba):
    prec, rec, thrs = precision_recall_curve(y_true, y_proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1s)
    best_thr = float(thrs[best_idx]) if best_idx < len(thrs) else 0.5
    return best_thr, float(f1s[best_idx])


def plot_comparison(results, path):
    """Bar chart comparando LSTM, LGBM y Stacking."""
    models = list(results.keys())
    roc_vals = [results[m]["roc_auc"] for m in models]
    pr_vals = [results[m]["pr_auc"] for m in models]
    f1_vals = [results[m]["best_f1"] for m in models]

    x = np.arange(len(models))
    w = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, roc_vals, w, label="ROC-AUC")
    ax.bar(x, pr_vals, w, label="PR-AUC")
    ax.bar(x + w, f1_vals, w, label="Best F1")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Comparacion de modelos (Test Set)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for i, (r, p, f) in enumerate(zip(roc_vals, pr_vals, f1_vals)):
        ax.text(i - w, r + 0.01, f"{r:.3f}", ha="center", fontsize=8)
        ax.text(i, p + 0.01, f"{p:.3f}", ha="center", fontsize=8)
        ax.text(i + w, f + 0.01, f"{f:.3f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    # --- Cargar modelos ---
    print("Cargando modelos...")
    keras_model = load_model(str(OUT_DIR / "mix_cnn_lstm_T32_F51.keras"), compile=False)
    stats = np.load(OUT_DIR / "mix_cnn_lstm_T32_F51_norm_stats.npz")
    mu = stats["mean"].astype(np.float32)
    sd = stats["std"].astype(np.float32)

    lgbm = joblib.load(OUT_DIR / "lgbm_model.pkl")
    print("  LSTM y LGBM cargados.")

    # --- Cargar datos ---
    paths = []
    for d in NPZ_DIRS:
        paths += sorted(Path(d).glob("*.npz"))

    x_list, y_list, vids, doms = [], [], [], []
    for p in paths:
        xw, yw, vid, dom = video_to_windows(p)
        if len(xw):
            x_list.append(xw)
            y_list.append(yw)
            vids += [vid] * len(yw)
            doms += [dom] * len(yw)

    X = np.concatenate(x_list, 0).astype(np.float32)
    y = np.concatenate(y_list, 0).astype(np.int64)
    vids, doms = np.array(vids), np.array(doms)

    # --- Split (mismo seed = mismos splits que train_lstm y train_lgbm) ---
    train_ids, val_ids, test_ids = stratified_split_by_domain(
        vids, doms, VAL_RATIO, TEST_RATIO, seed=SEED
    )
    m_va = np.isin(vids, list(val_ids))
    m_te = np.isin(vids, list(test_ids))
    x_va, y_va = X[m_va], y[m_va]
    x_te, y_te = X[m_te], y[m_te]

    # --- Generar predicciones de cada modelo ---
    print("Generando predicciones...")

    def lstm_predict(x_raw):
        n = x_raw.shape[0]
        x_flat = x_raw.reshape(-1, 51)
        x_norm = ((x_flat - mu) / (sd + 1e-6)).reshape(n, SEQ_LEN, 51).astype(np.float32)
        return keras_model.predict(x_norm, verbose=0, batch_size=256).ravel()

    def lgbm_predict(x_raw):
        probs = []
        for i in range(len(x_raw)):
            feat = featurize_T51(x_raw[i])
            probs.append(lgbm.predict_proba(feat)[:, 1][0])
        return np.array(probs, dtype=np.float32)

    # Val set para entrenar stacker
    p_lstm_va = lstm_predict(x_va)
    p_lgbm_va = lgbm_predict(x_va)
    meta_va = np.column_stack([p_lstm_va, p_lgbm_va])

    # Test set para evaluar
    p_lstm_te = lstm_predict(x_te)
    p_lgbm_te = lgbm_predict(x_te)
    meta_te = np.column_stack([p_lstm_te, p_lgbm_te])

    # --- Entrenar stacker ---
    print("Entrenando stacker (LogisticRegression)...")
    stacker = LogisticRegression(random_state=SEED, max_iter=1000)
    stacker.fit(meta_va, y_va)

    coef = stacker.coef_[0]
    print(f"  Coeficientes: LSTM={coef[0]:.4f}, LGBM={coef[1]:.4f}")
    print(f"  Intercepto: {stacker.intercept_[0]:.4f}")

    # --- Evaluar los 3 enfoques en TEST ---
    print("\n" + "=" * 60)
    print("EVALUACION EN TEST SET")
    print("=" * 60)

    p_stacker_te = stacker.predict_proba(meta_te)[:, 1]

    results = {}
    for name, proba in [("LSTM", p_lstm_te), ("LGBM", p_lgbm_te), ("Stacking", p_stacker_te)]:
        roc = roc_auc_score(y_te, proba)
        pr = average_precision_score(y_te, proba)
        best_thr, best_f1 = find_best_threshold(y_te, proba)
        pred = (proba >= best_thr).astype(int)

        print(f"\n-- {name} --")
        print(f"  ROC-AUC:  {roc:.4f}")
        print(f"  PR-AUC:   {pr:.4f}")
        print(f"  Best thr: {best_thr:.3f} (F1={best_f1:.4f})")
        print(classification_report(y_te, pred, target_names=["Normal", "Agresion"], digits=4))

        results[name] = {
            "roc_auc": float(roc),
            "pr_auc": float(pr),
            "best_threshold": float(best_thr),
            "best_f1": float(best_f1),
        }

    # --- Guardar ---
    joblib.dump(stacker, OUT_DIR / "stacker.pkl")
    print(f"Stacker guardado: {OUT_DIR / 'stacker.pkl'}")

    metrics_path = OUT_DIR / "stacker_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Metricas: {metrics_path}")

    # --- Actualizar threshold con el mejor modelo ---
    best_model = max(results, key=lambda m: results[m]["best_f1"])
    thr_path = OUT_DIR / "mix_cnn_lstm_T32_F51_threshold.json"
    thr_data = {
        "best_threshold": results[best_model]["best_threshold"],
        "best_f1": results[best_model]["best_f1"],
        "best_model": best_model,
    }
    thr_path.write_text(json.dumps(thr_data, indent=2), encoding="utf-8")
    print(f"Threshold actualizado ({best_model}): {thr_path}")

    # --- Grafica comparativa ---
    plot_comparison(results, OUT_DIR / "model_comparison.png")

    print("\nStacking completado.")
