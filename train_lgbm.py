# train_lgbm.py  –  Entrena LGBMClassifier con evaluacion completa para tesis
#
# Genera en OUT_DIR:
#   lgbm_model.pkl          - modelo entrenado
#   lgbm_metrics.json       - metricas (ROC-AUC, PR-AUC, F1, precision, recall)
#   lgbm_confusion.png      - confusion matrix
#   lgbm_roc_pr_curves.png  - curvas ROC y Precision-Recall
#   lgbm_feature_importance.png - top 30 features por importancia

import os, json, warnings
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve, f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.pipeline import (
    SEQ_LEN, CONF_MIN,
    pool_frame_to_51, frame_visible, featurize_T51,
    PAIR_DISTS, ANGLE_JOINTS,
)

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════
NPZ_DIRS = [
    r"out_npz/rwf_train",
    r"out_npz/rwf_val"
    #,r"out_npz\ubi",
]
STRIDE        = 12
TRIM_BORDERS  = 25
MIN_VIS_FRAC  = 0.40
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15
SEED          = 42
MAX_RATIO     = 2.0

OUT_DIR = Path("./models_mix3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PKL = OUT_DIR / "lgbm_model.pkl"


# ══════════════════════════════════════════════════════════
# HELPERS
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

    kps  = z["kps"]
    y    = z["labels_aligned"].astype(np.int64)
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
    vis   = np.zeros((F,), np.float32)
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

    xw = np.stack(xw_list, 0) if xw_list else np.zeros((0, SEQ_LEN, 51), np.float32)
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
# FEATURE NAMES (para interpretabilidad)
# ══════════════════════════════════════════════════════════
_JOINTS = ['nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear',
           'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow',
           'L_wrist', 'R_wrist', 'L_hip', 'R_hip',
           'L_knee', 'R_knee', 'L_ankle', 'R_ankle']

def build_feature_names():
    names = []
    for stat in ["mean", "std", "min", "max"]:
        for j in _JOINTS:
            names.append(f"x_{j}_{stat}")
    for stat in ["mean", "std", "min", "max"]:
        for j in _JOINTS:
            names.append(f"y_{j}_{stat}")
    for stat in ["mean", "std", "min", "max"]:
        for j in _JOINTS:
            names.append(f"vel_{j}_{stat}")
    for stat in ["mean", "std", "min", "max"]:
        for j in _JOINTS:
            names.append(f"conf_{j}_{stat}")
    for stat in ["mean", "std", "min", "max"]:
        for j in _JOINTS:
            names.append(f"acc_{j}_{stat}")
    for (i, j) in PAIR_DISTS:
        names.append(f"dist_{_JOINTS[i]}-{_JOINTS[j]}_mean")
        names.append(f"dist_{_JOINTS[i]}-{_JOINTS[j]}_std")
    for (a, v, b) in ANGLE_JOINTS:
        for stat in ["mean", "std", "min", "max"]:
            names.append(f"angle_{_JOINTS[a]}-{_JOINTS[v]}-{_JOINTS[b]}_{stat}")
    return names


# ══════════════════════════════════════════════════════════
# GRAFICAS PARA TESIS
# ══════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Agresion"])
    ax.set_yticklabels(["Normal", "Agresion"])
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Confusion Matrix - LGBM (Test)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=16)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")


def plot_roc_pr(y_true, y_proba, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    ax1.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve - LGBM (Test)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    ax2.plot(rec, prec, linewidth=2, label=f"AP = {pr_auc:.4f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve - LGBM (Test)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")


def plot_feature_importance(clf, feat_names, path, top_n=30):
    imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(range(top_n), imp[idx][::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feat_names[i] for i in idx][::-1], fontsize=8)
    ax.set_xlabel("Importancia (split)")
    ax.set_title(f"Top {top_n} Features - LGBM")
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")


def find_best_threshold(y_true, y_proba):
    """Busca threshold que maximiza F1."""
    prec, rec, thrs = precision_recall_curve(y_true, y_proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1s)
    best_thr = float(thrs[best_idx]) if best_idx < len(thrs) else 0.5
    return best_thr, float(f1s[best_idx])


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    # --- Carga de ventanas ---
    paths = []
    for d in NPZ_DIRS:
        paths += sorted(Path(d).glob("*.npz"))
    assert paths, "No hay .npz"

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
    vids = np.array(vids)
    doms = np.array(doms)
    print(f"Ventanas: {len(y)} | Pos={y.sum()} ({y.mean():.3f})")

    # --- Split estratificado por dominio y video ---
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

    # --- Featurizacion tabular ---
    def batch_feats(xb):
        if len(xb) == 0:
            return np.zeros((0, 1), np.float32)
        return np.concatenate([featurize_T51(w) for w in xb], axis=0)

    print("Featurizando...")
    f_tr = batch_feats(x_tr)
    f_va = batch_feats(x_va)
    f_te = batch_feats(x_te)
    feat_names = build_feature_names()
    print(f"Feature shape: {f_tr.shape} ({len(feat_names)} nombres)")

    # --- Pesos de clase ---
    classes = np.array([0, 1])
    cw = compute_class_weight("balanced", classes=classes, y=y_tr)
    scale_pos_weight = cw[1] / cw[0]
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    # --- Entrenamiento ---
    params = dict(
        objective="binary",
        boosting_type="gbdt",
        num_leaves=63,
        max_depth=-1,
        learning_rate=0.05,
        n_estimators=1200,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=SEED,
        n_jobs=-1,
        metric="None",
        scale_pos_weight=scale_pos_weight,
    )

    clf = lgb.LGBMClassifier(**params)
    clf.fit(
        f_tr, y_tr,
        eval_set=[(f_va, y_va)],
        eval_metric="auc",
    )

    # --- Evaluacion completa ---
    print("\n" + "=" * 60)
    print("EVALUACION")
    print("=" * 60)

    all_metrics = {}

    for split_name, f_split, y_split in [("VAL", f_va, y_va), ("TEST", f_te, y_te)]:
        proba = clf.predict_proba(f_split)[:, 1]
        roc = roc_auc_score(y_split, proba)
        pr  = average_precision_score(y_split, proba)
        best_thr, best_f1 = find_best_threshold(y_split, proba)

        pred_05 = (proba >= 0.5).astype(int)
        pred_best = (proba >= best_thr).astype(int)

        print(f"\n-- {split_name} ({len(y_split)} ventanas, pos={y_split.sum()}) --")
        print(f"  ROC-AUC:    {roc:.4f}")
        print(f"  PR-AUC:     {pr:.4f}")
        print(f"  Best thr:   {best_thr:.3f} (F1={best_f1:.4f})")
        print(f"\n  Classification Report (thr=0.5):")
        print(classification_report(y_split, pred_05, target_names=["Normal", "Agresion"], digits=4))
        print(f"  Classification Report (thr={best_thr:.3f}):")
        print(classification_report(y_split, pred_best, target_names=["Normal", "Agresion"], digits=4))

        all_metrics[split_name] = {
            "n_samples": int(len(y_split)),
            "n_positive": int(y_split.sum()),
            "roc_auc": float(roc),
            "pr_auc": float(pr),
            "best_threshold": float(best_thr),
            "best_f1": float(best_f1),
        }

    # --- Graficas ---
    print("\nGenerando graficas...")
    test_proba = clf.predict_proba(f_te)[:, 1]
    best_thr = all_metrics["TEST"]["best_threshold"]
    test_pred = (test_proba >= best_thr).astype(int)

    plot_confusion_matrix(y_te, test_pred, OUT_DIR / "lgbm_confusion.png")
    plot_roc_pr(y_te, test_proba, OUT_DIR / "lgbm_roc_pr_curves.png")
    plot_feature_importance(clf, feat_names, OUT_DIR / "lgbm_feature_importance.png")

    # --- Guardar modelo y metricas ---
    joblib.dump(clf, OUT_PKL)
    print(f"\nModelo: {OUT_PKL}")

    metrics_path = OUT_DIR / "lgbm_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"Metricas: {metrics_path}")

    # Actualizar threshold.json con el best threshold del test
    thr_path = OUT_DIR / "mix_cnn_lstm_T32_F51_threshold.json"
    thr_data = {}
    if thr_path.exists():
        thr_data = json.loads(thr_path.read_text(encoding="utf-8"))
    thr_data["best_threshold"] = all_metrics["TEST"]["best_threshold"]
    thr_data["best_f1"] = all_metrics["TEST"]["best_f1"]
    thr_path.write_text(json.dumps(thr_data, indent=2), encoding="utf-8")
    print(f"Threshold: {thr_path}")
