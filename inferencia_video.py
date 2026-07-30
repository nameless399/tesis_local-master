# inferencia_video.py
# Ejecuta inferencia (LSTM + opcional LGBM) sobre un video y guarda MP4 anotado + CSV.

import csv, time, collections, warnings
from pathlib import Path
from typing import List

import numpy as np
import cv2

from app.pipeline import (
    SEQ_LEN, CONF_MIN, MIN_VIS_FRAC, HYST_GAP,
    pool_frame_to_51, frame_visible, pool_scores,
    predict_window, load_artifacts,
)

# =========================
# CONFIG (edita aquí)
# =========================
MODELS_DIR   = Path("./models_mix")
VIDEO_IN     = Path(r"C:\Users\Usuario\Documents\GitHub\tesis-deteccionar-sistema\V_27.mp4")
VIDEO_OUT    = Path(r"C:\Users\Usuario\Documents\GitHub\tesis-deteccionar-sistema\V_222.mp4")
CSV_OUT      = VIDEO_OUT.with_suffix(".csv")

POSE_WEIGHTS = "yolo11m-pose.pt"
IMGSZ        = 920
CONF_POSE    = 0.25
IOU_POSE     = 0.50
TOPK_PERSONS = 4

STRIDE       = 1       # procesa 1 frame cada STRIDE (2 = saltea 1)

# Fusión con LGBM
FUSION_W     = 0.50    # 0 -> solo LSTM,  1 -> solo LGBM

# Pooling de scores a nivel video
POOL_METHOD  = "topk"  # "max" | "mean" | "topk"
TOPK_FRAC    = 0.20

# Warnings ruidosos
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# =========================
# Inferencia sobre video
# =========================
def run_video():
    KERAS, MU, SD, THR_ON, THR_OFF, LGBM, POSE, STACKER = load_artifacts(MODELS_DIR, POSE_WEIGHTS)

    cap = cv2.VideoCapture(str(VIDEO_IN))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir: {VIDEO_IN}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(VIDEO_OUT), fourcc, fps, (W, H), True)

    win_feats: collections.deque = collections.deque(maxlen=SEQ_LEN)
    win_vis: collections.deque   = collections.deque(maxlen=SEQ_LEN)
    video_scores: List[float]    = []
    on_state = False

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as csv_f:
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["frame_idx", "time_sec", "p_win", "p_vid", "on"])

        fidx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fidx += 1
            if STRIDE > 1 and (fidx % STRIDE != 0):
                vw.write(frame)
                continue

            # --- Pose ---
            res = POSE.predict(frame, imgsz=IMGSZ, conf=CONF_POSE, iou=IOU_POSE,
                               verbose=False, half=False)[0]

            kps_f = None
            if (res.keypoints is not None
                    and res.keypoints.xy is not None
                    and res.keypoints.xy.shape[0] > 0):
                xy = res.keypoints.xy.detach().cpu().numpy()
                c  = getattr(res.keypoints, "confidence", None) or getattr(res.keypoints, "conf", None)
                if c is not None:
                    c = c.detach().cpu().numpy()
                else:
                    c = np.ones(xy.shape[:2], dtype=np.float32)
                order = np.argsort(-c.mean(axis=1))
                P = min(len(order), TOPK_PERSONS)
                xy, c = xy[order[:P]], c[order[:P]]
                kps_f = np.concatenate([xy, c[..., None]], axis=-1).astype(np.float32)

            # --- Ventana ---
            feat51 = pool_frame_to_51(kps_f, W, H)
            vis = frame_visible(kps_f, CONF_MIN)
            win_feats.append(feat51)
            win_vis.append(1.0 if vis else 0.0)

            p_win = 0.0
            p_vid = 0.0
            if len(win_feats) == SEQ_LEN:
                vis_frac = np.mean(list(win_vis))
                if vis_frac >= MIN_VIS_FRAC:
                    Xw = np.stack(list(win_feats), axis=0)
                    p_win = predict_window(Xw, KERAS, MU, SD, lgbm=LGBM, fusion_w=FUSION_W, stacker=STACKER)
                    video_scores.append(p_win)
                    p_vid = pool_scores(video_scores, pool=POOL_METHOD, topk_frac=TOPK_FRAC)

                    if (not on_state) and p_vid >= THR_ON:
                        on_state = True
                    elif on_state and p_vid <= THR_OFF:
                        on_state = False

            # --- Render ---
            try:
                show = res.plot() if res is not None else frame
            except Exception:
                show = frame

            cv2.putText(show, f"p_win={p_win:.2f}  p_vid={p_vid:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
            if on_state:
                cv2.putText(show, "ALERTA", (20, 80),
                            cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 255), 3)

            vw.write(show)

            tsec = fidx / max(fps, 1e-6)
            csv_w.writerow([fidx, f"{tsec:.3f}", f"{p_win:.6f}", f"{p_vid:.6f}", int(on_state)])

    vw.release()
    cap.release()
    print(f"[DONE] Video: {VIDEO_OUT}")
    print(f"[DONE] CSV  : {CSV_OUT}")


if __name__ == "__main__":
    run_video()
