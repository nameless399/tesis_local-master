# preprocess_videos.py
# Extrae keypoints YOLO11-pose de videos crudos y los guarda como .npz
# Soporta videos de baja calidad: oscuridad, blur, movimiento rápido.
#
# Dos modos de etiquetado:
#   Video-level (RWF-2000):  --label 1   o  --label 0
#   Frame-level (UBI-Fights):  --annotation_dir ruta/csv/
#
# Uso RWF-2000:
#   python preprocess_videos.py --src_dir RWF-2000/train/Fight    --out_dir out_npz/rwf_train --label 1 --clahe
#   python preprocess_videos.py --src_dir RWF-2000/train/NonFight --out_dir out_npz/rwf_train --label 0 --clahe
#
# Uso UBI-Fights:
#   python preprocess_videos.py --src_dir UBI_FIGHTS/videos --out_dir out_npz/ubi --annotation_dir UBI_FIGHTS/annotation --clahe

import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}

# ═══════════════════════════════════════════════════════════
# Preprocesamiento de imagen
# ═══════════════════════════════════════════════════════════
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def is_dark(frame: np.ndarray, threshold: float = 40.0) -> bool:
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()) < threshold


# ═══════════════════════════════════════════════════════════
# Lectura de anotación CSV (UBI-Fights)
# ═══════════════════════════════════════════════════════════
def load_annotation_csv(csv_path: Path) -> np.ndarray:
    """Lee CSV con un valor 0/1 por línea. Retorna array int64."""
    vals = []
    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                vals.append(int(line))
    return np.array(vals, dtype=np.int64)


# ═══════════════════════════════════════════════════════════
# Extracción de keypoints de un resultado YOLO
# ═══════════════════════════════════════════════════════════
def _extract_frame_kps(res, topk: int, w: int, h: int) -> np.ndarray:
    frame_kps = np.zeros((topk, 17, 3), dtype=np.float32)
    if (res.keypoints is None
            or res.keypoints.xy is None
            or res.keypoints.xy.shape[0] == 0):
        return frame_kps

    xy = res.keypoints.xy.detach().cpu().numpy()
    c = getattr(res.keypoints, "confidence", None) or getattr(res.keypoints, "conf", None)
    c = c.detach().cpu().numpy() if c is not None else np.ones(xy.shape[:2], dtype=np.float32)

    order = np.argsort(-c.mean(axis=1))
    p = min(len(order), topk)

    xy_norm = xy[order[:p]].copy()
    xy_norm[..., 0] /= max(w, 1)
    xy_norm[..., 1] /= max(h, 1)
    xy_norm = np.clip(xy_norm, 0.0, 1.0)

    frame_kps[:p] = np.concatenate([xy_norm, c[order[:p]][..., None]], axis=-1)
    return frame_kps


# ═══════════════════════════════════════════════════════════
# Procesar un video → .npz
# ═══════════════════════════════════════════════════════════
def _resolve_label(fidx: int, label: int | None, frame_labels: np.ndarray | None) -> int:
    """Devuelve el label para el frame fidx según el modo de etiquetado."""
    if frame_labels is not None and fidx < len(frame_labels):
        return int(frame_labels[fidx])
    if label is not None:
        return label
    return 0


def _read_frames(cap, pose_model, topk, w, h, imgsz, conf, iou,
                 use_clahe, stride, dark_thresh, label, frame_labels):
    """Itera frames del video y retorna (kps_list, labels_list, dark_count, clahe_count)."""
    kps_list, labels_list = [], []
    dark_count = clahe_count = 0
    fidx = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fidx += 1

        if stride > 1 and (fidx % stride != 0):
            continue

        dark = is_dark(frame, dark_thresh)
        dark_count += int(dark)
        if use_clahe and dark:
            frame = apply_clahe(frame)
            clahe_count += 1

        res = pose_model.predict(
            frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False, half=True,
        )[0]

        kps_list.append(_extract_frame_kps(res, topk, w, h))
        labels_list.append(_resolve_label(fidx, label, frame_labels))

    return kps_list, labels_list, dark_count, clahe_count


def process_video(
    video_path: Path,
    out_dir: Path,
    pose_model,
    imgsz: int,
    conf: float,
    iou: float,
    topk: int,
    use_clahe: bool,
    stride: int,
    dark_thresh: float = 40.0,
    label: int | None = None,
    frame_labels: np.ndarray | None = None,
) -> dict:
    """Extrae keypoints y guarda .npz con labels video-level o frame-level."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "error", "reason": "no_open", "path": str(video_path)}

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    kps_list, labels_list, dark_count, clahe_count = _read_frames(
        cap, pose_model, topk, W, H, imgsz, conf, iou,
        use_clahe, stride, dark_thresh, label, frame_labels,
    )
    cap.release()

    if not kps_list:
        return {"status": "empty", "path": str(video_path)}

    kps_arr = np.stack(kps_list, axis=0)
    labels_arr = np.array(labels_list, dtype=np.int64)
    F = kps_arr.shape[0]

    fight_frac = float(labels_arr.sum()) / max(F, 1)
    vid_label = 1 if fight_frac > 0.0 else 0

    meta = json.dumps({
        "width": W, "height": H, "fps": fps,
        "n_frames": F, "n_frames_total": n_total,
        "video_level_label": vid_label,
        "fight_frame_pct": round(fight_frac * 100, 1),
        "stride": stride,
        "clahe_applied": clahe_count,
        "dark_frames": dark_count,
        "source": video_path.name,
        "label_type": "frame" if frame_labels is not None else "video",
    })

    out_path = out_dir / (video_path.stem + ".npz")
    np.savez_compressed(out_path, kps=kps_arr, labels_aligned=labels_arr, meta=meta)

    return {
        "status": "ok", "out": str(out_path), "frames": F,
        "fight_pct": round(fight_frac * 100, 1),
        "dark_pct": round(100 * dark_count / max(F, 1), 1),
        "clahe_pct": round(100 * clahe_count / max(F, 1), 1),
        "vid_label": vid_label,
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocesa videos a .npz con keypoints YOLO pose",
        epilog="Requiere --label O --annotation_dir (no ambos).",
    )
    parser.add_argument("--src_dir", required=True, help="Directorio con videos")
    parser.add_argument("--out_dir", required=True, help="Directorio de salida .npz")

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--label", type=int, choices=[0, 1],
                     help="Etiqueta video-level: 0=normal, 1=agresión (RWF-2000)")
    grp.add_argument("--annotation_dir", type=str,
                     help="Directorio con CSV frame-level (UBI-Fights)")

    parser.add_argument("--weights", default="yolo11s-pose.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--clahe", action="store_true", help="CLAHE en frames oscuros")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--dark_thr", type=float, default=40.0)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def _resolve_video_labels(vp: Path, ann_dir: Path | None, fixed_label: int | None):
    """Retorna (frame_labels, vid_label, skip_reason)."""
    if ann_dir is None:
        return None, fixed_label, None
    csv_path = ann_dir / (vp.stem + ".csv")
    if not csv_path.exists():
        return None, None, "no_csv"
    return load_annotation_csv(csv_path), None, None


def _print_result(stats: dict, frame_level: bool):
    if stats["status"] != "ok":
        print(f"FAIL ({stats.get('reason', stats['status'])})")
        return
    fight_info = f"fight={stats['fight_pct']}%" if frame_level else f"label={stats['vid_label']}"
    print(f"OK  F={stats['frames']}  {fight_info}  dark={stats['dark_pct']}%")


def main():
    args = _parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = Path(args.annotation_dir) if args.annotation_dir else None

    videos = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print(f"[ERROR] No se encontraron videos en: {src_dir}")
        return

    mode = "frame-level (CSV)" if ann_dir else f"video-level (label={args.label})"
    print(f"[INFO] {len(videos)} videos en {src_dir}")
    print(f"[INFO] Modo: {mode}  |  CLAHE={'on' if args.clahe else 'off'}")
    print(f"[INFO] YOLO: {args.weights}  imgsz={args.imgsz}  conf={args.conf}")
    print()

    from ultralytics import YOLO
    pose_model = YOLO(args.weights)

    ok_count = err_count = skip_count = no_ann = 0
    total_dark = []

    for i, vp in enumerate(videos, 1):
        if args.skip_existing and (out_dir / (vp.stem + ".npz")).exists():
            print(f"  [{i:04d}/{len(videos)}] SKIP  {vp.name}")
            skip_count += 1
            continue

        frame_labels, vid_label, skip_reason = _resolve_video_labels(vp, ann_dir, args.label)
        if skip_reason:
            print(f"  [{i:04d}/{len(videos)}] NO_ANN  {vp.name}")
            no_ann += 1
            continue

        print(f"  [{i:04d}/{len(videos)}] {vp.name} ...", end=" ", flush=True)

        stats = process_video(
            video_path=vp, out_dir=out_dir, pose_model=pose_model,
            imgsz=args.imgsz, conf=args.conf, iou=args.iou, topk=args.topk,
            use_clahe=args.clahe, stride=args.stride, dark_thresh=args.dark_thr,
            label=vid_label, frame_labels=frame_labels,
        )

        _print_result(stats, frame_labels is not None)
        if stats["status"] == "ok":
            ok_count += 1
            total_dark.append(stats["dark_pct"])
        else:
            err_count += 1

    print()
    print("=" * 60)
    print(f"OK={ok_count}  ERR={err_count}  SKIP={skip_count}  NO_ANN={no_ann}")
    if total_dark:
        print(f"Frames oscuros promedio: {np.mean(total_dark):.1f}%")
    print(f"NPZ en: {out_dir}")


if __name__ == "__main__":
    main()
