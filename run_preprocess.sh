#!/bin/bash

WEIGHTS="yolo11s-pose.pt"
COMMON="--clahe --skip_existing --imgsz 640 --conf 0.25 --topk 4"

echo "============================================================"
echo " PASO 1/5: RWF-2000 train/Fight"
echo "============================================================"

python3 preprocess_videos.py \
  --src_dir "RWF-2000/train/Fight" \
  --out_dir "out_npz/rwf_train" \
  --label 1 \
  --weights $WEIGHTS \
  $COMMON

echo "============================================================"
echo " PASO 2/5: RWF-2000 train/NonFight"
echo "============================================================"

python3 preprocess_videos.py \
  --src_dir "RWF-2000/train/NonFight" \
  --out_dir "out_npz/rwf_train" \
  --label 0 \
  --weights $WEIGHTS \
  $COMMON

echo "============================================================"
echo " PASO 3/5: RWF-2000 val/Fight"
echo "============================================================"

python3 preprocess_videos.py \
  --src_dir "RWF-2000/val/Fight" \
  --out_dir "out_npz/rwf_val" \
  --label 1 \
  --weights $WEIGHTS \
  $COMMON

echo "============================================================"
echo " PASO 4/5: RWF-2000 val/NonFight"
echo "============================================================"

python3 preprocess_videos.py \
  --src_dir "RWF-2000/val/NonFight" \
  --out_dir "out_npz/rwf_val" \
  --label 0 \
  --weights $WEIGHTS \
  $COMMON

echo "============================================================"
echo " PASO 5/5: UBI-Fights"
echo "============================================================"

python3 preprocess_videos.py \
  --src_dir "UBI_FIGHTS/videos" \
  --out_dir "out_npz/ubi" \
  --annotation_dir "UBI_FIGHTS/annotation" \
  --weights $WEIGHTS \
  $COMMON

echo "============================================================"
echo " PREPROCESAMIENTO COMPLETO"
echo "============================================================"

echo "Entrenando LSTM..."
python3 train_lstm.py

echo "Entrenando LightGBM..."
python3 train_lgbm.py

echo "Entrenando Stacker..."
python3 train_stacker.py

echo "TODO TERMINADO"
