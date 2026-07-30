@echo off
REM ============================================================
REM  Preprocesamiento completo: RWF-2000 + UBI-Fights
REM  Ejecutar desde la raiz del proyecto con el venv activado.
REM ============================================================

set WEIGHTS=yolo11s-pose.pt
set COMMON=--clahe --skip_existing --imgsz 640 --conf 0.25 --topk 4

echo ============================================================
echo  PASO 1/5: RWF-2000 train/Fight (label=1)
echo ============================================================
python preprocess_videos.py --src_dir "RWF-2000\train\Fight" --out_dir "out_npz\rwf_train" --label 1 --weights %WEIGHTS% %COMMON%

echo ============================================================
echo  PASO 2/5: RWF-2000 train/NonFight (label=0)
echo ============================================================
python preprocess_videos.py --src_dir "RWF-2000\train\NonFight" --out_dir "out_npz\rwf_train" --label 0 --weights %WEIGHTS% %COMMON%

echo ============================================================
echo  PASO 3/5: RWF-2000 val/Fight (label=1)
echo ============================================================
python preprocess_videos.py --src_dir "RWF-2000\val\Fight" --out_dir "out_npz\rwf_val" --label 1 --weights %WEIGHTS% %COMMON%

echo ============================================================
echo  PASO 4/5: RWF-2000 val/NonFight (label=0)
echo ============================================================
python preprocess_videos.py --src_dir "RWF-2000\val\NonFight" --out_dir "out_npz\rwf_val" --label 0 --weights %WEIGHTS% %COMMON%

echo ============================================================
echo  PASO 5/5: UBI-Fights (frame-level CSV)
echo ============================================================
python preprocess_videos.py --src_dir "UBI_FIGHTS\videos" --out_dir "out_npz\ubi" --annotation_dir "UBI_FIGHTS\annotation" --weights %WEIGHTS% %COMMON%

echo ============================================================
echo  PREPROCESAMIENTO COMPLETO
echo  NPZ guardados en: out_npz\rwf_train, out_npz\rwf_val, out_npz\ubi
echo.
echo  Siguiente paso: entrenar modelos
echo    python train_lstm.py
echo    python train_lgbm.py
echo    python train_stacker.py
echo ============================================================
pause
