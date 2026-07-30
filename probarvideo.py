import os
import cv2
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("1. Importando librerías...")
try:
    from app.pipeline import load_artifacts
    print("✓ Librerías importadas.")
except Exception as e:
    print("✗ Error al importar pipeline:", e)
    exit(1)

print("\n2. Intentando cargar modelos (Keras, YOLO, LGBM)...")
try:
    # Esto simula la carga que hace el servidor
    artifacts = load_artifacts("models_mix3", "yolo11s-pose.pt")
    keras_model, mu, sd, thr_on, thr_off, lgbm, pose, stacker = artifacts
    print("✓ ¡Modelos cargados con éxito!")
except Exception as e:
    print("✗ ERROR AL CARGAR MODELOS:")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n3. Leyendo un frame de tu video...")
cap = cv2.VideoCapture("/var/home/ibra/Documents/tesis_local-master/app/video2.mp4")
ok, frame = cap.read()
cap.release()

if not ok:
    print("✗ No se pudo leer el frame.")
    exit(1)

print("\n4. Intentando predicción con YOLO11...")
try:
    res = pose.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
    print("✓ ¡YOLO predijo con éxito!")
    if res.keypoints is not None:
        print("   Keypoints detectados:", res.keypoints.xy.shape)
except Exception as e:
    print("✗ ERROR EN LA PREDICCIÓN DE YOLO:")
    import traceback
    traceback.print_exc()