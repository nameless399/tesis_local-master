import pickle
from pathlib import Path

# Ruta a una de tus anotaciones .pkl
PKL_PATH = Path(r"C:\Users\Usuario\Documents\GitHub\tesis-deteccionar-sistema\CHAD\CHAD_Meta\annotations") / "1_001_0.pkl"  # cambia por un nombre real

# Leer archivo .pkl
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

print(f"Frames totales: {len(data)}")

# Mostrar primeros frames con contenido real
shown = 0
for frame_num, frame_data in data.items():
    if not frame_data:
        continue

    print(f"\n🟩 Frame {frame_num}:")
    for person_id, (bbox, keypoints) in frame_data.items():
        print(f"  👤 Person ID: {person_id}")
        print(f"  Bounding box shape: {bbox.shape}  → {bbox}")
        print(f"  Keypoints shape: {keypoints.shape}")
        print(f"  Primeros 9 valores: {keypoints[:9]}")
        shown += 1
        break  # solo primera persona del frame

    if shown >= 3:
        break  # mostrar 3 frames máximo
