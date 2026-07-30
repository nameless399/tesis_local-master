# guarda esto como verificar_datos.py y córrelo
import numpy as np
from pathlib import Path

DIRS = ["out_npz/rwf_train", "out_npz/rwf_val"]

total, positivos, negativos = 0, 0, 0

for d in DIRS:
    archivos = list(Path(d).glob("*.npz"))
    print(f"\n{d}: {len(archivos)} archivos")
    for p in archivos:
        z = np.load(p, allow_pickle=True)
        y = z["labels_aligned"]
        total     += len(y)
        positivos += int(y.sum())
        negativos += int((y == 0).sum())

print(f"\n{'='*40}")
print(f"Frames totales : {total:,}")
print(f"Positivos      : {positivos:,}  ({100*positivos/total:.1f}%)")
print(f"Negativos      : {negativos:,}  ({100*negativos/total:.1f}%)")
print(f"Ratio neg/pos  : {negativos/positivos:.1f}:1")