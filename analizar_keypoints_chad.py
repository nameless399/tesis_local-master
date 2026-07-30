import numpy as np
import matplotlib.pyplot as plt
COCO_EDGES = [
    (5, 7), (7, 9),      # brazo izquierdo
    (6, 8), (8, 10),     # brazo derecho
    (5, 6),               # hombros
    (11, 13), (13, 15),  # pierna izquierda
    (12, 14), (14, 16),  # pierna derecha
    (11, 12),             # caderas
    (5, 11), (6, 12),     # torso
    (0, 1), (0, 2), (1, 3), (2, 4),  # cabeza
    (0, 5), (0, 6)       # cuello con hombros
]
import numpy as np
import matplotlib.pyplot as plt

# Keypoints (x, y, conf)
keypoints = np.array([[3.0312500e+02, 1.1466250e+03 ,8.6231703e-01],
 [2.9925000e+02, 1.1466250e+03, 7.4183702e-01],
 [2.9925000e+02, 1.1427500e+03, 8.5295546e-01],
 [3.0312500e+02 ,1.1233750e+03, 5.1366192e-01],
 [3.0312500e+02 ,1.1311250e+03, 7.6697665e-01],
 [3.2637500e+02, 1.1311250e+03, 6.9280899e-01],
 [3.3800000e+02 ,1.1195000e+03, 7.7480799e-01],
 [3.6512500e+02 ,1.1272500e+03, 5.7477069e-01],
 [3.7675000e+02 ,1.1233750e+03, 9.9702567e-01],
 [3.6125000e+02 ,1.1427500e+03, 8.2138664e-01],
 [3.6900000e+02 ,1.1582500e+03, 8.7461400e-01],
 [4.0000000e+02 ,1.1272500e+03 ,7.2475147e-01],
 [4.0000000e+02, 1.1117500e+03, 7.4884665e-01],
 [4.3487500e+02 ,1.1388750e+03, 8.9376348e-01],
 [4.5425000e+02 ,1.1078750e+03, 8.1414264e-01],
 [4.8525000e+02, 1.1272500e+03, 8.2274061e-01],
 [4.9687500e+02, 1.1001250e+03, 8.1566066e-01]])

# Conexiones COCO (pares de índices)
COCO_EDGES = [
    (5, 7), (7, 9),      # brazo izquierdo
    (6, 8), (8, 10),     # brazo derecho
    (5, 6),               # hombros
    (11, 13), (13, 15),  # pierna izquierda
    (12, 14), (14, 16),  # pierna derecha
    (11, 12),             # caderas
    (5, 11), (6, 12),     # torso
    (0, 1), (0, 2), (1, 3), (2, 4),  # cabeza
    (0, 5), (0, 6)       # cuello con hombros
]

# Filtrar puntos válidos (confianza > 0.3)
valid = keypoints[:, 2] > 0.3
x = keypoints[:, 0]
y = keypoints[:, 1]

# Dibujar
plt.figure(figsize=(6, 6))
plt.scatter(x[valid], y[valid], c='red', s=40, label="Keypoints")

# Dibujar líneas del esqueleto
for (i, j) in COCO_EDGES:
    if valid[i] and valid[j]:
        plt.plot([x[i], x[j]], [y[i], y[j]], 'b-', linewidth=2)

# Etiquetas opcionales
for i, (xi, yi) in enumerate(zip(x, y)):
    if valid[i]:
        plt.text(xi + 3, yi - 3, str(i), fontsize=9, color='black')

# Ajustes visuales
plt.gca().invert_yaxis()  # importante para coordenadas de imagen
plt.title("Pose humana - Frame 1573, Persona 3")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()
