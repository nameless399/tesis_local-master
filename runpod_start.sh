#!/usr/bin/env bash
# Bootstrap del servidor de inferencia en RunPod tras un Stop/Start.
# Instala tmux + deps Python, corrige opencv, lanza uvicorn en tmux
# y verifica que /health responde. Falla rapido si algo va mal.
#
# Uso (dentro del pod, por SSH):
#   cd /workspace/app && git pull && bash runpod_start.sh
set -euo pipefail

APP_DIR="/workspace/app"
MODELS_DIR="/workspace/models_mix2"
POSE_WEIGHTS="/workspace/yolo11s-pose.pt"
LOG_FILE="/workspace/infer.log"
PORT="${PORT:-8000}"
SESSION="infer"

# Umbrales de disparo de alerta (sobrescriben el threshold.json del modelo).
# THR_ON  -> probabilidad minima del pooling para disparar alerta
# THR_OFF -> probabilidad por debajo de la cual se resetea el estado
# Subir estos valores = menos falsos positivos / alertas mas "duras".
THR_ON="${THR_ON:-0.65}"
THR_OFF="${THR_OFF:-0.50}"

log() { echo "[runpod_start] $*"; }
fail() { echo "[runpod_start][ERROR] $*" >&2; exit 1; }

# ── 0) Sanity checks ─────────────────────────────────────
[ -d "$APP_DIR" ] || fail "APP_DIR no existe: $APP_DIR"
[ -d "$MODELS_DIR" ] || fail "MODELS_DIR no existe: $MODELS_DIR"
[ -f "$POSE_WEIGHTS" ] || fail "POSE_WEIGHTS no existe: $POSE_WEIGHTS"

cd "$APP_DIR"

# ── 1) tmux ──────────────────────────────────────────────
if ! command -v tmux >/dev/null 2>&1; then
  log "apt: instalando tmux"
  apt-get update -qq
  apt-get install -y -qq tmux >/dev/null
fi

# ── 2) Dependencias Python ───────────────────────────────
log "pip: instalando deps de inferencia"
pip install --no-cache-dir -r requirements-inference.txt

# opencv-python (con GUI) y opencv-python-headless comparten el mismo
# paquete cv2. Si ultralytics instala opencv-python por encima del
# headless, TF 2.21 segfaultea al abrir VideoCapture. Reinstalamos
# headless limpio.
log "pip: reinstalando opencv-python-headless limpio"
pip uninstall -y opencv-python opencv-python-headless >/dev/null 2>&1 || true
pip install --no-cache-dir opencv-python-headless==4.10.0.84

# ── 3) Verificar imports criticos ────────────────────────
log "verificando imports (cv2, tensorflow, ultralytics, keras)"
python - <<'PY' || fail "imports rotos — revisa el stack arriba"
import cv2
import tensorflow as tf
import ultralytics
import keras
print(f"  cv2={cv2.__version__}  tf={tf.__version__}  "
      f"ultralytics={ultralytics.__version__}  keras={keras.__version__}")
PY

# ── 4) Git pull (best-effort) ────────────────────────────
log "git pull"
git pull --ff-only || log "git pull fallo (ok si es local)"

# ── 5) Lanzar uvicorn en tmux ────────────────────────────
log "matando sesion tmux previa si existe"
tmux kill-session -t "$SESSION" 2>/dev/null || true

log "lanzando uvicorn en tmux ($SESSION) puerto $PORT"
: > "$LOG_FILE"
tmux new -d -s "$SESSION" \
  "cd $APP_DIR && MODELS_DIR=$MODELS_DIR POSE_WEIGHTS=$POSE_WEIGHTS THR_ON=$THR_ON THR_OFF=$THR_OFF uvicorn app.server_inference:app --host 0.0.0.0 --port $PORT 2>&1 | tee $LOG_FILE"

# ── 6) Health check con reintentos ───────────────────────
log "esperando a que /health responda (hasta 90s)"
ok=0
for i in $(seq 1 45); do
  if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
    ok=1
    break
  fi
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    break
  fi
  sleep 2
done

if [ "$ok" != "1" ]; then
  echo
  echo "── Ultimas lineas de $LOG_FILE ──"
  tail -60 "$LOG_FILE" || true
  fail "uvicorn no respondio /health. Revisa el log completo: $LOG_FILE"
fi

log "OK — /health responde"
curl -s "http://localhost:$PORT/health"
echo
log "listo. tmux attach -t $SESSION  (Ctrl+B luego D para salir)"
