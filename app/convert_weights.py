#!/usr/bin/env python
# convert_weights.py  –  Conversión ONE-TIME de pesos Keras → PyTorch
#
# Uso:
#   CUDA_VISIBLE_DEVICES=-1 python convert_weights.py
#
# Genera:  models_mix2/mix_cnn_lstm_T32_F51.pt
#
# Solo necesitas ejecutarlo UNA VEZ.  Después el servidor usa solo PyTorch.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # evita el segfault de TF/CUDA

import sys
from pathlib import Path

import numpy as np
import torch

# Asegurarse de que app/ esté en el path
sys.path.insert(0, str(Path(__file__).parent))
from app.torch_model import CnnBiLstmClassifier


# ══════════════════════════════════════════════════════════════════════════════
# Helpers de conversión de pesos
# ══════════════════════════════════════════════════════════════════════════════

def _t(arr: np.ndarray) -> torch.Tensor:
    """ndarray float32 → torch float32."""
    return torch.from_numpy(arr.astype(np.float32).copy())


def copy_conv1d(keras_layer, sd: dict, prefix: str):
    """
    Keras Conv1D kernel: (kernel_size, in_ch, out_ch)
    PyTorch Conv1d weight: (out_ch, in_ch, kernel_size)
    """
    kernel, bias = keras_layer.get_weights()          # (3, 51, 48), (48,)
    sd[f'{prefix}.weight'] = _t(kernel.transpose(2, 1, 0))
    sd[f'{prefix}.bias']   = _t(bias)


def copy_batchnorm(keras_layer, sd: dict, prefix: str):
    """
    Keras BN devuelve: [gamma, beta, moving_mean, moving_var]
    PyTorch BN espera: weight=gamma, bias=beta, running_mean, running_var
    """
    gamma, beta, rmean, rvar = keras_layer.get_weights()
    sd[f'{prefix}.weight']       = _t(gamma)
    sd[f'{prefix}.bias']         = _t(beta)
    sd[f'{prefix}.running_mean'] = _t(rmean)
    sd[f'{prefix}.running_var']  = _t(rvar)
    sd[f'{prefix}.num_batches_tracked'] = torch.tensor(0, dtype=torch.long)


def copy_bilstm(keras_layer, sd: dict, prefix: str):
    """
    Keras Bidirectional devuelve 6 arrays:
      [fw_kernel(I,4H), fw_rec(H,4H), fw_bias(4H),
       bw_kernel(I,4H), bw_rec(H,4H), bw_bias(4H)]

    Keras gate order : [i, f, c, o]  ← igual que PyTorch [i, f, g, o]
    → solo hace falta transponer, no reordenar gates.

    PyTorch BiLSTM state_dict keys:
      weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
      weight_ih_l0_reverse, weight_hh_l0_reverse, ...
    """
    weights = keras_layer.get_weights()
    if len(weights) != 6:
        raise ValueError(
            f"Se esperaban 6 arrays de {keras_layer.name}, "
            f"se obtuvieron {len(weights)}.  "
            "Verifica que la capa sea Bidirectional(LSTM)."
        )
    fw_k, fw_r, fw_b, bw_k, bw_r, bw_b = weights

    # Forward
    sd[f'{prefix}.weight_ih_l0']         = _t(fw_k.T)   # (4H, I)
    sd[f'{prefix}.weight_hh_l0']         = _t(fw_r.T)   # (4H, H)
    sd[f'{prefix}.bias_ih_l0']           = _t(fw_b)
    sd[f'{prefix}.bias_hh_l0']           = torch.zeros_like(_t(fw_b))

    # Backward
    sd[f'{prefix}.weight_ih_l0_reverse'] = _t(bw_k.T)
    sd[f'{prefix}.weight_hh_l0_reverse'] = _t(bw_r.T)
    sd[f'{prefix}.bias_ih_l0_reverse']   = _t(bw_b)
    sd[f'{prefix}.bias_hh_l0_reverse']   = torch.zeros_like(_t(bw_b))


def copy_dense(keras_layer, sd: dict, prefix: str):
    """
    Keras Dense kernel: (in, out)
    PyTorch Linear weight: (out, in)
    """
    kernel, bias = keras_layer.get_weights()
    sd[f'{prefix}.weight'] = _t(kernel.T)
    sd[f'{prefix}.bias']   = _t(bias)


# ══════════════════════════════════════════════════════════════════════════════
# Función principal
# ══════════════════════════════════════════════════════════════════════════════

def convert(keras_path: str = "models_mix2/mix_cnn_lstm_T32_F51.keras",
            output_path: str = "models_mix2/mix_cnn_lstm_T32_F51.pt"):

    print("═" * 60)
    print("  Conversión Keras → PyTorch")
    print("═" * 60)

    # ── Cargar Keras ──────────────────────────────────────────────
    print("\n[1/5] Cargando modelo Keras...")
    from keras.models import load_model
    km = load_model(keras_path, compile=False)
    print(f"      OK  ({keras_path})")

    # ── Crear modelo PyTorch vacío ────────────────────────────────
    print("[2/5] Creando modelo PyTorch...")
    pt_model = CnnBiLstmClassifier()
    sd = pt_model.state_dict()

    # ── Copiar pesos capa por capa ────────────────────────────────
    print("[3/5] Transfiriendo pesos...")

    copy_conv1d(km.get_layer('conv1d'),             sd, 'conv1')
    copy_batchnorm(km.get_layer('batch_normalization'), sd, 'bn1')
    copy_bilstm(km.get_layer('bidirectional'),      sd, 'lstm1')
    copy_bilstm(km.get_layer('bidirectional_1'),    sd, 'lstm2')
    copy_dense(km.get_layer('dense'),               sd, 'fc1')
    copy_dense(km.get_layer('dense_1'),             sd, 'fc2')

    pt_model.load_state_dict(sd)
    pt_model.eval()
    print("      OK")

    # ── Validar salidas (mismo input → misma predicción) ─────────
    print("[4/5] Validando equivalencia con Keras...")

    rng = np.random.default_rng(42)
    dummy_np = rng.random((1, 32, 51)).astype(np.float32)

    # Keras predict
    p_keras = float(km.predict(dummy_np, verbose=0).ravel()[0])

    # PyTorch predict
    with torch.no_grad():
        p_torch = float(pt_model(torch.from_numpy(dummy_np)).item())

    diff = abs(p_keras - p_torch)
    print(f"      Keras:   {p_keras:.6f}")
    print(f"      PyTorch: {p_torch:.6f}")
    print(f"      Diff:    {diff:.2e}  {'✓ OK' if diff < 1e-4 else '⚠ REVISAR'}")

    if diff >= 1e-4:
        print("\n  ⚠  Diferencia mayor a 1e-4.  Puede deberse a:")
        print("     · Activación de la capa Dense (ajustar en torch_model.py)")
        print("     · Orden de gates LSTM no estándar en el modelo entrenado")
        print("  → El modelo puede seguir funcionando; verifica con datos reales.")

    # ── Guardar ───────────────────────────────────────────────────
    print(f"[5/5] Guardando pesos en {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(pt_model.state_dict(), output_path)
    print(f"      ✓ Listo.\n")

    print("═" * 60)
    print("  Ahora edita load_artifacts() en app/pipeline.py")
    print("  y elimina la dependencia de Keras/TensorFlow.")
    print("═" * 60)
    return pt_model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--keras",  default="models_mix2/mix_cnn_lstm_T32_F51.keras")
    parser.add_argument("--output", default="models_mix2/mix_cnn_lstm_T32_F51.pt")
    args = parser.parse_args()
    convert(args.keras, args.output)
