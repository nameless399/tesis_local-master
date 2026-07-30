# app/torch_model.py  –  Equivalente PyTorch de mix_cnn_lstm_T32_F51.keras
#
# Arquitectura replicada del summary de Keras:
#   Input(32, 51)
#   → Conv1D(48, k=3, same) → BN → ReLU → Dropout → SpatialDropout1D
#   → BiLSTM(48, return_seq=True) → Dropout
#   → BiLSTM(24, return_seq=False)
#   → Dense(32, relu) → Dropout
#   → Dense(1, sigmoid)

import torch
import torch.nn as nn


class CnnBiLstmClassifier(nn.Module):
    """
    Input:  (batch, T=32, F=51)
    Output: (batch, 1)  – probabilidad sigmoid [0, 1]
    """

    def __init__(self,
                 seq_len: int = 32,
                 n_features: int = 51,
                 conv_filters: int = 48,
                 conv_kernel: int = 3,
                 lstm1_units: int = 48,   # BiLSTM → salida 96
                 lstm2_units: int = 24,   # BiLSTM → salida 48
                 dense_units: int = 32,
                 dropout: float = 0.3,
                 spatial_dropout: float = 0.1):
        super().__init__()

        # ── Bloque Conv ──────────────────────────────────────────────────────
        # padding=1 con kernel=3 reproduce el 'same' de Keras para T=32
        self.conv1  = nn.Conv1d(n_features, conv_filters,
                                kernel_size=conv_kernel, padding=1)
        self.bn1    = nn.BatchNorm1d(conv_filters)
        self.drop1  = nn.Dropout(p=dropout)
        # SpatialDropout1D ≡ Dropout2d: desactiva canales completos
        self.sdrop  = nn.Dropout2d(p=spatial_dropout)

        # ── BiLSTM 1 (return_sequences=True) ────────────────────────────────
        self.lstm1  = nn.LSTM(conv_filters, lstm1_units,
                              batch_first=True, bidirectional=True)
        self.drop2  = nn.Dropout(p=dropout)

        # ── BiLSTM 2 (return_sequences=False → último paso) ──────────────────
        self.lstm2  = nn.LSTM(lstm1_units * 2, lstm2_units,
                              batch_first=True, bidirectional=True)

        # ── Cabeza densa ──────────────────────────────────────────────────────
        self.fc1    = nn.Linear(lstm2_units * 2, dense_units)
        self.drop3  = nn.Dropout(p=dropout)
        self.fc2    = nn.Linear(dense_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=32, F=51)

        # ── Conv block ──
        h = x.transpose(1, 2)       # (B, 51, 32)
        h = self.conv1(h)            # (B, 48, 32)
        h = self.bn1(h)
        h = torch.relu(h)
        h = self.drop1(h)
        # SpatialDropout1D sobre canales: Dropout2d espera (B, C, H, W)
        h = h.unsqueeze(-1)          # (B, 48, 32, 1)
        h = self.sdrop(h)
        h = h.squeeze(-1)           # (B, 48, 32)
        h = h.transpose(1, 2)       # (B, 32, 48)

        # ── BiLSTM 1 ──
        h, _ = self.lstm1(h)         # (B, 32, 96)
        h = self.drop2(h)

        # ── BiLSTM 2 ──
        h, _ = self.lstm2(h)         # (B, 32, 48)
        h = h[:, -1, :]              # (B, 48)  último timestep

        # ── Dense head ──
        h = torch.relu(self.fc1(h))  # (B, 32)
        h = self.drop3(h)
        h = torch.sigmoid(self.fc2(h))  # (B, 1)
        return h


def load_torch_model(pt_path: str, device: str = "cpu") -> CnnBiLstmClassifier:
    """Carga el modelo PyTorch desde un .pt y lo pone en eval mode."""
    model = CnnBiLstmClassifier()
    state = torch.load(pt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
