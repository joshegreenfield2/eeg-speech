"""models.py — deep learning architectures for EEG classification.

Two PyTorch models:

  EEGNet — compact CNN designed for EEG (Lawhence et al. 2018, ~2k params).
           Currently the most-cited compact BCI architecture. Subject-dependent
           training only on small data.

  CNN_BiLSTM — hybrid spatial CNN + temporal BiLSTM. The 77.8% paper
           on KaraOne 11-class used this structure (PMC 2024).

Both consume raw `(n_epochs, n_channels, n_times)` arrays — NO hand-crafted
features. Use `train.py` to fit/evaluate per subject.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- EEGNet ----------
class EEGNet(nn.Module):
    """EEGNet v2 (Lawhence et al. 2018).

    Args:
        n_classes: number of output classes.
        n_channels: number of EEG channels (e.g. 40 good channels for KaraOne).
        n_times: number of time samples per trial (e.g. 4900 for 4.9s @ 1kHz).
        dropout: dropout rate (default 0.5).
        kernel_length: temporal conv kernel length (default sfreq/2 = 500 for 1kHz).
        F1: number of temporal filters in block 1 (default 8).
        D: depth multiplier in depthwise conv (default 2).
        F2: number of point-wise filters in block 2 (default F1*D = 16).
    """
    def __init__(
        self,
        n_classes: int,
        n_channels: int,
        n_times: int,
        dropout: float = 0.5,
        kernel_length: int = 500,  # ~half a second at 1kHz
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
    ):
        super().__init__()
        self.n_classes = n_classes

        # Block 1 — temporal conv + depthwise spatial conv
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2 — separable conv (depthwise + pointwise)
        self.sep_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.sep_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Compute classifier input size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out = self._features(dummy)
            classifier_in = out.numel()
        self.classifier = nn.Linear(classifier_in, n_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, C, T)
        x = self.bn1(self.conv1(x))
        x = F.elu(self.bn2(self.depthwise(x)))
        x = self.drop1(self.pool1(x))
        x = self.sep_point(self.sep_depth(x))
        x = F.elu(self.bn3(x))
        x = self.drop2(self.pool2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, C, T) or (B, 1, C, T)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self._features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


# ---------- CNN-BiLSTM ----------
class CNN_BiLSTM(nn.Module):
    """Spatial CNN feature extractor + temporal Bidirectional LSTM.

    Following the structure from the PMC 2024 KaraOne paper which reported
    77.8% on 11-class. Approximate (paper details vary):
      - 2D CNN over (channels × time) → temporal feature sequence
      - BiLSTM over the temporal sequence
      - Linear classifier on the final hidden states
    """
    def __init__(
        self,
        n_classes: int,
        n_channels: int,
        n_times: int,
        cnn_channels: tuple[int, int, int] = (32, 64, 128),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        c1, c2, c3 = cnn_channels
        # CNN over (B, 1, C, T) — collapse the channel dim with grouped conv,
        # then progressively reduce time
        self.cnn = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=(n_channels, 16), padding=(0, 8)),  # collapse channels
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d((1, 4)),
            nn.Dropout(dropout),

            nn.Conv2d(c1, c2, kernel_size=(1, 16), padding=(0, 8)),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d((1, 4)),
            nn.Dropout(dropout),

            nn.Conv2d(c2, c3, kernel_size=(1, 8), padding=(0, 4)),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.MaxPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # Compute LSTM input size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out = self.cnn(dummy)
            # out: (1, c3, 1, T') → squeeze to (1, c3, T') then transpose to (1, T', c3)
            self._lstm_input_size = out.shape[1]
            self._seq_len = out.shape[3]

        self.lstm = nn.LSTM(
            input_size=self._lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.cnn(x)
        # (B, c3, 1, T') → (B, T', c3)
        x = x.squeeze(2).transpose(1, 2)
        # LSTM
        out, (h_n, _) = self.lstm(x)
        # Concatenate final forward + backward hidden states
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h = torch.cat([h_fwd, h_bwd], dim=1)
        h = self.dropout(h)
        return self.classifier(h)


def build_model(name: str, n_classes: int, n_channels: int, n_times: int) -> nn.Module:
    """Factory function — returns an instantiated model by name."""
    if name.lower() == "eegnet":
        return EEGNet(n_classes=n_classes, n_channels=n_channels, n_times=n_times)
    if name.lower() in ("cnn_bilstm", "cnn-bilstm"):
        return CNN_BiLSTM(n_classes=n_classes, n_channels=n_channels, n_times=n_times)
    raise ValueError(f"Unknown model: {name}. Available: 'eegnet', 'cnn_bilstm'")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
