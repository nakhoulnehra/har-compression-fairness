"""Candidate HAR models (plan Section 2.3), written from the papers in papers/ (git-ignored); no
external code is used or copied.

Citation keys (PDFs in papers/), cited below as [KEY section, page]:
    [HARTH] Logacjov et al. 2021, "HARTH: A Human Activity Recognition Dataset for Machine Learning",
            Sensors 21:7853                                   papers/sensors-21-07853 (1).pdf
    [DCL]   Ordonez & Roggen 2016, "Deep Convolutional and LSTM Recurrent Neural Networks for
            Multimodal Wearable Activity Recognition", Sensors 16:115   papers/sensors-16-00115.pdf
    [TINY]  Zhou et al. 2022, "TinyHAR: A Lightweight Deep Learning Model Designed for HAR", ISWC '22
            papers/2022_ISWC____TinyHAR__A_Lightweight_Deep_Learning_Model_Designed_for_HAR (2).pdf
    [HART]  Ek, Portet & Lalanda, "Transformer-based Models to Deal with Heterogeneous Environments
            in Human Activity Recognition", arXiv 2209.11750v2   papers/2209.11750v2.pdf
"our choice" = the paper does not state the value. "plan" = a decision in plan/HAR_IMPLEMENTATION.md
that deliberately differs from the paper (same windows, classes, normalisation and training budget for
every candidate). The full table is in plan/HAR_IMPLEMENTATION.md, Appendix A.

Common to every model (plan, not from these papers): input (B, 6, 128) raw acceleration in g at 50 Hz
(2.56 s, as in [HART Sec 2.2, p4]); channels 0-2 back, 3-5 thigh; per-channel z-normalisation with
statistics from the fold's HARTH training subjects, stored as buffers (channel-wise z-normalisation as
in [HART Sec 2.2, p4]); 7 output logits; every backbone exposes its classification layer as `.head`
(excluded from pruning, plan Q3). Width 0.5 halves the channel/unit counts uniformly, like the width
multiplier [TINY Sec 4.1, p3] uses for DeepConvLSTM.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CHANNELS, WIN, N_CLASSES = 6, 128, 7
DEEP_MODELS = ("cnn1d", "tinyhar", "deepconvlstm", "hart")
ALL_MODELS = DEEP_MODELS + ("xgboost",)

SOURCES = {
    "cnn1d": ("Logacjov et al. (2021) [HARTH Sec 4.1, p11-12]",
              "checked: 4 conv layers x 128 kernels, sizes [6,12,12,32], dropout 0.4, 2 x FC-512 ReLU from the paper; "
              "conv type (axes as input channels), 'same' padding and global average pooling are our choices"),
    "deepconvlstm": ("Ordonez & Roggen (2016) [DCL Sec 3, 3.1, 3.3, Table 1]",
                     "checked: C(64)x4 kernel 5 no padding, R(128)x2, dropout 0.5 on dense inputs, orthogonal init, last step"),
    "tinyhar": ("Zhou et al. (2022) [TINY Sec 3.2]",
                "checked: 4 individual conv (5x1, stride 2, BN, ReLU), 1 transformer block across channels, FC to 2F, "
                "1 LSTM, temporal attention + gamma; F = 28, heads, FF size, dropout and attention scoring are our choices"),
    "hart": ("Ek, Portet & Lalanda (arXiv 2209.11750v2) [HART Sec 3.2, 4.1, 4.2, Table 7]",
             "checked: d=192, 6 blocks, 3 heads, sensor-wise MSA d/4 + LightConv d/2, Swish, dropout 0.3, GAP; "
             "frame length 16, LightConv internals/kernels, FF 2d, head 1024, DropPath 0.1 are our choices"),
    "xgboost": ("Logacjov et al. (2021) [HARTH Sec 3.3, 4.1]; Ustad et al. (2023) [Sec 2.3.3]",
                "reference only; HARTH's feature definitions (155 of its 161 features identifiable) and XGB settings"),
}

# Default sizes, each cited in the class below. Width 0.5 halves the keys in WIDTH_KEYS.
DEFAULTS = {
    "cnn1d": {"filters": 128, "kernels": (6, 12, 12, 32), "fc_units": 512, "n_fc": 2, "dropout": 0.4},
    "deepconvlstm": {"filters": 64, "kernel": 5, "n_conv": 4, "lstm_hidden": 128, "lstm_layers": 2, "dropout": 0.5},
    "tinyhar": {"filters": 28, "n_conv": 4, "kernel": 5, "stride": 2, "heads": 1, "ff_mult": 2, "dropout": 0.2},
    "hart": {"d": 192, "depth": 6, "heads": 3, "patch": 16, "stride": 16, "ff_mult": 2, "head_units": 1024,
             "dropout": 0.3, "drop_path": 0.1, "conv_kernels": (3, 5, 7, 7, 7, 7)},
}
WIDTH_KEYS = {"cnn1d": ("filters", "fc_units"), "deepconvlstm": ("filters", "lstm_hidden"), "tinyhar": ("filters",),
              "hart": ("d", "head_units")}
LIST_KEYS = ("kernels", "conv_kernels")


def resolve_config(name: str, width: float = 1.0, **overrides) -> dict:
    """Full architecture config for `name` at a width multiplier (1.0 = paper default, 0.5 = half)."""
    cfg = dict(DEFAULTS[name])
    for k in WIDTH_KEYS[name]:
        cfg[k] = int(round(cfg[k] * width))
    if name == "hart":  # d must split into 4 parts that are divisible by the heads
        q = 4 * cfg["heads"]
        cfg["d"] = max(q, int(round(cfg["d"] / q)) * q)
    if name == "tinyhar":
        cfg["filters"] = max(cfg["heads"], cfg["filters"] - cfg["filters"] % cfg["heads"])
    cfg.update(overrides)
    for k in LIST_KEYS:
        if k in cfg:
            cfg[k] = list(cfg[k])
    cfg["width"] = width
    return cfg


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class ChannelNorm(nn.Module):
    """(x - mean) / std per channel, with the statistics as (non-trainable) buffers."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).reshape(1, -1, 1).clone())
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).reshape(1, -1, 1).clone())

    def forward(self, x):
        return (x - self.mean) / self.std


class HarModel(nn.Module):
    def __init__(self, name: str, cfg: dict, backbone: nn.Module, mean, std):
        super().__init__()
        self.name, self.cfg = name, cfg
        self.norm = ChannelNorm(mean, std)
        self.backbone = backbone

    @property
    def head(self) -> nn.Linear:
        return self.backbone.head

    def forward(self, x):
        return self.backbone(self.norm(x))


# ---------------------------------------------------------------------------------------- CNN1D

class CNN1D(nn.Module):
    """The HARTH CNN [HARTH Sec 4.1, p11-12; Sec 3.2.6, p9]:
    - 4 conv layers, 128 kernels in each, kernel sizes [6, 12, 12, 32]          [Sec 4.1]
    - 1D kernels "to enable a single kernel to learn to extract useful information of each axis" [Sec 4.1]
      -> our reading: Conv1d over time with the 6 axes as input channels (a per-axis shared kernel
         is the other possible reading; the text does not settle it)
    - ReLU after convolutions                                                   [Sec 3.2.6]
    - dropout 0.4 after each layer                                              [Sec 4.1]
    - last two hidden layers fully connected, 512 units, ReLU; then the output layer   [Sec 4.1, p12]
    - our choice: 'same' padding (not stated; with the paper's 50-sample windows four valid
      convolutions with kernels 6+12+12+32 would not fit, so some padding is implied), no
      normalisation layers (none stated), and global average pooling over time before the dense
      layers (not stated; flattening 128 x 128 into 512 units would add 8.4M parameters).
    Not used (plan: equal budget for all candidates): SGD, lr 0.001, 80 epochs, no early stopping [Sec 4.1].
    """

    def __init__(self, filters=128, kernels=(6, 12, 12, 32), fc_units=512, n_fc=2, dropout=0.4, **_):
        super().__init__()
        layers, ch = [], N_CHANNELS
        for k in kernels:  # 'same' padding, written out so even kernels pad (k-1)//2 left and k//2 right
            layers += [nn.ConstantPad1d(((k - 1) // 2, k // 2), 0.0), nn.Conv1d(ch, filters, k), nn.ReLU(),
                       nn.Dropout(dropout)]
            ch = filters
        self.features = nn.Sequential(*layers)
        fc, d = [], filters
        for _ in range(n_fc):
            fc += [nn.Linear(d, fc_units), nn.ReLU(), nn.Dropout(dropout)]
            d = fc_units
        self.fc = nn.Sequential(*fc)
        self.head = nn.Linear(d, N_CLASSES)

    def forward(self, x):
        return self.head(self.fc(self.features(x).mean(dim=2)))


# --------------------------------------------------------------------------------- DeepConvLSTM

class DeepConvLSTM(nn.Module):
    """[DCL Sec 3, 3.1, 3.3, Table 1; Sec 4.2]:
    - C(64)-C(64)-C(64)-C(64)-R(128)-R(128)-Sm                                  [Sec 3.1, p8]
    - convolutions along time only, every sensor channel kept separate          [Sec 3.1]
      (here a (1, 5) kernel on a (1, C, T) map: 384 and 20,544 parameters per conv layer, as Table 1)
    - kernel length 5 in every conv layer, no padding ("only computed where the input and the
      kernel fully overlap"), no pooling                                        [Sec 3, Eq. 10, p7-8]
    - ReLU in conv layers; LSTM inputs at time t = all feature maps at time t   [Sec 3.1]
    - dropout p = 0.5 on the inputs of every dense layer (both LSTMs, the output layer) [Sec 3.3, p9]
    - orthogonal weight initialisation                                          [Sec 3.3, p9]
    - prediction at the last time step                                          [Sec 4.2, p12]
    Not used (plan): RMSProp lr 1e-3, decay 0.9, batch 100 [Sec 3.3]; last-sample labelling [Sec 4.2].
    """

    def __init__(self, filters=64, kernel=5, n_conv=4, lstm_hidden=128, lstm_layers=2, dropout=0.5, **_):
        super().__init__()
        convs, ch = [], 1
        for _ in range(n_conv):
            convs += [nn.Conv2d(ch, filters, (1, kernel)), nn.ReLU()]
            ch = filters
        self.convs = nn.Sequential(*convs)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(filters * N_CHANNELS, lstm_hidden, num_layers=lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.head = nn.Linear(lstm_hidden, N_CLASSES)
        for p in self.parameters():           # "Weights are randomly orthogonally initialized" [DCL Sec 3.3]
            if p.dim() >= 2:
                nn.init.orthogonal_(p)

    def forward(self, x):
        z = self.convs(x.unsqueeze(1))                       # (B, F, C, T')
        b, f, c, t = z.shape
        z = z.permute(0, 3, 1, 2).reshape(b, t, f * c)       # (B, T', F*C)
        out, _ = self.lstm(self.drop(z))
        return self.head(self.drop(out[:, -1]))


# -------------------------------------------------------------------------------------- TinyHAR

class TinyHAR(nn.Module):
    """[TINY Sec 3.2, p2-3; Fig. 1]:
    - individual convolutional subnet: 4 layers, kernel 5 x 1 along time only, each channel
      processed separately, stride 2 in each layer, same number of filters F, BN + ReLU   [3.2.1]
      (128 samples -> T* = 8)
    - one transformer encoder block across the sensor-channel dimension at each time step:
      scaled dot-product self-attention + two-layer FC feed-forward               [3.2.2]
    - FC cross-channel fusion of the flattened (C x F) features to F* = 2F        [3.2.3]
    - one LSTM layer (hidden size F*)                                              [3.2.4]
    - temporal attention: weighted average c of the hidden states, added to the last hidden
      state x_T* with a trainable multiplier gamma                                 [3.2.5]
    - prediction layer                                                             [Fig. 1]
    - F: the paper adjusts F per dataset; the only stated value is 28 (Opportunity) [Sec 4.2, p4]
    - our choice: 1 attention head (the text describes one scaled dot-product attention), FF hidden
      2F, dropout 0.2, ReLU after the fusion FC, attention scores from a linear layer with softmax
      over time, gamma initialised to 0 (none of these are stated).
    Not used (plan): Adam lr 1e-4, decay to 10% on 5-epoch plateau, <= 150 epochs [Sec 4.1, p4]
    (batch size 256 [Sec 4.1] is the plan's batch size too).
    """

    def __init__(self, filters=28, n_conv=4, kernel=5, stride=2, heads=1, ff_mult=2, dropout=0.2, **_):
        super().__init__()
        convs, ch = [], 1
        for _ in range(n_conv):
            convs += [nn.Conv2d(ch, filters, (kernel, 1), stride=(stride, 1), padding=(kernel // 2, 0)),
                      nn.BatchNorm2d(filters), nn.ReLU()]
            ch = filters
        self.convs = nn.Sequential(*convs)
        self.channel_attn = nn.TransformerEncoderLayer(filters, heads, dim_feedforward=ff_mult * filters,
                                                       dropout=dropout, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(filters * N_CHANNELS, 2 * filters), nn.ReLU())
        self.lstm = nn.LSTM(2 * filters, 2 * filters, batch_first=True)
        self.attn_score = nn.Linear(2 * filters, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * filters, N_CLASSES)

    def forward(self, x):
        z = self.convs(x.transpose(1, 2).unsqueeze(1))        # (B, F, T*, C)
        b, f, t, c = z.shape
        z = z.permute(0, 2, 3, 1).reshape(b * t, c, f)        # tokens = sensor channels
        z = self.channel_attn(z).reshape(b, t, c * f)
        h, _ = self.lstm(self.fusion(z))                      # (B, T*, 2F)
        a = torch.softmax(self.attn_score(h).squeeze(-1), dim=1)
        context = (a.unsqueeze(-1) * h).sum(dim=1)
        return self.head(self.drop(h[:, -1] + self.gamma * context))


# ----------------------------------------------------------------------------------------- HART

class DropPath(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = (torch.rand(x.shape[0], *([1] * (x.dim() - 1)), device=x.device) >= self.p).to(x.dtype)
        return x * keep / (1 - self.p)


class LightConv(nn.Module):
    """LightConv block, which [HART Sec 3.2, p8] takes from Wu et al. (2019) without describing it.
    Implemented after Wu et al.'s definition (our choice, that paper is not in papers/): GLU input
    projection, depthwise convolution over tokens with softmax-normalised kernels shared by the
    channels of each head, output projection."""

    def __init__(self, dim: int, kernel: int, heads: int, dropout: float):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.kernel, self.heads = dim, kernel, heads
        self.inp = nn.Linear(dim, 2 * dim)
        self.weight = nn.Parameter(torch.randn(heads, kernel) * 0.1)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                     # (B, N, dim)
        z = F.glu(self.inp(x), dim=-1).transpose(1, 2)        # (B, dim, N)
        w = torch.softmax(self.weight, dim=-1)                # (H, k)
        w = w.repeat_interleave(self.dim // self.heads, dim=0).unsqueeze(1)  # (dim, 1, k)
        z = F.conv1d(z, w, padding=self.kernel // 2, groups=self.dim)
        return self.drop(self.out(z.transpose(1, 2)))


class HARTBlock(nn.Module):
    """[HART Sec 3.2, p8]: LN -> (accelerometer-wise MSA on half of the accelerometer embedding,
    d/4 | LightConv on the remaining halves of both sensors, d/2 | gyroscope-wise MSA on half of the
    gyroscope embedding, d/4) -> concatenated "with their original position kept in place" + residual
    -> LN -> feed-forward + residual. Here accelerometer/gyroscope = back/thigh (plan)."""

    def __init__(self, d, heads, kernel, ff_mult, dropout, drop_path):
        super().__init__()
        self.q = d // 4
        self.ln1 = nn.LayerNorm(d)
        self.attn_back = nn.MultiheadAttention(self.q, heads, dropout=dropout, batch_first=True)
        self.attn_thigh = nn.MultiheadAttention(self.q, heads, dropout=dropout, batch_first=True)
        self.lite = LightConv(d // 2, kernel, heads, dropout)
        self.ln2 = nn.LayerNorm(d)
        # Swish activation [HART Sec 4.1]; hidden size ff_mult * d is our choice (not stated)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.SiLU(), nn.Dropout(dropout),
                                nn.Linear(ff_mult * d, d), nn.Dropout(dropout))
        self.drop_path = DropPath(drop_path)   # DropPath before every residual connection [HART Sec 4.1]

    def forward(self, x):
        z = self.ln1(x)
        q = self.q
        back, mid, thigh = z[..., :q], z[..., q:3 * q], z[..., 3 * q:]   # back half | mixed | thigh half
        back = self.attn_back(back, back, back, need_weights=False)[0]
        thigh = self.attn_thigh(thigh, thigh, thigh, need_weights=False)[0]
        x = x + self.drop_path(torch.cat([back, self.lite(mid), thigh], dim=-1))
        return x + self.drop_path(self.ff(self.ln2(x)))


class HART(nn.Module):
    """[HART Sec 3.2, p7-8; Sec 4.1-4.2, p10; Table 7, p13]:
    - inputs: non-overlapping windows x in R^{W x 3S}, W = 128, S = 2 sensors      [Sec 3.2, p7]
    - one convolutional embedding per sensor, d/S filters each, non-overlapping frames [Sec 3.2, p7]
    - d = 192                                                                   [Sec 3.2, p7; Sec 4.2, p10]
    - learnable position embedding, no class token, global average pooling     [Sec 3.2, p7]
    - 6 encoder blocks, three attention heads per block                        [Sec 4.2, p10]
    - Swish; dropout 0.3 after most feed-forward layers; DropPath before every residual [Sec 4.1, p10]
    - GAP -> feed-forward layer -> classification head                         [Sec 3.2, p8]
    - 1,445,918 parameters in the paper                                         [Table 7, p13]
    - our choice: frame (patch) length and stride 16 (the paper gives no frame length; 8 tokens),
      3 heads in EACH sensor-wise MSA (the paper says three per block), LightConv kernel sizes
      (3, 5, 7, 7, 7, 7), feed-forward hidden 2d, head feed-forward 1,024 units with Swish,
      DropPath rate 0.1 rising linearly with depth. These give 1,389,555 parameters (-3.9%).
    Not used (plan): Adam lr 5e-4, batch 32, 200 epochs, label smoothing 0.1 [Sec 4.1] (class-weighted
    training [Sec 4.1] is used, as for every candidate).
    """

    def __init__(self, d=192, depth=6, heads=3, patch=16, stride=16, ff_mult=2, head_units=1024,
                 dropout=0.3, drop_path=0.1, conv_kernels=(3, 5, 7, 7, 7, 7), **_):
        super().__init__()
        assert d % 4 == 0 and (d // 4) % heads == 0 and len(conv_kernels) == depth
        self.embed_back = nn.Conv1d(3, d // 2, patch, stride=stride)    # back sensor: channels 0-2
        self.embed_thigh = nn.Conv1d(3, d // 2, patch, stride=stride)   # thigh sensor: channels 3-5
        n_tokens = (WIN - patch) // stride + 1
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([HARTBlock(d, heads, k, ff_mult, dropout, drop_path * (i + 1) / depth)
                                     for i, k in enumerate(conv_kernels)])
        self.ln = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, head_units), nn.SiLU(), nn.Dropout(dropout))
        self.head = nn.Linear(head_units, N_CLASSES)

    def embed(self, x):
        return torch.cat([self.embed_back(x[:, :3]), self.embed_thigh(x[:, 3:])], dim=1).transpose(1, 2)

    def forward(self, x):
        z = self.embed(x) + self.pos
        for blk in self.blocks:
            z = blk(z)
        return self.head(self.mlp(self.ln(z).mean(dim=1)))


BACKBONES = {"cnn1d": CNN1D, "deepconvlstm": DeepConvLSTM, "tinyhar": TinyHAR, "hart": HART}


def build_model(name: str, cfg: dict, norm_mean, norm_std) -> HarModel:
    if name not in BACKBONES:
        raise ValueError(f"unknown model {name!r}; choose from {DEEP_MODELS}")
    return HarModel(name, cfg, BACKBONES[name](**cfg), norm_mean, norm_std)
