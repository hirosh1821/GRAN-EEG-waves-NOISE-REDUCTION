# -*- coding: utf-8 -*-
"""
================================================================================
  GRAN v2.0  --  FULL TRAINING SCRIPT  (run this in VS Code)
================================================================================

  USAGE:
      python TRAIN_NOW.py                   (full run, both datasets)
      python TRAIN_NOW.py --bci-only        (skip Sleep-EDF, faster)
      python TRAIN_NOW.py --resume          (continue from last checkpoint)
      python TRAIN_NOW.py --epochs 150      (override epoch count)

  WHAT THIS DOES:
      1. Loads BCI Competition IV (7 train subjects + 2 val subjects)
      2. Loads Sleep-EDF cassette PSG files  (up to 25 files, 1500 wins/file)
      3. Generates 7-type synthetic artifacts on every window
      4. Trains GRAN v2.0 (2.4M params, deep U-Net + attention + dilated conv)
      5. Saves best checkpoint every time val_loss improves
      6. Saves full checkpoint every 5 epochs  (safe resume on crash)
      7. On GPU OOM -> automatically halves batch size and retries
      8. Prints a live table every epoch (loss / SNR / corr / LR)
      9. Saves final metrics JSON + training curve PNG + signal plots

  OUTPUT:
      checkpoints/gran_best.pth        best model weights
      checkpoints/gran_resume.pth      periodic checkpoint for safe resume
      outputs_v2/metrics_v2.json       evaluation results
      outputs_v2/training_curve_v2.png loss + LR curves
      outputs_v2/signal_comparison.png noisy vs restored vs clean
      TRAIN_LOG.txt                    full console log saved to disk
================================================================================
"""

# ─── Force UTF-8 output on Windows so special chars don't crash ───────────────
import sys
import io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import gc
import math
import time
import json
import argparse
import warnings
import traceback
import logging
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy import signal as scipy_signal
from scipy import stats

# ─── Script root ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

BCI_DIR   = os.path.join(ROOT, "BCI Competition IV")
SLEEP_DIR = os.path.join(ROOT, "sleep-edf-database-expanded-1.0.0")
OUT_DIR   = os.path.join(ROOT, "outputs_v2")
CKPT_DIR  = os.path.join(ROOT, "checkpoints")
LOG_FILE  = os.path.join(ROOT, "TRAIN_LOG.txt")

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ─── Dual logger (console + file) ─────────────────────────────────────────────
class _TeeLogger:
    """Writes every print() AND stderr call to both terminal and a log file."""
    def __init__(self, path, original_stream):
        self._file   = open(path, 'w', encoding='utf-8', errors='replace', buffering=1)
        self._stream = original_stream
    def write(self, msg):
        try:
            self._stream.write(msg)
            self._stream.flush()
        except Exception:
            pass
        try:
            # Strip ANSI / tqdm carriage-return junk before writing to file
            clean = msg.replace('\r', '\n')
            self._file.write(clean)
            self._file.flush()
        except Exception:
            pass
    def flush(self):
        try: self._stream.flush()
        except Exception: pass
        try: self._file.flush()
        except Exception: pass
    def close(self):
        try: self._file.close()
        except Exception: pass
    # Make it behave like a real file object
    def isatty(self): return False
    def fileno(self): raise io.UnsupportedOperation("no fileno")

_log_file_handle = open(LOG_FILE, 'w', encoding='utf-8', errors='replace', buffering=1)

class _TeeOut(_TeeLogger):
    pass
class _TeeErr(_TeeLogger):
    pass

_tee_out = _TeeOut(LOG_FILE, sys.stdout)
# Reuse same file for stderr — share the handle
_tee_err = _TeeErr.__new__(_TeeErr)
_TeeLogger.__init__(_tee_err, LOG_FILE, sys.stderr)
# Point both to the same open file so writes interleave correctly
_tee_err._file = _tee_out._file

sys.stdout = _tee_out
sys.stderr = _tee_err


# ==============================================================================
# SECTION 1 -- GRAN MODEL (inline, no import needed)
# ==============================================================================

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dropout=0.1):
        super().__init__()
        p = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=p, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=p, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch))
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        r = max(channels // reduction, 4)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.mx  = nn.AdaptiveMaxPool1d(1)
        self.fc  = nn.Sequential(
            nn.Linear(channels, r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(r, channels, bias=False))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.shape
        w = self.sig(self.fc(self.avg(x).squeeze(-1)) +
                     self.fc(self.mx(x).squeeze(-1)))
        return x * w.unsqueeze(-1)


class DilatedMultiScaleBlock(nn.Module):
    """Parallel dilated convolutions (dilation 1/2/4/8) for large receptive field."""
    def __init__(self, channels):
        super().__init__()
        c = channels // 4
        self.d1   = nn.Conv1d(channels, c, 3, padding=1,  dilation=1, bias=False)
        self.d2   = nn.Conv1d(channels, c, 3, padding=2,  dilation=2, bias=False)
        self.d4   = nn.Conv1d(channels, c, 3, padding=4,  dilation=4, bias=False)
        self.d8   = nn.Conv1d(channels, c, 3, padding=8,  dilation=8, bias=False)
        self.fuse = nn.Conv1d(channels, channels, 1, bias=False)
        self.bn   = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.se   = SqueezeExcitation(channels)

    def forward(self, x):
        out = torch.cat([self.d1(x), self.d2(x), self.d4(x), self.d8(x)], dim=1)
        out = self.relu(self.bn(self.fuse(out)))
        return self.se(out) + x


class LightTemporalAttention(nn.Module):
    """Multi-head self-attention along the time axis (applied at bottleneck)."""
    def __init__(self, channels, num_heads=8, dropout=0.1):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.scale     = self.head_dim ** -0.5
        self.norm = nn.LayerNorm(channels)
        self.qkv  = nn.Linear(channels, 3 * channels, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, c, t = x.shape
        xt     = x.permute(0, 2, 1)            # [B, T, C]
        normed = self.norm(xt)
        qkv    = self.qkv(normed).reshape(b, t, 3, self.num_heads, self.head_dim)
        qkv    = qkv.permute(2, 0, 3, 1, 4)   # [3, B, H, T, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn   = self.drop(F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1))
        out    = (attn @ v).transpose(1, 2).reshape(b, t, c)
        out    = self.proj(out)
        return (xt + out).permute(0, 2, 1)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_res=2, stride=2, dropout=0.1):
        super().__init__()
        layers = [ResBlock1D(in_ch, out_ch, stride=stride, dropout=dropout)]
        for _ in range(n_res - 1):
            layers.append(ResBlock1D(out_ch, out_ch, dropout=dropout))
        self.blocks = nn.Sequential(*layers)
        self.se = SqueezeExcitation(out_ch)

    def forward(self, x):
        return self.se(self.blocks(x))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, n_res=2, dropout=0.1):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True))
        layers = [ResBlock1D(out_ch + skip_ch, out_ch, dropout=dropout)]
        for _ in range(n_res - 1):
            layers.append(ResBlock1D(out_ch, out_ch, dropout=dropout))
        self.blocks = nn.Sequential(*layers)
        self.se = SqueezeExcitation(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.se(self.blocks(x))


class GRANModel(nn.Module):
    """
    GRAN v2.0 - Generative-Refinement-Annotation EEG Restoration Network

    Deep U-Net with:
      - 4-level encoder/decoder with Residual blocks
      - Squeeze-Excitation channel attention at every level
      - Dilated multi-scale bottleneck  (receptive fields: 1/2/4/8 samples)
      - Lightweight temporal self-attention at bottleneck
      - Residual learning: output = noisy + correction
      ~ 2.38 M trainable parameters
    """

    def __init__(self, n_channels=3, n_samples=1000, base_ch=32, dropout=0.1):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples  = n_samples
        ch = base_ch

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_channels, ch, 7, padding=3, bias=False),
            nn.BatchNorm1d(ch), nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(ch), nn.ReLU(inplace=True))

        # Encoder: s0=[B,32,1000]  s1=[B,32,500]  s2=[B,64,250]  s3=[B,128,125]  b=[B,256,62]
        self.enc1 = EncoderBlock(ch,   ch,   n_res=2, stride=2, dropout=dropout)
        self.enc2 = EncoderBlock(ch,   ch*2, n_res=2, stride=2, dropout=dropout)
        self.enc3 = EncoderBlock(ch*2, ch*4, n_res=2, stride=2, dropout=dropout)
        self.enc4 = EncoderBlock(ch*4, ch*8, n_res=2, stride=2, dropout=dropout)

        # Bottleneck
        self.bottleneck    = nn.Sequential(DilatedMultiScaleBlock(ch*8),
                                           DilatedMultiScaleBlock(ch*8))
        self.temporal_attn = LightTemporalAttention(ch*8, num_heads=8, dropout=dropout)

        # Decoder
        self.dec4 = DecoderBlock(ch*8, ch*4, ch*4, n_res=2, dropout=dropout)
        self.dec3 = DecoderBlock(ch*4, ch*2, ch*2, n_res=2, dropout=dropout)
        self.dec2 = DecoderBlock(ch*2, ch,   ch,   n_res=2, dropout=dropout)
        self.dec1 = DecoderBlock(ch,   ch,   ch,   n_res=2, dropout=dropout)

        # Output (predicts CORRECTION added to noisy input)
        self.output_head = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(ch), nn.ReLU(inplace=True),
            nn.Conv1d(ch, n_channels, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        s0 = self.input_proj(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b  = self.enc4(s3)
        b  = self.bottleneck(b)
        b  = self.temporal_attn(b)
        d4 = self.dec4(b,  s3)
        d3 = self.dec3(d4, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2, s0)
        return x + self.output_head(d1)   # residual: noisy + correction

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# SECTION 2 -- LOSS FUNCTION
# ==============================================================================

class GRANLoss(nn.Module):
    """MSE + L1 + log-frequency magnitude MSE."""
    def __init__(self, mse_w=1.0, l1_w=0.1, freq_w=0.05):
        super().__init__()
        self.mse_w  = mse_w
        self.l1_w   = l1_w
        self.freq_w = freq_w

    def forward(self, output, target):
        mse  = F.mse_loss(output, target)
        l1   = F.l1_loss(output, target)
        # Frequency domain: log-magnitude MSE
        om   = torch.abs(torch.fft.rfft(output, dim=-1)) + 1e-8
        tm   = torch.abs(torch.fft.rfft(target, dim=-1)) + 1e-8
        freq = F.mse_loss(torch.log(om), torch.log(tm))
        total = self.mse_w * mse + self.l1_w * l1 + self.freq_w * freq
        return total, {'mse': mse.item(), 'l1': l1.item(), 'freq': freq.item()}


# ==============================================================================
# SECTION 3 -- AUGMENTED DATASET
# ==============================================================================

class AugEEGDataset(Dataset):
    """On-the-fly augmentation: amplitude scale, sign flip, circular shift, jitter."""
    def __init__(self, X_noisy, X_clean, augment=True):
        # [N, T, C] -> [N, C, T]
        self.noisy   = torch.from_numpy(np.ascontiguousarray(X_noisy.transpose(0,2,1)).astype(np.float32))
        self.clean   = torch.from_numpy(np.ascontiguousarray(X_clean.transpose(0,2,1)).astype(np.float32))
        self.augment = augment

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        n = self.noisy[idx].clone()
        c = self.clean[idx].clone()
        if self.augment:
            # 1. Amplitude scale
            s = 0.8 + 0.4 * torch.rand(1).item()
            n, c = n * s, c * s
            # 2. Polarity flip
            if torch.rand(1).item() < 0.30:
                n, c = -n, -c
            # 3. Circular time shift (up to 5%)
            T = n.shape[-1]
            sh = torch.randint(-T//20, T//20+1, (1,)).item()
            if sh:
                n = torch.roll(n, sh, dims=-1)
                c = torch.roll(c, sh, dims=-1)
            # 4. Tiny jitter on noisy only
            n = n + 0.02 * torch.randn_like(n)
        return n, c


# ==============================================================================
# SECTION 4 -- ARTIFACT GENERATOR  (7 types)
# ==============================================================================

class ArtifactGenerator:
    """7-type physiological artifact synthesiser."""

    def __init__(self, fs=250, powerline=50.0, seed=42):
        self.fs  = fs
        self.pl  = powerline
        self.rng = np.random.RandomState(seed)
        nyq = fs / 2.0
        self._mb, self._ma = scipy_signal.butter(2, [20.0/nyq, min(45.0/nyq, 0.99)], btype='band')

    # ── individual artifact types ──────────────────────────────────────────────

    def _blink(self, n, amp):
        dur = int(self.rng.randint(150, 601) * self.fs / 1000)
        start = self.rng.randint(0, max(1, n - dur))
        a = self.rng.uniform(4.0, 10.0) * amp
        t = np.linspace(-3, 3, dur)
        out = np.zeros(n); out[start:start+dur] = a * np.exp(-0.5*t**2)
        return out

    def _muscle(self, n, amp):
        dur = int(self.rng.randint(100, 501) * self.fs / 1000)
        start = self.rng.randint(0, max(1, n - dur))
        a = self.rng.uniform(1.5, 4.0) * amp
        noise = scipy_signal.filtfilt(self._mb, self._ma, self.rng.randn(dur) * a)
        burst = noise * np.hanning(dur)
        out = np.zeros(n); out[start:start+dur] = burst[:n-start]
        return out

    def _drift(self, n, amp):
        f = self.rng.uniform(0.05, 0.5); ph = self.rng.uniform(0, 2*np.pi)
        a = self.rng.uniform(1.5, 4.0) * amp
        t = np.arange(n) / self.fs
        return a * np.sin(2*np.pi*f*t + ph)

    def _powerline(self, n, amp):
        t = np.arange(n) / self.fs; ph = self.rng.uniform(0, 2*np.pi)
        a = self.rng.uniform(0.5, 2.0) * amp
        out = a * np.sin(2*np.pi*self.pl*t + ph)
        if self.rng.random() < 0.4:
            h = self.rng.uniform(0.1, 0.5) * a
            out += h * np.sin(4*np.pi*self.pl*t + self.rng.uniform(0, 2*np.pi))
        return out

    def _pop(self, n, amp):
        start = self.rng.randint(10, n-10)
        a = self.rng.uniform(3.0, 8.0) * amp * self.rng.choice([-1,1])
        tau = self.rng.uniform(0.05, 0.3) * self.fs
        out = np.zeros(n)
        idxs = np.arange(start, n)
        out[start:] = a * np.exp(-(idxs - start) / tau)
        return out

    def _motion(self, n, amp):
        fd = self.rng.uniform(0.5, 2.0); fm = self.rng.uniform(0.1, 0.4)
        a  = self.rng.uniform(2.0, 5.0) * amp
        t  = np.arange(n) / self.fs
        env = 0.5 * (1 + np.sin(2*np.pi*fm*t + self.rng.uniform(0, 2*np.pi)))
        return a * env * np.sin(2*np.pi*fd*t + self.rng.uniform(0, 2*np.pi))

    def _whitenoise(self, n, amp):
        return self.rng.randn(n) * self.rng.uniform(0.2, 1.0) * amp

    _PROBS = {'blink':0.55, 'muscle':0.45, 'drift':0.40,
              'powerline':0.35, 'pop':0.20, 'motion':0.35, 'noise':0.30}

    def _make(self, kind, n, amp):
        fn = {'blink':self._blink,'muscle':self._muscle,'drift':self._drift,
              'powerline':self._powerline,'pop':self._pop,'motion':self._motion,
              'noise':self._whitenoise}
        return fn[kind](n, amp)

    def corrupt_batch(self, X_clean, artifact_prob=0.92, verbose=False):
        """X_clean: [N, T, C]  ->  X_noisy [N, T, C]"""
        N, T, C = X_clean.shape
        X_noisy = np.empty_like(X_clean)
        for i in range(N):
            if self.rng.random() > artifact_prob:
                X_noisy[i] = X_clean[i]
                continue
            std = max(float(np.std(X_clean[i])), 1e-6)
            art = np.zeros((T, C))
            active = [k for k, p in self._PROBS.items() if self.rng.random() < p]
            if not active: active = ['blink']
            for kind in active:
                sig = self._make(kind, T, std)
                spread = self.rng.random() < 0.70
                chs = list(range(C)) if spread else self.rng.choice(C, self.rng.randint(1,C+1), replace=False).tolist()
                for ch in chs:
                    art[:, ch] += sig * self.rng.uniform(0.8, 1.2)
            X_noisy[i] = X_clean[i] + art
            if verbose and (i+1) % 500 == 0:
                print(f"  corrupted {i+1}/{N}")
        return X_noisy


# ==============================================================================
# SECTION 5 -- DATA LOADERS
# ==============================================================================

def load_bci(data_dir, train_subjects, val_subjects, verbose=True):
    """Load BCI Competition IV Dataset 2a mat files."""
    import scipy.io as sio
    from tqdm import tqdm

    FS = 250; WIN = 1000; STEP = 500
    CH_IDX = [7, 9, 11]   # C3, Cz, C4

    nyq = FS / 2.0
    b, a = scipy_signal.butter(4, [0.5/nyq, 45.0/nyq], btype='band')

    def load_one(subj_id):
        path = os.path.join(data_dir, f"{subj_id}.mat")
        if not os.path.exists(path):
            print(f"  [BCI] WARNING: {path} not found, skipping.")
            return None
        mat  = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
        runs = mat['data']
        raw  = np.concatenate([r.X for r in runs], axis=0)  # [T, 25]
        eeg  = raw[:, CH_IDX].astype(np.float32)             # [T, 3]
        for ch in range(3):
            eeg[:, ch] = scipy_signal.filtfilt(b, a, eeg[:, ch])
        eeg -= eeg.mean(axis=0)
        # Segment
        wins = []
        for s in range(0, len(eeg) - WIN + 1, STEP):
            w = eeg[s:s+WIN]
            mu, sd = w.mean(axis=0), w.std(axis=0)
            sd[sd < 1e-8] = 1.0
            wins.append((w - mu) / sd)
        return np.stack(wins).astype(np.float32) if wins else None

    def load_set(subjects, label):
        all_w = []
        it = tqdm(subjects, desc=f"[BCI] {label}") if verbose else subjects
        for s in it:
            w = load_one(s)
            if w is not None:
                all_w.append(w)
        return np.concatenate(all_w, axis=0) if all_w else np.empty((0,WIN,3), dtype=np.float32)

    X_train = load_set(train_subjects, "train")
    X_val   = load_set(val_subjects,   "val")
    return X_train, X_val


def _read_edf_eeg(filepath, ch_names=('EEG Fpz-Cz', 'EEG Pz-Oz')):
    """Pure-numpy EDF reader: returns (data_array [T, n_ch], fs)."""
    try:
        with open(filepath, 'rb') as f:
            def rd(n): return f.read(n).decode('ascii', errors='replace').strip()
            rd(8); rd(80); rd(80); rd(8); rd(8)
            n_hdr   = int(rd(8) or 0)
            rd(44)
            n_rec   = int(rd(8) or 0)
            dur     = float(rd(8) or 1)
            nc      = int(rd(4) or 0)
            labels  = [f.read(16).decode('ascii', errors='replace').strip() for _ in range(nc)]
            [f.read(80) for _ in range(nc)]   # transducer
            [f.read(8)  for _ in range(nc)]   # phys_dim
            phys_min = [float(f.read(8).decode('ascii', errors='replace').strip() or -1) for _ in range(nc)]
            phys_max = [float(f.read(8).decode('ascii', errors='replace').strip() or  1) for _ in range(nc)]
            dig_min  = [int(f.read(8).decode('ascii', errors='replace').strip() or -32768) for _ in range(nc)]
            dig_max  = [int(f.read(8).decode('ascii', errors='replace').strip() or  32767) for _ in range(nc)]
            [f.read(80) for _ in range(nc)]   # prefilter
            nsamp    = [int(f.read(8).decode('ascii', errors='replace').strip() or 0) for _ in range(nc)]
            [f.read(32) for _ in range(nc)]   # reserved
            gain   = [(phys_max[i]-phys_min[i]) / (dig_max[i]-dig_min[i]) if dig_max[i]!=dig_min[i] else 1.0 for i in range(nc)]
            offset = [phys_min[i] - gain[i]*dig_min[i] for i in range(nc)]
            fs_all = [nsamp[i]/dur for i in range(nc)]

            # Find wanted channels
            want_idx = []
            for name in ch_names:
                for ci, lbl in enumerate(labels):
                    if name.lower() in lbl.lower():
                        want_idx.append(ci); break
                else:
                    want_idx.append(None)

            buffers = {ci: [] for ci in want_idx if ci is not None}
            for _ in range(n_rec):
                for ci in range(nc):
                    raw = f.read(nsamp[ci] * 2)
                    if ci in buffers:
                        s = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                        buffers[ci].append(s * gain[ci] + offset[ci])

        arrays = []
        fs_out = None
        for ci in want_idx:
            if ci is None:
                return None, 0
            arr = np.concatenate(buffers[ci])
            arrays.append(arr)
            fs_out = fs_all[ci]
        return np.stack(arrays, axis=1), fs_out   # [T, n_ch]
    except Exception as e:
        return None, 0


def load_sleep_edf(data_dir, n_files=25, max_per_file=1500, seed=42, verbose=True):
    """Load Sleep-EDF cassette PSG files -> [N, 1000, 3]."""
    import glob
    rng = np.random.RandomState(seed)

    cassette = os.path.join(data_dir, 'sleep-cassette')
    telemetry = os.path.join(data_dir, 'sleep-telemetry')
    files = []
    for d in [cassette, telemetry]:
        if os.path.isdir(d):
            files.extend(sorted(glob.glob(os.path.join(d, '*PSG.edf'))))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, '**', '*PSG.edf'), recursive=True))

    files = files[:n_files]
    if verbose:
        print(f"[Sleep-EDF] {len(files)} PSG files  (max {max_per_file} wins/file)")

    WIN = 1000; STEP = 500; TARGET_FS = 250
    nyq = TARGET_FS / 2.0
    b, a = scipy_signal.butter(4, [0.5/nyq, 45.0/nyq], btype='band')

    all_wins = []
    for i, fp in enumerate(files):
        data, src_fs = _read_edf_eeg(fp)
        if data is None or len(data) < WIN:
            if verbose: print(f"  [{i+1:3d}] {os.path.basename(fp):35s} -> skip (read fail)")
            continue
        # Resample 100 Hz -> 250 Hz
        if int(round(src_fs)) != TARGET_FS:
            up = TARGET_FS; dn = int(round(src_fs))
            data = scipy_signal.resample_poly(data, up, dn).astype(np.float32)
        # Filter + mean removal
        for ch in range(data.shape[1]):
            data[:, ch] = scipy_signal.filtfilt(b, a, data[:, ch])
        data -= data.mean(axis=0)
        # Segment
        wins = []
        for s in range(0, len(data) - WIN + 1, STEP):
            w = data[s:s+WIN]
            mu, sd = w.mean(axis=0), w.std(axis=0)
            sd[sd < 1e-8] = 1.0
            wins.append((w - mu) / sd)
        if not wins:
            continue
        wins = np.stack(wins).astype(np.float32)  # [N, T, 2]
        # Pad to 3 channels (repeat ch 0)
        ch3 = np.concatenate([wins, wins[:, :, :1]], axis=2)  # [N, T, 3]
        # Cap
        if len(ch3) > max_per_file:
            idx = np.sort(rng.choice(len(ch3), max_per_file, replace=False))
            ch3 = ch3[idx]
        all_wins.append(ch3)
        if verbose:
            print(f"  [{i+1:3d}] {os.path.basename(fp):35s} -> {len(ch3):5d} windows")

    if not all_wins:
        return np.empty((0, 1000, 3), dtype=np.float32)
    X = np.concatenate(all_wins, axis=0)
    if verbose: print(f"[Sleep-EDF] total: {len(X)} windows")
    return X


# ==============================================================================
# SECTION 6 -- TRAINING UTILITIES
# ==============================================================================

class WarmupCosineScheduler:
    """Linear warmup then cosine decay."""
    def __init__(self, optimizer, warmup, total, base_lr, min_lr=1e-6):
        self.opt = optimizer; self.warmup = warmup; self.total = total
        self.base = base_lr; self.min = min_lr

    def step(self, epoch):
        if epoch < self.warmup:
            lr = self.base * (epoch + 1) / self.warmup
        else:
            prog = (epoch - self.warmup) / max(1, self.total - self.warmup)
            lr   = self.min + 0.5 * (self.base - self.min) * (1 + math.cos(math.pi * prog))
        for pg in self.opt.param_groups:
            pg['lr'] = lr
        return lr


def run_epoch(model, loader, criterion, optimizer, scaler, device, train, desc=""):
    from tqdm import tqdm
    model.train(train)
    tot_loss = 0.0; tot_mse = 0.0; n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    # tqdm writes to stderr so it always shows in the terminal live
    pbar = tqdm(loader, desc=desc, ncols=88, leave=False, file=sys.__stderr__,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
    with ctx:
        for noisy, clean in pbar:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            if train: optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                out  = model(noisy)
                loss, comps = criterion(out, clean)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            tot_loss += loss.item(); tot_mse += comps['mse']; n += 1
            pbar.set_postfix({'loss': f'{tot_loss/n:.4f}'})
    pbar.close()
    return tot_loss / max(n,1), tot_mse / max(n,1)


def restore_batch(model, X_noisy, device, batch_size=64):
    """X_noisy [N,T,C] -> X_restored [N,T,C]"""
    model.eval()
    Xt = torch.from_numpy(np.ascontiguousarray(X_noisy.transpose(0,2,1)).astype(np.float32))
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xt), batch_size):
            b   = Xt[i:i+batch_size].to(device)
            outs.append(model(b).cpu())
    return torch.cat(outs, dim=0).permute(0,2,1).numpy()


def compute_snr(clean, noisy_or_restored):
    noise = noisy_or_restored - clean
    sp = np.mean(clean**2); np_ = np.mean(noise**2)
    return 10*np.log10(sp / max(np_, 1e-10))

def compute_corr(a, b):
    r, _ = stats.pearsonr(a.flatten(), b.flatten())
    return float(r)

def compute_rmse(a, b):
    return float(np.sqrt(np.mean((a-b)**2)))


# ==============================================================================
# SECTION 7 -- VISUALISATION
# ==============================================================================

def save_training_curve(history, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#1a1a2e')
        for ax in axes:
            ax.set_facecolor('#16213e')
            for spine in ax.spines.values(): spine.set_edgecolor('#444')
            ax.tick_params(colors='white'); ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white'); ax.title.set_color('white')

        ep = range(1, len(history['train_loss'])+1)
        axes[0].plot(ep, history['train_loss'], color='#e94560', lw=2, label='Train')
        axes[0].plot(ep, history['val_loss'],   color='#4ecca3', lw=2, label='Val')
        axes[0].set_title('Total Loss'); axes[0].set_xlabel('Epoch')
        axes[0].legend(facecolor='#1a1a2e', labelcolor='white'); axes[0].grid(alpha=0.2)

        axes[1].plot(ep, history['train_mse'], color='#e94560', lw=2, label='Train MSE')
        axes[1].plot(ep, history['val_mse'],   color='#4ecca3', lw=2, label='Val MSE')
        axes[1].set_title('MSE (Time Domain)'); axes[1].set_xlabel('Epoch')
        axes[1].legend(facecolor='#1a1a2e', labelcolor='white'); axes[1].grid(alpha=0.2)

        axes[2].plot(ep, history['lr'], color='#f5a623', lw=2)
        axes[2].set_title('Learning Rate (Warmup + Cosine)')
        axes[2].set_xlabel('Epoch'); axes[2].grid(alpha=0.2)

        fig.suptitle('GRAN v2.0  --  Training History', color='white', fontsize=14)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close(fig)
        print(f"[Plot] Training curve -> {out_path}")
    except Exception as e:
        print(f"[Plot] Could not save training curve: {e}")


def save_signal_plot(X_clean, X_noisy, X_restored, sample_idx, out_path, fs=250):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        ch_names = ['C3', 'Cz', 'C4']
        t = np.arange(1000) / fs

        fig, axes = plt.subplots(3, 3, figsize=(20, 10), facecolor='#1a1a2e')
        col_labels = ['Noisy Input', 'GRAN Restored', 'Clean Reference']
        colors     = ['#e94560', '#4ecca3', '#f5a623']

        for row in range(3):
            for col in range(3):
                ax = axes[row][col]
                ax.set_facecolor('#16213e')
                for spine in ax.spines.values(): spine.set_edgecolor('#333')
                ax.tick_params(colors='white')

                if col == 0: data = X_noisy[sample_idx, :, row]
                elif col == 1: data = X_restored[sample_idx, :, row]
                else: data = X_clean[sample_idx, :, row]

                ax.plot(t, data, color=colors[col], lw=0.8, alpha=0.9)
                if row == 0: ax.set_title(col_labels[col], color='white', fontsize=11, fontweight='bold')
                if col == 0: ax.set_ylabel(ch_names[row], color='white', fontsize=10)
                if row == 2: ax.set_xlabel('Time (s)', color='white', fontsize=9)
                ax.grid(alpha=0.15)

        fig.suptitle('GRAN v2.0  --  EEG Signal Comparison (4 second window)',
                     color='white', fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close(fig)
        print(f"[Plot] Signal comparison -> {out_path}")
    except Exception as e:
        print(f"[Plot] Could not save signal plot: {e}")


def save_spectrum_plot(X_clean, X_noisy, X_restored, sample_idx, out_path, fs=250):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.signal import welch

        ch_names = ['C3', 'Cz', 'C4']
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#1a1a2e')

        for ci in range(3):
            ax = axes[ci]
            ax.set_facecolor('#16213e')
            for spine in ax.spines.values(): spine.set_edgecolor('#333')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

            for sig, col, lbl in [
                (X_noisy[sample_idx, :, ci],    '#e94560', 'Noisy'),
                (X_restored[sample_idx, :, ci], '#4ecca3', 'Restored'),
                (X_clean[sample_idx, :, ci],    '#f5a623', 'Clean')]:
                f, psd = welch(sig, fs=fs, nperseg=256)
                ax.semilogy(f, psd, color=col, lw=1.5, label=lbl, alpha=0.85)

            ax.set_xlim(0, 50); ax.set_title(f'PSD  {ch_names[ci]}')
            ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Power (dB/Hz)')
            ax.legend(facecolor='#1a1a2e', labelcolor='white'); ax.grid(alpha=0.2)

        fig.suptitle('GRAN v2.0  --  Power Spectral Density Comparison',
                     color='white', fontsize=13)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close(fig)
        print(f"[Plot] PSD plot -> {out_path}")
    except Exception as e:
        print(f"[Plot] Could not save spectrum plot: {e}")


# ==============================================================================
# SECTION 8 -- MAIN TRAINING LOOP
# ==============================================================================

def main(args):
    t_start = time.time()

    SEED          = 42
    EPOCHS        = args.epochs
    BATCH_SIZE    = args.batch_size
    LR            = 2e-4
    WEIGHT_DECAY  = 1e-4
    WARMUP        = 5
    PATIENCE      = args.patience
    BCI_TRAIN     = [f"A0{i}T" for i in range(1, 8)]
    BCI_VAL       = ["A08T", "A09T"]
    SLEEP_FILES   = 0 if args.bci_only else 25
    SLEEP_MAX_WIN = 1500

    np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("  GRAN v2.0 -- EEG Restoration Training")
    print("=" * 70)
    print(f"  Device        : {device}")
    print(f"  Epochs        : {EPOCHS}   patience={PATIENCE}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  BCI-only mode : {args.bci_only}")
    print(f"  Resume        : {args.resume}")
    print(f"  Log file      : {LOG_FILE}")
    print("=" * 70)

    # ── STEP 1: Load data ────────────────────────────────────────────────────
    print("\n[1/6] Loading BCI Competition IV ...")
    X_bci_train, X_bci_val = load_bci(BCI_DIR, BCI_TRAIN, BCI_VAL, verbose=True)
    print(f"  BCI train : {X_bci_train.shape}   val : {X_bci_val.shape}")

    print("\n[2/6] Loading Sleep-EDF ...")
    if SLEEP_FILES > 0 and os.path.isdir(SLEEP_DIR):
        X_sleep = load_sleep_edf(SLEEP_DIR, n_files=SLEEP_FILES,
                                 max_per_file=SLEEP_MAX_WIN, seed=SEED, verbose=True)
        ns = len(X_sleep)
        ntr = int(0.80 * ns)
        perm = np.random.permutation(ns)
        X_sleep_train = X_sleep[perm[:ntr]]
        X_sleep_val   = X_sleep[perm[ntr:]]
        print(f"  Sleep train : {X_sleep_train.shape}   val : {X_sleep_val.shape}")
    else:
        print("  Sleep-EDF not found or disabled.")
        X_sleep_train = np.empty((0,1000,3), dtype=np.float32)
        X_sleep_val   = np.empty((0,1000,3), dtype=np.float32)

    # Combine
    X_train_clean = np.concatenate([X_bci_train, X_sleep_train], axis=0)
    X_val_clean   = np.concatenate([X_bci_val,   X_sleep_val],   axis=0)
    perm          = np.random.permutation(len(X_train_clean))
    X_train_clean = X_train_clean[perm]
    print(f"\n  Combined train : {X_train_clean.shape}")
    print(f"  Combined val   : {X_val_clean.shape}")

    # ── STEP 2: Generate artifacts ───────────────────────────────────────────
    print("\n[3/6] Generating artifacts (7 types) ...")
    gen = ArtifactGenerator(fs=250, seed=SEED)
    print("  Corrupting training set ...")
    X_train_noisy = gen.corrupt_batch(X_train_clean, artifact_prob=0.92, verbose=True)
    print("  Corrupting validation set ...")
    X_val_noisy   = gen.corrupt_batch(X_val_clean,   artifact_prob=0.92, verbose=True)

    snr_in = compute_snr(X_train_clean, X_train_noisy)
    print(f"  Input SNR (train) : {snr_in:.2f} dB")

    # Free numpy arrays we no longer need on CPU
    del gen; gc.collect()

    # ── STEP 3: Build model ──────────────────────────────────────────────────
    print("\n[4/6] Building GRAN v2.0 model ...")
    model = GRANModel(n_channels=3, n_samples=1000, base_ch=32, dropout=0.10)
    model = model.to(device)
    print(f"  Trainable parameters : {model.n_params():,}")

    criterion = GRANLoss(mse_w=1.0, l1_w=0.10, freq_w=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=1e-8)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    scheduler = WarmupCosineScheduler(optimizer, WARMUP, EPOCHS, LR, min_lr=1e-6)

    history = {'train_loss':[], 'val_loss':[], 'train_mse':[], 'val_mse':[], 'lr':[]}
    start_epoch    = 0
    best_val_loss  = float('inf')
    best_state     = None
    patience_ctr   = 0

    # ── Resume from checkpoint ───────────────────────────────────────────────
    resume_path = os.path.join(CKPT_DIR, 'gran_resume.pth')
    if args.resume and os.path.exists(resume_path):
        print(f"  Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        history       = ckpt.get('history', history)
        start_epoch   = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        patience_ctr  = ckpt.get('patience_ctr', 0)
        print(f"  Resumed at epoch {start_epoch}  best_val={best_val_loss:.6f}")

    # ── STEP 4: Training loop ────────────────────────────────────────────────
    print(f"\n[5/6] Training for up to {EPOCHS} epochs (patience={PATIENCE}) ...")
    print("-" * 70)
    print(f"{'Ep':>4} | {'LR':>8} | {'Train':>9} | {'Val':>9} | "
          f"{'Best':>9} | {'P':>3} | {'Corr':>6} | {'SNR+':>6}")
    print("-" * 70)

    batch_size = BATCH_SIZE   # may be halved on OOM

    for epoch in range(start_epoch, EPOCHS):
        current_lr = scheduler.step(epoch)

        # Build DataLoaders (rebuild each epoch so aug seeds vary)
        train_ds = AugEEGDataset(X_train_noisy, X_train_clean, augment=True)
        val_ds   = AugEEGDataset(X_val_noisy,   X_val_clean,   augment=False)

        # Auto-retry with smaller batch on OOM
        while True:
            try:
                train_dl = DataLoader(train_ds, batch_size=batch_size,
                                      shuffle=True, num_workers=0, pin_memory=False)
                val_dl   = DataLoader(val_ds,   batch_size=batch_size,
                                      shuffle=False, num_workers=0, pin_memory=False)
                t_loss, t_mse = run_epoch(model, train_dl, criterion, optimizer, scaler, device, train=True,  desc=f"  Ep{epoch+1:3d} Train")
                v_loss, v_mse = run_epoch(model, val_dl,   criterion, optimizer, scaler, device, train=False, desc=f"  Ep{epoch+1:3d} Val  ")
                break
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and batch_size > 4:
                    batch_size //= 2
                    print(f"\n  [OOM] Reduced batch size -> {batch_size}")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                else:
                    raise

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_mse'].append(t_mse)
        history['val_mse'].append(v_mse)
        history['lr'].append(current_lr)

        # Compute quick correlation on 500-sample subset for live reporting
        sub = min(500, len(X_val_clean))
        idxs = np.random.choice(len(X_val_clean), sub, replace=False)
        X_sub_r = restore_batch(model, X_val_noisy[idxs], device, batch_size=64)
        corr = compute_corr(X_val_clean[idxs], X_sub_r)
        snr_before = compute_snr(X_val_clean[idxs], X_val_noisy[idxs])
        snr_after  = compute_snr(X_val_clean[idxs], X_sub_r)
        snr_imp    = snr_after - snr_before

        # Track best model
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
            # Save best checkpoint
            torch.save({'epoch': epoch, 'model_state': best_state,
                        'val_loss': best_val_loss, 'history': history},
                       os.path.join(CKPT_DIR, 'gran_best.pth'))
            flag = " <-- BEST"
        else:
            patience_ctr += 1
            flag = ""

        print(f"{epoch+1:4d} | {current_lr:8.2e} | {t_loss:9.5f} | {v_loss:9.5f} | "
              f"{best_val_loss:9.5f} | {patience_ctr:3d} | {corr:6.4f} | {snr_imp:+6.2f}{flag}")
        sys.stdout.flush()

        # Periodic safe checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'best_val_loss': best_val_loss, 'patience_ctr': patience_ctr,
                        'history': history},
                       resume_path)

        if patience_ctr >= PATIENCE:
            print(f"\n  [Early Stop] No val improvement for {PATIENCE} epochs.")
            break

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)
        print(f"\nRestored best model  (val_loss={best_val_loss:.6f})")

    # Save final model
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, 'gran_final.pth'))
    print(f"Final model saved -> {os.path.join(CKPT_DIR, 'gran_final.pth')}")

    # ── STEP 5: Full evaluation ──────────────────────────────────────────────
    print("\n[6/6] Full evaluation on validation set ...")
    X_val_restored = restore_batch(model, X_val_noisy, device, batch_size=64)

    snr_noisy    = compute_snr(X_val_clean, X_val_noisy)
    snr_restored = compute_snr(X_val_clean, X_val_restored)
    snr_imp      = snr_restored - snr_noisy
    corr_noisy   = compute_corr(X_val_clean, X_val_noisy)
    corr_rest    = compute_corr(X_val_clean, X_val_restored)
    rmse_noisy   = compute_rmse(X_val_clean, X_val_noisy)
    rmse_rest    = compute_rmse(X_val_clean, X_val_restored)
    mae_noisy    = float(np.mean(np.abs(X_val_clean - X_val_noisy)))
    mae_rest     = float(np.mean(np.abs(X_val_clean - X_val_restored)))

    per_ch = {}
    for ci, ch in enumerate(['C3','Cz','C4']):
        cl = X_val_clean[:, :, ci]; ns = X_val_noisy[:,:,ci]; rs = X_val_restored[:,:,ci]
        per_ch[ch] = {
            'snr_noisy'    : compute_snr(cl, ns),
            'snr_restored' : compute_snr(cl, rs),
            'snr_improvement' : compute_snr(cl,rs) - compute_snr(cl,ns),
            'corr_noisy'   : compute_corr(cl, ns),
            'corr_restored': compute_corr(cl, rs),
            'rmse_noisy'   : compute_rmse(cl, ns),
            'rmse_restored': compute_rmse(cl, rs),
        }

    metrics = {
        'snr_noisy': snr_noisy, 'snr_restored': snr_restored,
        'snr_improvement': snr_imp,
        'correlation_noisy': corr_noisy, 'correlation_restored': corr_rest,
        'rmse_noisy': rmse_noisy, 'rmse_restored': rmse_rest,
        'mae_noisy': mae_noisy, 'mae_restored': mae_rest,
        'n_params': model.n_params(),
        'total_train_windows': int(len(X_train_clean)),
        'epochs_trained': len(history['train_loss']),
    }

    # Save JSON
    with open(os.path.join(OUT_DIR, 'metrics_v2.json'), 'w') as f:
        json.dump({k: float(v) if not isinstance(v, str) else v for k, v in metrics.items()}, f, indent=2)
    with open(os.path.join(OUT_DIR, 'per_channel_metrics.json'), 'w') as f:
        json.dump({ch: {k: float(v) for k,v in d.items()} for ch,d in per_ch.items()}, f, indent=2)

    # ── STEP 6: Visualisations ───────────────────────────────────────────────
    save_training_curve(history, os.path.join(OUT_DIR, 'training_curve_v2.png'))

    # Pick window with moderate artifact (not the noisiest — more informative visually)
    art_energy = np.mean(np.abs(X_val_noisy - X_val_clean), axis=(1,2))
    pct75 = np.percentile(art_energy, 75)
    good  = np.where(art_energy >= pct75)[0]
    sidx  = int(good[len(good)//2])   # median of top 25%

    save_signal_plot(X_val_clean, X_val_noisy, X_val_restored, sidx,
                     os.path.join(OUT_DIR, 'signal_comparison.png'))
    save_spectrum_plot(X_val_clean, X_val_noisy, X_val_restored, sidx,
                       os.path.join(OUT_DIR, 'spectrum_comparison.png'))

    # ── Final report ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    sep = "=" * 70
    print(f"\n{sep}")
    print("  GRAN v2.0  --  FINAL RESULTS")
    print(sep)
    print(f"  {'Metric':<35} {'Noisy':>10}  {'Restored':>10}")
    print(f"  {'-'*55}")
    print(f"  {'SNR (dB)':<35} {snr_noisy:>10.2f}  {snr_restored:>10.2f}  (+{snr_imp:.2f} dB)")
    print(f"  {'Pearson Correlation':<35} {corr_noisy:>10.4f}  {corr_rest:>10.4f}")
    print(f"  {'RMSE':<35} {rmse_noisy:>10.4f}  {rmse_rest:>10.4f}  "
          f"({(1-rmse_rest/rmse_noisy)*100:.1f}% reduction)")
    print(f"  {'MAE':<35} {mae_noisy:>10.4f}  {mae_rest:>10.4f}  "
          f"({(1-mae_rest/mae_noisy)*100:.1f}% reduction)")
    print()
    print("  Per-channel SNR improvement:")
    for ch, d in per_ch.items():
        print(f"    {ch}: {d['snr_improvement']:+.2f} dB  (corr={d['corr_restored']:.4f}  "
              f"rmse={d['rmse_restored']:.4f})")
    print()
    print(f"  Model parameters       : {model.n_params():,}")
    print(f"  Train windows used     : {len(X_train_clean):,}")
    print(f"  Epochs trained         : {len(history['train_loss'])}")
    print(f"  Best val loss          : {best_val_loss:.6f}")
    print(f"  Total time             : {elapsed/60:.1f} min")
    print()
    print("  Output files:")
    for fname in sorted(os.listdir(OUT_DIR)):
        fp = os.path.join(OUT_DIR, fname)
        kb = os.path.getsize(fp) / 1024
        print(f"    {fname:<45} ({kb:8.1f} KB)")
    print(f"  Log file: {LOG_FILE}")
    print(sep)
    print("  Training complete.  Open outputs_v2/ to view results.")
    print(sep)

    sys.stdout.flush()
    _tee_out.close()
    return metrics


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRAN v2.0 Training Script")
    parser.add_argument('--bci-only',  action='store_true',
                        help='Skip Sleep-EDF, use BCI data only (faster)')
    parser.add_argument('--resume',    action='store_true',
                        help='Resume from last safe checkpoint')
    parser.add_argument('--epochs',    type=int, default=100,
                        help='Maximum training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Initial batch size (auto-halves on GPU OOM, default: 64)')
    parser.add_argument('--patience',  type=int, default=20,
                        help='Early stopping patience (default: 20)')
    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Checkpoints saved to: " + CKPT_DIR)
        _tee_out.close()
        sys.exit(0)
    except Exception as exc:
        print("\n\n[FATAL ERROR]")
        traceback.print_exc()
        _tee_out.close()
        sys.exit(1)
