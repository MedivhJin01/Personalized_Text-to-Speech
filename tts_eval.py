
"""
tts_eval.py — Objective metrics for TTS evaluation (mel + audio).

Metrics implemented
-------------------
Mel-space (requires reference & predicted mels OR we compute from audio)
- mel_L1, mel_L2
- mel_cosine (1 - cosine similarity)
- mel_spectral_convergence
- (optional) DTW alignment over time before computing the above

Audio-space
- MCD (Mel Cepstral Distortion, dB) using DTW over MFCC (exclude c0)
- LSD (Log-Spectral Distance) with DTW
- F0 metrics using librosa.pyin: RMSE, MAE, Pearson corr, Voicing Decision Error
- Multi-resolution STFT losses: spectral convergence & log-mag distance
- Optional: PESQ (narrow/wide band) and STOI, if packages installed

CLI usage
---------
python tts_eval.py --ref_dir <ref_wavs> --syn_dir <syn_wavs> --sr 22050 \
    --compute_mel --save_csv results.csv

If you have pre-computed mels (as .npy tensors BxTxC), you can pass directories:
python tts_eval.py --ref_mel_dir <ref_mels> --syn_mel_dir <syn_mels> --save_csv mel_eval.csv

Notes
-----
• Metrics are *proxies*; human MOS is still the gold standard.
• For MCD we follow the common formula: 10/ln(10) * sqrt(2) * mean||Δmfcc||_2 (exclude c0).
• DTW uses Euclidean distance and returns the mean along the alignment path.
• F0 metrics rely on pyin; adjust fmin/fmax to match your data if needed.
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa
import librosa.display  # noqa: F401 (needed if user plots externally)
from librosa.sequence import dtw as librosa_dtw
import scipy.signal
import scipy.stats

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

# Optional metrics
try:
    from pesq import pesq  # type: ignore
except Exception:  # pragma: no cover
    pesq = None

try:
    from pystoi import stoi  # type: ignore
except Exception:  # pragma: no cover
    stoi = None


# ----------------------------- Helpers ----------------------------------

def _safe_log(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return np.log(np.maximum(x, eps))


def _pad_trim(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Center-trim or pad b to match a along time (axis=1 for [freq, time])."""
    T = a.shape[-1]
    S = b.shape[-1]
    if S == T:
        return a, b
    if S > T:
        # center trim
        start = (S - T) // 2
        return a, b[..., start:start + T]
    else:
        pad = T - S
        left = pad // 2
        right = pad - left
        return a, np.pad(b, ((0, 0), (left, right)), mode="constant")


def _dtw_align(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """DTW-align two time sequences (feature x time). Returns warped sequences.
    Uses Euclidean distance on transposed (time x feat)."""
    # librosa expects [frames, features]
    D, wp = librosa_dtw(A.T, B.T, metric="euclidean")
    # warping path is list of (i, j) indices from start to end
    path = np.array(wp[::-1])  # from start to end
    Aw = A[:, path[:, 0]]
    Bw = B[:, path[:, 1]]
    return Aw, Bw


def _mel_from_audio(y: np.ndarray, sr: int, n_mels: int = 80,
                    n_fft: int = 1024, hop_length: int = 256,
                    win_length: Optional[int] = None,
                    fmin: int = 0, fmax: Optional[int] = 8000) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, n_mels=n_mels, fmin=fmin, fmax=fmax, power=1.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db  # [n_mels, T] in dB


# -------------------------- Mel-space metrics ---------------------------

def mel_metrics(ref_mel: np.ndarray, syn_mel: np.ndarray, use_dtw: bool = True) -> Dict[str, float]:
    """Compute mel-space distances; inputs are [n_mels, T] (dB or linear ok).
    If shapes mismatch, we DTW-align by default; else we pad/trim.
    """
    assert ref_mel.ndim == 2 and syn_mel.ndim == 2
    A, B = ref_mel.copy(), syn_mel.copy()
    if use_dtw:
        A, B = _dtw_align(A, B)
    else:
        A, B = _pad_trim(A, B)

    # L1, L2
    l1 = float(np.mean(np.abs(A - B)))
    l2 = float(np.mean((A - B) ** 2) ** 0.5)

    # Cosine (per-frame, then mean) -> 1 - cosine similarity
    eps = 1e-10
    An = A / (np.linalg.norm(A, axis=0, keepdims=True) + eps)
    Bn = B / (np.linalg.norm(B, axis=0, keepdims=True) + eps)
    cos = float(1.0 - np.mean(np.sum(An * Bn, axis=0)))

    # Spectral convergence (||A-B||_F / ||A||_F)
    sc = float(np.linalg.norm(A - B, ord="fro") / (np.linalg.norm(A, ord="fro") + eps))

    return {
        "mel_L1": l1,
        "mel_L2": l2,
        "mel_cosine": cos,
        "mel_spectral_convergence": sc,
    }


# --------------------------- Audio metrics ------------------------------

def mfcc_features(y: np.ndarray, sr: int, n_mfcc: int = 13,
                  n_fft: int = 1024, hop_length: int = 256) -> np.ndarray:
    """Librosa MFCCs (include c0 at index 0)."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=40)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=n_mfcc)
    return mfcc  # [n_mfcc, T]


def mcd(ref: np.ndarray, syn: np.ndarray, sr: int,
        n_mfcc: int = 13, exclude_c0: bool = True) -> float:
    """Mel Cepstral Distortion in dB, DTW-aligned over MFCCs."""
    A = mfcc_features(ref, sr, n_mfcc=n_mfcc)
    B = mfcc_features(syn, sr, n_mfcc=n_mfcc)
    if exclude_c0:
        A, B = A[1:, :], B[1:, :]

    Aw, Bw = _dtw_align(A, B)  # [K, T'] each
    diff = Aw - Bw
    # Factor 10 / ln(10) * sqrt(2)
    const = 10.0 / math.log(10.0) * math.sqrt(2.0)
    dist = const * np.mean(np.linalg.norm(diff, axis=0))
    return float(dist)


def lsd(ref: np.ndarray, syn: np.ndarray, sr: int,
        n_fft: int = 2048, hop_length: int = 512) -> float:
    """Log Spectral Distance using magnitude STFTs with DTW alignment."""
    A = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop_length))
    B = np.abs(librosa.stft(syn, n_fft=n_fft, hop_length=hop_length))
    A_log = np.log10(np.maximum(A, 1e-10))
    B_log = np.log10(np.maximum(B, 1e-10))

    Aw, Bw = _dtw_align(A_log, B_log)  # [F, T']
    # Per-frame L2 over freq, then mean
    frame_l2 = np.sqrt(np.mean((Aw - Bw) ** 2, axis=0))
    return float(np.mean(frame_l2))


def f0_track(y: np.ndarray, sr: int,
             frame_length: int = 1024, hop_length: int = 256,
             fmin: float = 50.0, fmax: float = 600.0) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate F0 via librosa.pyin. Returns (f0_Hz, voiced_mask)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, vflag, _ = librosa.pyin(
            y, fmin=fmin, fmax=fmax,
            frame_length=frame_length, hop_length=hop_length,
        )
    f0 = f0.astype(np.float32)
    vmask = (vflag.astype(bool)) & np.isfinite(f0)
    return f0, vmask


def f0_metrics(ref: np.ndarray, syn: np.ndarray, sr: int,
               frame_length: int = 1024, hop_length: int = 256,
               fmin: float = 50.0, fmax: float = 600.0) -> Dict[str, float]:
    """F0 RMSE/MAE/corr on voiced frames + Voicing Decision Error."""
    f0_r, v_r = f0_track(ref, sr, frame_length, hop_length, fmin, fmax)
    f0_s, v_s = f0_track(syn, sr, frame_length, hop_length, fmin, fmax)

    # Align length
    T = min(len(f0_r), len(f0_s))
    f0_r, f0_s = f0_r[:T], f0_s[:T]
    v_r, v_s = v_r[:T], v_s[:T]

    # VDE: fraction of frames where voiced/unvoiced decision differs
    vde = float(np.mean(v_r != v_s))

    # On common voiced frames
    both_voiced = v_r & v_s
    if np.any(both_voiced):
        diff = f0_r[both_voiced] - f0_s[both_voiced]
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))
        if len(diff) >= 2:
            corr = float(np.corrcoef(f0_r[both_voiced], f0_s[both_voiced])[0, 1])
        else:
            corr = float("nan")
    else:
        rmse = mae = float("nan")
        corr = float("nan")

    return {
        "f0_rmse": rmse,
        "f0_mae": mae,
        "f0_corr": corr,
        "vde": vde,
    }


def multi_res_stft_losses(ref: np.ndarray, syn: np.ndarray, sr: int,
                          fft_sizes=(1024, 2048, 512), hop_lengths=(256, 512, 128),
                          win_lengths=(1024, 2048, 512)) -> Dict[str, float]:
    """Multi-resolution STFT losses: spectral convergence & log-mag distance."""
    eps = 1e-7
    sc_list = []
    lmag_list = []
    for n_fft, hop, win in zip(fft_sizes, hop_lengths, win_lengths):
        A = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop, win_length=win))
        B = np.abs(librosa.stft(syn, n_fft=n_fft, hop_length=hop, win_length=win))
        # Trim/pad to same T
        A, B = _pad_trim(A, B)
        sc = np.linalg.norm(A - B, 'fro') / (np.linalg.norm(A, 'fro') + eps)
        lmag = np.mean(np.abs(_safe_log(A) - _safe_log(B)))
        sc_list.append(sc)
        lmag_list.append(lmag)
    return {
        "mrstft_sc": float(np.mean(sc_list)),
        "mrstft_logmag": float(np.mean(lmag_list)),
    }


def optional_pesq_stoi(ref: np.ndarray, syn: np.ndarray, sr: int) -> Dict[str, Optional[float]]:
    out = {"pesq_nb": None, "pesq_wb": None, "stoi": None}
    # PESQ: requires specific sampling rates (8k for NB, 16k for WB in the package)
    if pesq is not None:
        try:
            if sr == 8000:
                out["pesq_nb"] = float(pesq(sr, ref, syn, 'nb'))
            if sr == 16000:
                out["pesq_wb"] = float(pesq(sr, ref, syn, 'wb'))
        except Exception:
            pass
    if stoi is not None:
        try:
            out["stoi"] = float(stoi(ref, syn, sr, extended=False))
        except Exception:
            pass
    return out


# ----------------------- Dataset-level evaluation -----------------------

def pair_paths(ref_dir: Path, syn_dir: Path, exts=(".wav", ".flac")) -> List[Tuple[Path, Path]]:
    ref_map = {}
    for p in ref_dir.rglob("*"):
        if p.suffix.lower() in exts:
            ref_map[p.stem] = p
    pairs = []
    for q in syn_dir.rglob("*"):
        if q.suffix.lower() in exts and q.stem in ref_map:
            pairs.append((ref_map[q.stem], q))
    return sorted(pairs, key=lambda x: x[0].name)


def evaluate_pair(ref_audio: np.ndarray, syn_audio: np.ndarray, sr: int,
                  include_pesq_stoi: bool = True) -> Dict[str, float]:
    res: Dict[str, float] = {}

    # Audio metrics
    res["MCD"] = mcd(ref_audio, syn_audio, sr)
    res["LSD"] = lsd(ref_audio, syn_audio, sr)
    res.update(f0_metrics(ref_audio, syn_audio, sr))
    res.update(multi_res_stft_losses(ref_audio, syn_audio, sr))

    if include_pesq_stoi:
        res.update(optional_pesq_stoi(ref_audio, syn_audio, sr))

    return res


def evaluate_dirs(ref_dir: Path, syn_dir: Path, sr: int,
                  include_pesq_stoi: bool = True) -> "pd.DataFrame":
    assert pd is not None, "pandas is required for directory evaluation"
    pairs = pair_paths(ref_dir, syn_dir)
    rows = []
    for ref_p, syn_p in pairs:
        ref, _ = librosa.load(ref_p, sr=sr, mono=True)
        syn, _ = librosa.load(syn_p, sr=sr, mono=True)
        metrics = evaluate_pair(ref, syn, sr, include_pesq_stoi=include_pesq_stoi)
        metrics["utterance"] = ref_p.stem
        rows.append(metrics)
    df = pd.DataFrame(rows).set_index("utterance")
    return df


def evaluate_mels(ref_mel: np.ndarray, syn_mel: np.ndarray, dtw: bool = True) -> Dict[str, float]:
    return mel_metrics(ref_mel, syn_mel, use_dtw=dtw)


# ------------------------------ CLI ------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TTS objective metrics (mel + audio)")
    parser.add_argument("--ref_dir", type=str, default=None, help="Directory of reference wavs")
    parser.add_argument("--syn_dir", type=str, default=None, help="Directory of synthesized wavs")
    parser.add_argument("--sr", type=int, default=22050, help="Target sampling rate for loading audio")
    parser.add_argument("--include_pesq_stoi", action="store_true", help="Compute PESQ/STOI if available")
    parser.add_argument("--save_csv", type=str, default=None, help="Where to save the aggregated CSV")
    parser.add_argument("--ref_mel_dir", type=str, default=None, help="Directory of reference mels (.npy shaped [n_mels,T])")
    parser.add_argument("--syn_mel_dir", type=str, default=None, help="Directory of synthesized mels (.npy shaped [n_mels,T])")
    parser.add_argument("--no_dtw", action="store_true", help="Disable DTW for mel metrics")
    args = parser.parse_args()

    if args.ref_dir and args.syn_dir:
        ref_dir = Path(args.ref_dir)
        syn_dir = Path(args.syn_dir)
        df = evaluate_dirs(ref_dir, syn_dir, sr=args.sr, include_pesq_stoi=args.include_pesq_stoi)
        print(df.describe().T)
        if args.save_csv:
            df.to_csv(args.save_csv, index=True)
            print(f"Saved: {args.save_csv}")

    if args.ref_mel_dir and args.syn_mel_dir:
        if pd is None:
            raise RuntimeError("pandas required for mel-dir evaluation")
        rows = []
        ref_map = {p.stem: p for p in Path(args.ref_mel_dir).rglob("*.npy")}
        for q in Path(args.syn_mel_dir).rglob("*.npy"):
            if q.stem not in ref_map:
                continue
            A = np.load(ref_map[q.stem])  # [n_mels, T]
            B = np.load(q)
            metrics = evaluate_mels(A, B, dtw=not args.no_dtw)
            metrics["utterance"] = q.stem
            rows.append(metrics)
        if rows:
            df = pd.DataFrame(rows).set_index("utterance")
            print(df.describe().T)
            if args.save_csv:
                df.to_csv(args.save_csv, index=True)
                print(f"Saved: {args.save_csv}")

if __name__ == "__main__":
    main()
