# eval_pair.py (robust)
import argparse, math, warnings
import numpy as np
import librosa
from librosa.sequence import dtw as librosa_dtw

def _safe_log(x, eps=1e-10): return np.log(np.maximum(x, eps))

def _ensure_2d(X):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    return X

def _match_feat_dim(A, B):
    # Make sure feature dims (rows) match by truncating to the min
    k = min(A.shape[0], B.shape[0])
    return A[:k, :], B[:k, :]

def _center_trim_time(A, B):
    T = min(A.shape[1], B.shape[1])
    return A[:, :T], B[:, :T]

def _dtw_align(A, B, metric="euclidean"):
    # A,B: [feat, T] -> warped along time
    A = _ensure_2d(A); B = _ensure_2d(B)
    A, B = _match_feat_dim(A, B)
    try:
        _, wp = librosa_dtw(A.T, B.T, metric=metric)
        path = np.array(wp[::-1])
        return A[:, path[:, 0]], B[:, path[:, 1]]
    except Exception:
        # Fallback: center-trim (no DTW)
        return _center_trim_time(A, B)

def mfcc_features(y, sr, n_mfcc=13, n_fft=1024, hop=256, n_mels=40):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=n_mels)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=n_mfcc)
    return _ensure_2d(mfcc)  # [n_mfcc, T]

def MCD(ref, syn, sr, n_mfcc=13, exclude_c0=True, use_dtw=True):
    A = mfcc_features(ref, sr, n_mfcc=n_mfcc)
    B = mfcc_features(syn, sr, n_mfcc=n_mfcc)
    if exclude_c0:
        A, B = A[1:, :], B[1:, :]
    if use_dtw:
        Aw, Bw = _dtw_align(A, B)
    else:
        Aw, Bw = _center_trim_time(A, B)
    const = 10.0 / math.log(10.0) * math.sqrt(2.0)
    return float(const * np.mean(np.linalg.norm(Aw - Bw, axis=0)))

def LSD(ref, syn, sr, n_fft=2048, hop=512, use_dtw=True):
    A = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop))
    B = np.abs(librosa.stft(syn, n_fft=n_fft, hop_length=hop))
    A = _ensure_2d(np.log10(np.maximum(A, 1e-10)))
    B = _ensure_2d(np.log10(np.maximum(B, 1e-10)))
    if use_dtw:
        Aw, Bw = _dtw_align(A, B)
    else:
        Aw, Bw = _center_trim_time(A, B)
    return float(np.mean(np.sqrt(np.mean((Aw - Bw) ** 2, axis=0))))

def f0_track(y, sr, frame=1024, hop=256, fmin=50.0, fmax=600.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, vflag, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, frame_length=frame, hop_length=hop)
    f0 = f0.astype(np.float32)
    vmask = (vflag.astype(bool)) & np.isfinite(f0)
    return f0, vmask

def F0_metrics(ref, syn, sr, frame=1024, hop=256, fmin=50.0, fmax=600.0):
    f0_r, v_r = f0_track(ref, sr, frame, hop, fmin, fmax)
    f0_s, v_s = f0_track(syn, sr, frame, hop, fmin, fmax)
    T = min(len(f0_r), len(f0_s))
    f0_r, f0_s, v_r, v_s = f0_r[:T], f0_s[:T], v_r[:T], v_s[:T]
    vde = float(np.mean(v_r != v_s))
    both = v_r & v_s
    if np.any(both):
        d = f0_r[both] - f0_s[both]
        rmse = float(np.sqrt(np.mean(d ** 2)))
        mae  = float(np.mean(np.abs(d)))
        corr = float(np.corrcoef(f0_r[both], f0_s[both])[0, 1]) if np.sum(both) >= 2 else float("nan")
    else:
        rmse = mae = corr = float("nan")
    return {"f0_rmse": rmse, "f0_mae": mae, "f0_corr": corr, "vde": vde}

def MRSTFT(ref, syn, sr, fft_sizes=(1024, 2048, 512), hops=(256, 512, 128), wins=(1024, 2048, 512)):
    eps = 1e-7; sc_list = []; lmag_list = []
    for n_fft, hop, win in zip(fft_sizes, hops, wins):
        A = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop, win_length=win))
        B = np.abs(librosa.stft(syn, n_fft=n_fft, hop_length=hop, win_length=win))
        A, B = _center_trim_time(A, B)
        sc = np.linalg.norm(A - B, 'fro') / (np.linalg.norm(A, 'fro') + eps)
        lmag = np.mean(np.abs(_safe_log(A) - _safe_log(B)))
        sc_list.append(sc); lmag_list.append(lmag)
    return {"mrstft_sc": float(np.mean(sc_list)), "mrstft_logmag": float(np.mean(lmag_list))}

def main():
    ap = argparse.ArgumentParser("Evaluate a single ref/syn wav pair")
    ap.add_argument("--ref_wav", required=True)
    ap.add_argument("--syn_wav", required=True)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--no_dtw", action="store_true", help="Disable DTW for MCD/LSD")
    args = ap.parse_args()

    ref, _ = librosa.load(args.ref_wav, sr=args.sr, mono=True)
    syn, _ = librosa.load(args.syn_wav, sr=args.sr, mono=True)

    use_dtw = not args.no_dtw
    metrics = {
        "MCD": MCD(ref, syn, args.sr, use_dtw=use_dtw),
        "LSD": LSD(ref, syn, args.sr, use_dtw=use_dtw),
        **F0_metrics(ref, syn, args.sr),
        **MRSTFT(ref, syn, args.sr),
    }

    print("=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
