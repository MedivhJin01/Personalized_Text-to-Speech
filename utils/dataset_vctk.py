import os
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

# --- Audio Config ---
SR = 22050
N_MELS = 80
N_FFT = 512
HOP = 256
MIN_LEVEL = 1e-5


# --- Text Mapping ---
_SYMBOLS = ["<pad>", "<unk>"] + list("abcdefghijklmnopqrstuvwxyz!'?,.- ")
CHAR2IDX = {c: i for i, c in enumerate(_SYMBOLS)}

def text_to_sequence(text: str) -> List[int]:
    text = text.lower()
    return [CHAR2IDX.get(ch, CHAR2IDX["<unk>"]) for ch in text]

def wav_to_mel(wav: np.ndarray, sr: int = SR) -> torch.Tensor:
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=1.0
    )
    mel = np.log(np.maximum(mel, MIN_LEVEL))
    return torch.from_numpy(mel).float()  # [n_mels, T]

class VCTK_Dataset(Dataset):
    """
    Dataset for VCTK, using only *_mic1.flac files.
    Returns: (text_token_ids, mel_spectrogram, speaker_id)
    """
    def __init__(self, root: str, sample_rate=SR, limit_speakers=None, cache_mels=True):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.cache_mels = cache_mels
        self.items = []

        wav_dir = self.root / "wav48_silence_trimmed"
        txt_dir = self.root / "txt"
        speakers = sorted([d for d in os.listdir(wav_dir) if (wav_dir / d).is_dir()])

        if limit_speakers:
            speakers = speakers[:limit_speakers]

        self.spk2id = {spk: i for i, spk in enumerate(speakers)}

        for spk in speakers:
            wav_spk_dir = wav_dir / spk
            txt_spk_dir = txt_dir / spk
            for fname in os.listdir(wav_spk_dir):
                if fname.endswith("_mic1.flac"):
                    wav_path = wav_spk_dir / fname
                    txt_path = txt_spk_dir / fname.replace("_mic1.flac", ".txt")
                    if txt_path.exists():
                        self.items.append((wav_path, txt_path, self.spk2id[spk]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, txt_path, spk_id = self.items[idx]
        text = Path(txt_path).read_text().strip()
        text_ids = torch.LongTensor(text_to_sequence(text))

        # Load and resample
        wav, sr = librosa.load(str(wav_path), sr=None)
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)

        # Load mel from cache or extract
        if self.cache_mels:
            cache_path = wav_path.with_suffix(".npy").as_posix().replace("wav48_silence_trimmed", "mel_cache")
            mel_path = Path(cache_path)
            if mel_path.exists():
                mel = torch.from_numpy(np.load(mel_path)).float()
            else:
                mel = wav_to_mel(wav, sr=self.sample_rate)
                mel_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(mel_path, mel.numpy())
        else:
            mel = wav_to_mel(wav, sr=self.sample_rate)

        return text_ids, mel, torch.LongTensor([spk_id])

    @staticmethod
    def collate_fn(batch):
        texts, mels, spks = zip(*batch)
        text_lens = torch.LongTensor([t.size(0) for t in texts])
        mel_lens = torch.LongTensor([m.size(1) for m in mels])

        max_text = int(text_lens.max().item())
        max_mel = int(mel_lens.max().item())

        text_pad = torch.zeros(len(batch), max_text, dtype=torch.long)
        mel_pad = torch.zeros(len(batch), N_MELS, max_mel)

        for i, (t, m) in enumerate(zip(texts, mels)):
            text_pad[i, : t.size(0)] = t
            mel_pad[i, :, : m.size(1)] = m

        spk_ids = torch.cat(spks, dim=0)
        return text_pad, text_lens, mel_pad, mel_lens, spk_ids
