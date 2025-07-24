from pathlib import Path
import librosa, numpy as np, torch, os
from torch.utils.data import Dataset
from utils.text import text_to_sequence
from utils.tools import pad_1D, pad_2D

SR = 22050
N_MELS, N_FFT, HOP = 80, 1024, 256
MIN_LEVEL = 1e-5
N_FRAMES_PER_STEP = 2          # Tacotron2 default is 2

def wav_to_mel(wav, sr=SR):
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, power=1.0
    )
    mel = np.log(np.maximum(mel, MIN_LEVEL))
    return torch.from_numpy(mel).float()    # [80, T]

class VCTKDataset(Dataset):
    def __init__(self, root, limit_speakers=None, cache_mel=True):
        self.root = Path(root)
        self.cache_mel = cache_mel
        wav_root = self.root / "wav48_silence_trimmed"
        txt_root = self.root / "txt"

        speakers = sorted([d for d in os.listdir(wav_root) if (wav_root/d).is_dir()])
        if limit_speakers:
            speakers = speakers[:limit_speakers]
        self.spk2id = {s: i for i, s in enumerate(speakers)}

        self.items = []
        for spk in speakers:
            for fname in os.listdir(wav_root / spk):
                if fname.endswith("_mic1.flac"):
                    wav_path = wav_root / spk / fname
                    txt_path = txt_root / spk / fname.replace("_mic1.flac", ".txt")
                    if txt_path.exists():
                        self.items.append((wav_path, txt_path, self.spk2id[spk]))

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, txt_path, spk_id = self.items[idx]
        # ---- text
        text_str = Path(txt_path).read_text().strip()
        text_ids = torch.LongTensor(text_to_sequence(text_str, ['english_cleaners']))
        # ---- audio
        wav, sr = librosa.load(str(wav_path), sr=None)
        if sr != SR: 
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        # ---- mel (cache)
        cache_path = wav_path.with_suffix(".npy").as_posix().replace("wav48_silence_trimmed", "mel_cache")
        if self.cache_mel and Path(cache_path).exists():
            mel = torch.from_numpy(np.load(cache_path)).float()
        else:
            mel = wav_to_mel(wav)
            if self.cache_mel:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, mel.numpy())
        return text_ids, mel, torch.LongTensor([spk_id])

    @staticmethod
    def collate_cvae(batch):
        """
        Returns:
        text_pad        : [B, max_text]
        text_len        : [B]
        mel_pad         : [B, 80, max_mel]  (match n_frames_per_step)
        mel_len         : [B]
        gate_pad        : [B, max_mel]      (0/1 stop token)
        speaker_ids     : [B]
        """
        texts, mels, spk_ids = zip(*batch)
        text_len = torch.LongTensor([t.size(0) for t in texts])
        mel_len  = torch.LongTensor([m.shape[1]     for m in mels])

        text_pad = pad_1D(texts, pad_value=0)

        # ------ 让 mel 长度对齐 n_frames_per_step ------
        max_mel = max(mel_len)
        r = N_FRAMES_PER_STEP
        if max_mel % r != 0: max_mel += r - max_mel % r
        mel_pad  = pad_2D(mels, pad_value=0.0)      # 先 pad 到统一最长
        if mel_pad.shape[2] < max_mel:              # 二次 pad 到步长倍数
            pad_extra = max_mel - mel_pad.shape[2]
            mel_pad = torch.nn.functional.pad(mel_pad, (0, pad_extra), value=0.0)

        # ------ stop gate (0 until last frame, 1 afterwards) ------
        gate_pad = torch.zeros(len(batch), max_mel)
        for i, l in enumerate(mel_len):
            gate_pad[i, l-1:] = 1.0

        speaker_ids = torch.cat(spk_ids, dim=0)

        return text_pad, text_len, mel_pad, mel_len, gate_pad, speaker_ids

