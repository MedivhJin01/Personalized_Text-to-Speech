import os
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from resemblyzer import VoiceEncoder, preprocess_wav
import soundfile as sf

# Path configs
VCTK_ROOT = Path("dataset/VCTK")
WAV_DIR = VCTK_ROOT / "wav48_silence_trimmed"
SPEAKER_INFO = VCTK_ROOT / "speaker-info.txt"
EMBEDDING_OUT = Path("src/speaker_embedding/speaker_emb_lookup.npy")

# Parameters
NUM_SPEAKERS = 10  # can be 8-12
MIN_TOTAL_SECONDS = 60  # minimum total audio per speaker (in seconds)


# 1. Parse speaker-info.txt
def parse_speaker_info(path):
    speakers = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            spk_id = parts[0]
            age = parts[1]
            gender = parts[2]
            accent = parts[3]
            region = " ".join(parts[4:]) if len(parts) > 4 else ""
            speakers[spk_id] = {
                "age": age,
                "gender": gender,
                "accent": accent,
                "region": region,
            }
    return speakers


# 2. Get all speakers with available audio
def get_available_speakers(wav_dir):
    return [d for d in os.listdir(wav_dir) if (wav_dir / d).is_dir()]


# 3. For each speaker, get all *_mic1.flac files and total duration
def get_speaker_utterances_and_duration(wav_dir, spk_id):
    spk_dir = wav_dir / spk_id
    files = [spk_dir / f for f in os.listdir(spk_dir) if f.endswith("_mic1.flac")]
    total_sec = 0
    for f in files:
        try:
            info = sf.info(str(f))
            total_sec += info.duration
        except Exception:
            continue
    return files, total_sec


# 4. Sample diverse speakers (by gender/accent if possible)
def sample_speakers(speaker_info, available_speakers, num=NUM_SPEAKERS):
    # Group by gender and accent for diversity
    by_gender = defaultdict(list)
    for spk in available_speakers:
        gender = speaker_info.get(spk, {}).get("gender", "U")
        by_gender[gender].append(spk)
    # Try to sample evenly from each gender
    selected = []
    genders = list(by_gender.keys())
    per_gender = max(1, num // len(genders))
    for gender in genders:
        random.shuffle(by_gender[gender])
        selected.extend(by_gender[gender][:per_gender])
    # Fill up if not enough
    if len(selected) < num:
        remaining = [spk for spk in available_speakers if spk not in selected]
        random.shuffle(remaining)
        selected.extend(remaining[: num - len(selected)])
    return selected[:num]


# 5. Main pipeline
def main():
    speaker_info = parse_speaker_info(SPEAKER_INFO)
    available_speakers = get_available_speakers(WAV_DIR)
    print(f"Found {len(available_speakers)} speakers with audio.")

    # Filter speakers with enough data
    eligible = []
    for spk in available_speakers:
        files, total_sec = get_speaker_utterances_and_duration(WAV_DIR, spk)
        if total_sec >= MIN_TOTAL_SECONDS:
            eligible.append(spk)
    print(f"{len(eligible)} speakers have >= {MIN_TOTAL_SECONDS}s of audio.")

    # Sample diverse speakers
    selected_speakers = sample_speakers(speaker_info, eligible, NUM_SPEAKERS)
    print(f"Selected speakers: {selected_speakers}")

    encoder = VoiceEncoder()
    speaker_embeddings = {}

    for spk in selected_speakers:
        files, _ = get_speaker_utterances_and_duration(WAV_DIR, spk)
        utter_embeds = []
        for f in files:
            try:
                wav = preprocess_wav(str(f))
                embed = encoder.embed_utterance(wav)
                utter_embeds.append(embed)
            except Exception as e:
                print(f"Error processing {f}: {e}")
        if utter_embeds:
            speaker_embeddings[spk] = np.mean(utter_embeds, axis=0)
            print(f"Speaker {spk}: {len(utter_embeds)} utterances embedded.")
        else:
            print(f"Speaker {spk}: No valid utterances.")

    # Save embeddings
    np.save(EMBEDDING_OUT, speaker_embeddings)
    print(f"Saved speaker embeddings to {EMBEDDING_OUT}")


if __name__ == "__main__":
    main()
