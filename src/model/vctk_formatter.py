import os
from pathlib import Path
from typing import List, Tuple


def vctk_formatter(
    root_path: str, meta_file_train: str = None, **kwargs
) -> List[dict]:
    """
    VCTK dataset formatter for TTS training.

    Args:
        root_path (str): Path to VCTK dataset
        meta_file_train (str): Not used for VCTK, kept for compatibility

    Returns:
        List[dict]: List of dictionaries with required TTS keys
    """
    root_path = Path(root_path)
    wav_root = root_path / "wav48_silence_trimmed"
    txt_root = root_path / "txt"

    # Load selected speakers - use absolute path
    # Try different possible paths for the selected speakers file
    possible_paths = [
        Path(root_path).parent / "src/speaker_embedding/selected_speakers.txt",  # From dataset/VCTK
        Path(root_path).parent.parent / "src/speaker_embedding/selected_speakers.txt",  # From dataset
        Path("src/speaker_embedding/selected_speakers.txt"),  # Relative to current working directory
    ]
    
    selected_speakers_path = None
    for path in possible_paths:
        if path.exists():
            selected_speakers_path = path
            break
    if selected_speakers_path.exists():
        with open(selected_speakers_path) as f:
            selected_speakers = [line.strip() for line in f if line.strip()]
        print(f"Using selected speakers: {selected_speakers}")
    else:
        # Get all available speakers if no selection file
        selected_speakers = [d for d in os.listdir(wav_root) if (wav_root / d).is_dir()]
        selected_speakers.sort()
        print(f"Using all available speakers: {selected_speakers}")

    samples = []

    for spk in selected_speakers:
        spk_wav_dir = wav_root / spk
        spk_txt_dir = txt_root / spk

        if not spk_wav_dir.exists() or not spk_txt_dir.exists():
            print(f"Warning: Missing directories for speaker {spk}")
            continue

        # Get all audio files for this speaker
        for wav_file in os.listdir(spk_wav_dir):
            if wav_file.endswith("_mic1.flac"):
                wav_path = spk_wav_dir / wav_file
                txt_file = wav_file.replace("_mic1.flac", ".txt")
                txt_path = spk_txt_dir / txt_file

                if txt_path.exists():
                    # Read the text
                    text = txt_path.read_text().strip()

                    # Create sample in format expected by TTS
                    samples.append({
                        "text": text,
                        "audio_file": str(wav_path),
                        "root_path": str(root_path),
                        "speaker_name": spk
                    })

    print(f"Loaded {len(samples)} samples from {len(selected_speakers)} speakers")
    return samples
