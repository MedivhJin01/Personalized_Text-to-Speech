import os
from pathlib import Path
from typing import List, Tuple
from glob import glob


def vctk_formatter(root_path: str, meta_file_train: str = None, **kwargs) -> List[dict]:
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
        Path(root_path).parent
        / "src/speaker_embedding/selected_speakers.txt",  # From dataset/VCTK
        Path(root_path).parent.parent
        / "src/speaker_embedding/selected_speakers.txt",  # From dataset
        Path(
            "src/speaker_embedding/selected_speakers.txt"
        ),  # Relative to current working directory
    ]

    selected_speakers_path = None
    for path in possible_paths:
        if path.exists():
            selected_speakers_path = path
            break

    if selected_speakers_path and selected_speakers_path.exists():
        with open(selected_speakers_path) as f:
            selected_speakers = [line.strip() for line in f if line.strip()]
        print(f"Using selected speakers: {selected_speakers}")
    else:
        # Get all available speakers if no selection file
        selected_speakers = [d for d in os.listdir(wav_root) if (wav_root / d).is_dir()]
        selected_speakers.sort()
        print(f"Using all available speakers: {selected_speakers}")

    samples = []
    file_ext = "flac"

    # Get all text files using glob pattern like the example
    meta_files = glob(f"{os.path.join(root_path, 'txt')}/**/*.txt", recursive=True)

    for meta_file in meta_files:
        # Extract speaker_id and file_id from path
        _, speaker_id, txt_file = os.path.relpath(meta_file, root_path).split(os.sep)
        file_id = txt_file.split(".")[0]

        # Only process selected speakers
        if speaker_id not in selected_speakers:
            continue

        # Read the text file
        try:
            with open(meta_file, "r", encoding="utf-8") as file_text:
                text = file_text.readlines()[0].strip()
        except (IOError, IndexError) as e:
            print(f"Warning: Could not read text file {meta_file}: {e}")
            continue

        # Handle audio file path - use mic1 by default
        # p280 has no mic2 recordings, so always use mic1 for p280
        if speaker_id == "p280":
            wav_file = os.path.join(
                root_path,
                "wav48_silence_trimmed",
                speaker_id,
                file_id + f"_mic1.{file_ext}",
            )
        else:
            wav_file = os.path.join(
                root_path,
                "wav48_silence_trimmed",
                speaker_id,
                file_id + f"_mic1.{file_ext}",
            )

        # Check if audio file exists
        if os.path.exists(wav_file):
            # Create sample in format expected by TTS
            samples.append(
                {
                    "text": text,
                    "audio_file": wav_file,
                    "root_path": str(root_path),
                    "speaker_name": speaker_id,  # Use just speaker_id instead of "VCTK_" + speaker_id
                }
            )
        else:
            print(f" [!] wav files don't exist - {wav_file}")

    print(f"Loaded {len(samples)} samples from {len(selected_speakers)} speakers")
    return samples
