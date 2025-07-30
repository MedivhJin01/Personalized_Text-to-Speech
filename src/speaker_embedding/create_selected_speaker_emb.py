import os
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging
import warnings

# Try to import resemblyzer, but provide fallback if not available
try:
    from resemblyzer import VoiceEncoder, preprocess_wav

    RESEMBLYZER_AVAILABLE = True
except ImportError:
    print("Warning: resemblyzer not available. Using random embeddings as fallback.")
    RESEMBLYZER_AVAILABLE = False

try:
    import soundfile as sf

    SOUNDFILE_AVAILABLE = True
except ImportError:
    print("Warning: soundfile not available. Using basic file detection.")
    SOUNDFILE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Path configs
VCTK_ROOT = Path("dataset/VCTK")
WAV_DIR = VCTK_ROOT / "wav48_silence_trimmed"
SPEAKER_INFO = VCTK_ROOT / "speaker-info.txt"
EMBEDDING_OUT = Path("src/speaker_embedding/speaker_emb_lookup.npy")

# Parameters
NUM_SPEAKERS = 10  # can be 8-12
MIN_TOTAL_SECONDS = 30  # reduced minimum for more flexibility
EMBEDDING_DIM = 512  # standard embedding dimension


# 1. Parse speaker-info.txt with better error handling
def parse_speaker_info(path):
    speakers = {}
    if not os.path.exists(path):
        logger.warning(f"Speaker info file not found: {path}")
        return speakers

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
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
        logger.info(f"Loaded speaker info for {len(speakers)} speakers")
    except Exception as e:
        logger.error(f"Error parsing speaker info: {e}")

    return speakers


# 2. Get all speakers with available audio
def get_available_speakers(wav_dir):
    if not os.path.exists(wav_dir):
        logger.error(f"WAV directory not found: {wav_dir}")
        return []

    speakers = [d for d in os.listdir(wav_dir) if (wav_dir / d).is_dir()]
    speakers.sort()  # Sort for consistent ordering
    logger.info(f"Found {len(speakers)} speakers with audio directories")
    return speakers


# 3. For each speaker, get all audio files and total duration
def get_speaker_utterances_and_duration(wav_dir, spk_id):
    spk_dir = wav_dir / spk_id
    if not os.path.exists(spk_dir):
        return [], 0

    # Support multiple audio formats
    audio_extensions = [".flac", ".wav", ".mp3"]
    files = []

    for ext in audio_extensions:
        # Try different naming patterns
        patterns = [f"*_mic1{ext}", f"*{ext}", f"*_mic2{ext}"]
        for pattern in patterns:
            found_files = list(spk_dir.glob(pattern))
            files.extend(found_files)

    # Remove duplicates
    files = list(set(files))
    files.sort()

    total_sec = 0
    valid_files = []

    for f in files:
        try:
            if SOUNDFILE_AVAILABLE:
                info = sf.info(str(f))
                duration = info.duration
            else:
                # Basic file size estimation (rough approximation)
                file_size = os.path.getsize(f)
                duration = file_size / (16000 * 2)  # Assume 16kHz, 16-bit audio

            if duration > 0:
                total_sec += duration
                valid_files.append(f)
        except Exception as e:
            logger.debug(f"Error processing {f}: {e}")
            continue

    return valid_files, total_sec


# 4. Sample diverse speakers with better fallback
def sample_speakers(speaker_info, available_speakers, num=NUM_SPEAKERS):
    if len(available_speakers) <= num:
        logger.info(f"Using all {len(available_speakers)} available speakers")
        return available_speakers[:num]

    # Try to sample by gender if speaker info is available
    if speaker_info:
        by_gender = defaultdict(list)
        for spk in available_speakers:
            gender = speaker_info.get(spk, {}).get("gender", "U")
            by_gender[gender].append(spk)

        # Sample evenly from each gender
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

        logger.info(f"Selected {len(selected)} speakers with gender diversity")
        return selected[:num]
    else:
        # Fallback: random selection
        selected = random.sample(available_speakers, num)
        logger.info(f"Selected {len(selected)} speakers randomly")
        return selected


# 5. Create embeddings with fallback
def create_speaker_embedding(spk, files, encoder=None):
    """Create speaker embedding from audio files"""

    if RESEMBLYZER_AVAILABLE and encoder is not None:
        # Use resemblyzer for real embeddings
        utter_embeds = []
        for f in files[:10]:  # Limit to first 10 files for speed
            try:
                wav = preprocess_wav(str(f))
                embed = encoder.embed_utterance(wav)
                utter_embeds.append(embed)
            except Exception as e:
                logger.debug(f"Error processing {f}: {e}")
                continue

        if utter_embeds:
            return np.mean(utter_embeds, axis=0)

    # Fallback: create random embedding
    logger.warning(f"Using random embedding for speaker {spk}")
    return np.random.randn(EMBEDDING_DIM).astype(np.float32)


# 6. Create TTS-compatible embedding format
def create_tts_embedding_format(selected_speakers, wav_dir, encoder=None):
    """
    Create embeddings in the format expected by TTS library.
    Each audio file path maps to its corresponding speaker embedding.
    """
    tts_embeddings = {}

    for i, spk in enumerate(selected_speakers, 1):
        logger.info(f"Processing speaker {spk} ({i}/{len(selected_speakers)})")

        files, total_sec = get_speaker_utterances_and_duration(wav_dir, spk)
        logger.info(f"  - {len(files)} files, {total_sec:.1f}s total")

        if not files:
            logger.warning(f"No valid files found for speaker {spk}")
            continue

        # Create speaker embedding
        speaker_embedding = create_speaker_embedding(spk, files, encoder)

        # Create TTS-compatible keys for each audio file
        for audio_file in files:
            # Create the key format expected by TTS: '#wav48_silence_trimmed/speaker_id/filename'
            # Get relative path from wav_dir
            rel_path = audio_file.relative_to(wav_dir)
            tts_key = f"#{rel_path}"

            # Store the speaker embedding for this audio file
            tts_embeddings[tts_key] = {
                "embedding": speaker_embedding,
                "speaker_id": spk,
            }

        logger.info(f"  - Created {len(files)} TTS embedding entries for speaker {spk}")

    return tts_embeddings


# 7. Main pipeline with better error handling
def main():
    logger.info("Starting speaker embedding creation...")

    # Check if output directory exists
    EMBEDDING_OUT.parent.mkdir(parents=True, exist_ok=True)

    # Parse speaker info
    speaker_info = parse_speaker_info(SPEAKER_INFO)

    # Get available speakers
    available_speakers = get_available_speakers(WAV_DIR)
    if not available_speakers:
        logger.error("No speakers found. Please check the dataset path.")
        return

    # Filter speakers with enough data
    eligible = []
    speaker_stats = {}

    for spk in available_speakers:
        files, total_sec = get_speaker_utterances_and_duration(WAV_DIR, spk)
        speaker_stats[spk] = {"files": len(files), "duration": total_sec}

        if total_sec >= MIN_TOTAL_SECONDS:
            eligible.append(spk)

    logger.info(f"{len(eligible)} speakers have >= {MIN_TOTAL_SECONDS}s of audio.")

    if not eligible:
        logger.warning(
            f"No speakers meet the minimum duration requirement. Lowering threshold..."
        )
        # Find speakers with the most data
        eligible = sorted(
            available_speakers, key=lambda x: speaker_stats[x]["duration"], reverse=True
        )[:NUM_SPEAKERS]

    # Sample diverse speakers
    selected_speakers = sample_speakers(speaker_info, eligible, NUM_SPEAKERS)
    logger.info(f"Selected speakers: {selected_speakers}")

    # Save selected speakers list for reference
    selected_speakers_path = EMBEDDING_OUT.parent / "selected_speakers.txt"
    with open(selected_speakers_path, "w") as f:
        for spk in selected_speakers:
            f.write(f"{spk}\n")
    logger.info(f"Saved selected speakers list to {selected_speakers_path}")

    # Initialize encoder if available
    encoder = None
    if RESEMBLYZER_AVAILABLE:
        try:
            encoder = VoiceEncoder()
            logger.info("Using resemblyzer for speaker embeddings")
        except Exception as e:
            logger.warning(f"Failed to initialize resemblyzer: {e}")

    # Create TTS-compatible embeddings
    tts_embeddings = create_tts_embedding_format(selected_speakers, WAV_DIR, encoder)

    # Save embeddings with error handling
    try:
        np.save(EMBEDDING_OUT, tts_embeddings)
        logger.info(
            f"Saved {len(tts_embeddings)} TTS-compatible speaker embeddings to {EMBEDDING_OUT}"
        )

        # Print summary
        print("\n" + "=" * 50)
        print("SPEAKER EMBEDDING SUMMARY")
        print("=" * 50)
        print(f"Total embedding entries: {len(tts_embeddings)}")
        print(f"Selected speakers: {len(selected_speakers)}")
        for spk in selected_speakers:
            stats = speaker_stats.get(spk, {})
            # Count embeddings for this speaker
            spk_embeddings = sum(1 for key in tts_embeddings.keys() if spk in key)
            print(
                f"{spk}: {stats.get('files', 0)} files, {stats.get('duration', 0):.1f}s, {spk_embeddings} embeddings"
            )
        print("=" * 50)

    except Exception as e:
        logger.error(f"Error saving embeddings: {e}")
        # Try alternative save method
        try:
            import pickle

            with open(EMBEDDING_OUT.with_suffix(".pkl"), "wb") as f:
                pickle.dump(tts_embeddings, f)
            logger.info(f"Saved embeddings using pickle as fallback")
        except Exception as e2:
            logger.error(f"Failed to save embeddings: {e2}")


if __name__ == "__main__":
    main()
