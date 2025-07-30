import os
import sys
import numpy as np

# Add the current directory to the path so we can import our custom formatter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vctk_formatter import vctk_formatter

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.utils.manage import ModelManager

# Paths
output_path = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (three levels up from src/model/)
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
data_path = os.path.join(project_root, "dataset/VCTK")
speaker_emb_path = os.path.join(
    project_root, "src/speaker_embedding/speaker_emb_lookup.npy"
)

# Download and get path to pretrained Glow-TTS model
model_manager = ModelManager()
pretrained_model_path, _, _ = model_manager.download_model(
    "tts_models/en/ljspeech/glow-tts"
)
print(f"Using pretrained model from: {pretrained_model_path}")


# Load speaker embeddings to get selected speakers
speaker_embeddings = np.load(speaker_emb_path, allow_pickle=True).item()
selected_speakers = list(speaker_embeddings.keys())
num_speakers = len(selected_speakers)

print(f"Fine-tuning Glow-TTS with {num_speakers} speakers: {selected_speakers}")

# Custom VCTK dataset configuration
dataset_config = BaseDatasetConfig(
    formatter="vctk",  # This will be mapped to our custom formatter
    meta_file_train=None,  # VCTK doesn't use metadata.csv
    path=data_path,
    ignored_speakers=None,
)

# Audio config for Glow-TTS
# ❗ resample the dataset externally using `TTS/bin/resample.py` and set `resample=False` for faster training
audio_config = BaseAudioConfig(
    sample_rate=22050,
    resample=False,  # Set to False since we already have processed audio
    do_trim_silence=True,
    trim_db=23.0,
    # Glow-TTS specific audio settings
    fft_size=1024,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0.0,
    mel_fmax=11025,
    ref_level_db=20,
    preemphasis=0.0,
)

# Define Glow-TTS model config
config = GlowTTSConfig(
    batch_size=8,
    eval_batch_size=4,
    num_loader_workers=0,
    num_eval_loader_workers=0,
    precompute_num_workers=0,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=100,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=5,
    print_eval=False,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    use_speaker_embedding=True,
    min_text_len=0,
    max_text_len=500,
    min_audio_len=0,
    max_audio_len=500000,
    lr=1e-4,
    # Use StepwiseGradualLR instead of NoamLR for fine-tuning
    lr_scheduler="StepwiseGradualLR",
    lr_scheduler_params={
        "gradual_learning_rates": [
            [0, 1e-4],  # Start with 1e-4
            [1000, 5e-5],  # Reduce to 5e-5 after 1000 steps
            [3000, 1e-5],  # Reduce to 1e-5 after 3000 steps
            [5000, 5e-6],  # Reduce to 5e-6 after 5000 steps
        ]
    },
    scheduler_after_epoch=False,  # scheduler works per step, not per epoch
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# If characters are not defined in the config, default characters are passed to the config
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    formatter=vctk_formatter,
)

# Filter samples to only include selected speakers
train_samples = [s for s in train_samples if s["speaker_name"] in selected_speakers]
eval_samples = [s for s in eval_samples if s["speaker_name"] in selected_speakers]

print(f"Training samples: {len(train_samples)}")
print(f"Evaluation samples: {len(eval_samples)}")

# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(
    train_samples + eval_samples, parse_key="speaker_name"
)
config.num_speakers = speaker_manager.num_speakers

print(f"Number of speakers in speaker manager: {speaker_manager.num_speakers}")
print(f"Speaker names: {speaker_manager.name_to_id}")

# init model
model = GlowTTS(config, ap, tokenizer, speaker_manager=speaker_manager)

# INITIALIZE THE TRAINER
# Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# distributed training, etc.
trainer_args = TrainerArgs(
    restore_path=pretrained_model_path,  # Load pre-trained Glow-TTS model
    skip_train_epoch=False,  # Continue training from pretrained model
)

trainer = Trainer(
    trainer_args,
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

# AND... 3,2,1... 🚀
if __name__ == "__main__":
    trainer.fit()
