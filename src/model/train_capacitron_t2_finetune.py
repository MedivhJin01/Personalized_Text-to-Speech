import os
import sys
import numpy as np

# Add the current directory to the path so we can import our custom formatter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vctk_formatter import vctk_formatter

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CapacitronVAEConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.speakers import SpeakerManager

# Paths
output_path = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (three levels up from src/model/)
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
data_path = os.path.join(project_root, "dataset/VCTK")
pretrained_model_path = "/Users/yimingyao/Library/Application Support/tts/tts_models--en--blizzard2013--capacitron-t2-c50/model_file.pth"
speaker_emb_path = os.path.join(
    project_root, "src/speaker_embedding/speaker_emb_lookup.npy"
)


# Load speaker embeddings
speaker_embeddings = np.load(speaker_emb_path, allow_pickle=True).item()
selected_speakers = list(speaker_embeddings.keys())
num_speakers = len(selected_speakers)

print(f"Fine-tuning with {num_speakers} speakers: {selected_speakers}")

# Custom VCTK dataset configuration
dataset_config = BaseDatasetConfig(
    formatter="vctk",  # This will be mapped to our custom formatter
    meta_file_train=None,  # VCTK doesn't use metadata.csv
    path=data_path,
    ignored_speakers=None,
)

# Audio config matching the pre-trained model
audio_config = BaseAudioConfig(
    sample_rate=24000,  # Match pre-trained model
    do_trim_silence=True,
    trim_db=60.0,
    signal_norm=True,  # Match pre-trained model
    mel_fmin=80.0,  # Match pre-trained model
    mel_fmax=12000,  # Match pre-trained model
    spec_gain=25.0,  # Match pre-trained model
    log_func="np.log10",  # Match pre-trained model
    ref_level_db=20,
    preemphasis=0.0,
    num_mels=80,
    fft_size=1024,
    win_length=1024,
    hop_length=256,
    power=1.5,  # Match pre-trained model
    min_level_db=-100,  # Match pre-trained model
    symmetric_norm=True,  # Match pre-trained model
    max_norm=4.0,  # Match pre-trained model
    clip_norm=True,  # Match pre-trained model
)

# Capacitron config matching pre-trained model
capacitron_config = CapacitronVAEConfig(
    capacitron_VAE_loss_alpha=1.0,
    capacitron_capacity=50,  # Match pre-trained model
    capacitron_VAE_embedding_dim=128,
    capacitron_text_summary_embedding_dim=128,
    capacitron_use_text_summary_embeddings=True,
    capacitron_use_speaker_embedding=False,  # Disable for fine-tuning - use Tacotron2's speaker embedding instead
    capacitron_grad_clip=1.0,  # Reduced from 5.0 for better stability during fine-tuning
)

config = Tacotron2Config(
    run_name="Capacitron-T2-VCTK-Finetune",
    audio=audio_config,
    capacitron_vae=capacitron_config,
    use_capacitron_vae=True,
    # Multi-speaker setup
    num_speakers=num_speakers,
    use_speaker_embedding=True,
    speaker_embedding_dim=256,
    # Fine-tuning parameters - smaller batch size and lower learning rate
    batch_size=4,  # Smaller for fine-tuning
    max_audio_len=3 * 24000,  # Adjust for 24kHz
    min_audio_len=0.5 * 24000,  # Adjust for 24kHz
    eval_batch_size=4,
    num_loader_workers=0,
    num_eval_loader_workers=0,
    precompute_num_workers=8,
    run_eval=True,
    test_delay_epochs=10,  # Start evaluation earlier
    ga_alpha=0.0,
    r=2,
    # Fine-tuning optimizer settings
    optimizer="CapacitronOptimizer",
    optimizer_params={
        "RAdam": {"betas": [0.9, 0.998], "weight_decay": 1e-6},
        "SGD": {"lr": 1e-6, "momentum": 0.9},  # Lower learning rate for fine-tuning
    },
    attention_type="dynamic_convolution",
    grad_clip=1.0,  # Enable to see gradient norms (works alongside capacitron_grad_clip)
    double_decoder_consistency=False,
    epochs=50,  # Reduced from 200 for initial testing
    # Text processing - match pre-trained model
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phonemizer="espeak",
    phoneme_cache_path=os.path.join(data_path, "phoneme_cache"),
    stopnet_pos_weight=15,
    print_step=5,
    print_eval=True,
    mixed_precision=False,  # Disable mixed precision for stability during fine-tuning
    seq_len_norm=True,
    output_path=output_path,
    datasets=[dataset_config],
    # Fine-tuning learning rate schedule - much lower rates for stability
    lr=1e-4,  # Much lower learning rate to prevent NaN values
    # lr_scheduler="StepwiseGradualLR",
    # lr_scheduler_params={
    #     "gradual_learning_rates": [
    #         [0, 5e-5],  # Start very low for fine-tuning
    #         [2000, 2e-5],  # Reduce gradually
    #         [5000, 1e-5],  # Even lower
    #         [8000, 5e-6],  # Very conservative
    #     ]
    # },
    scheduler_after_epoch=False,  # scheduler doesn't work without this flag
    # Loss configuration
    loss_masking=False,
    decoder_loss_alpha=1.0,
    postnet_loss_alpha=1.0,
    postnet_diff_spec_alpha=0.0,
    decoder_diff_spec_alpha=0.0,
    decoder_ssim_alpha=0.0,
    postnet_ssim_alpha=0.0,
)

ap = AudioProcessor(**config.audio.to_dict())

tokenizer, config = TTSTokenizer.init_from_config(config)

# Load samples using our custom formatter
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    formatter=vctk_formatter,  # Use our custom formatter
)

print(
    f"Loaded {len(train_samples)} training samples and {len(eval_samples)} eval samples"
)

# Create SpeakerManager with training samples and d-vectors file
model = Tacotron2(
    config,
    ap,
    tokenizer,
    speaker_manager=SpeakerManager(
        data_items=train_samples  # Pass the training samples to extract speaker info
    ),
)

# Create trainer args with weights_only=False to handle older checkpoint format
trainer_args = TrainerArgs(
    restore_path=pretrained_model_path,  # Load pre-trained model
    skip_train_epoch=False,  # Continue training
)

# Monkey patch the trainer to handle older checkpoint format
import torch

original_load_fsspec = torch.load


def safe_load_fsspec(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return original_load_fsspec(*args, **kwargs)


# Temporarily replace torch.load
torch.load = safe_load_fsspec

try:
    trainer = Trainer(
        trainer_args,
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        training_assets={"audio_processor": ap},
    )
finally:
    # Restore original torch.load
    torch.load = original_load_fsspec

trainer.fit()
