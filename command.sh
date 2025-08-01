python src/model/train_glow_tts_finetune.py --restore_path "C:\Users\jx\AppData\Local\tts\tts_models--en--ljspeech--glow-tts/model_file.pth" --speakers VCTK_p283 VCTK_p361 VCTK_p363

tts --text "test number one, hello world" --model_path src/outputs/vctk_multispeaker_finetune-July-31-2025_08+12PM-9b52b67/best_model.pth --config_path src/outputs/vctk_multispeaker_finetune-July-31-2025_08+12PM-9b52b67/config.json --speaker_idx VCTK_p361
