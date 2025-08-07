# VITS Model Fine-Tuning
## Build environment
* Clong the repo, require `Python==3.8`, CMake & C/C++ compilers, ffmpeg, CUDA 11.6 

* `pip install -r requirements.txt`

* Build Monotonic Alignment for training
   ```
    cd monotonic_align
    mkdir monotonic_align
    python setup.py build_ext --inplace
    ```

* Create necessary dirsctories for fine-tuning
   ```
    mkdir pretrained_models
    mkdir custom_character_voice
   ```

## Pretrained Download
* The Trilingual (Chinese, Japanese, English) Pretrained Model.
https://drive.google.com/drive/folders/1jI1bCpVZ1S_LNlRIdY9rPM5grS2Nl08F?usp=drive_link

Manually download from the google drive provided above. Required files are `G_0.pth`, `D_0.pth`, `finetune_speaker.json`.

Put the models under pretrained_models dir, put finetune_speaker.json under configs dir.

## Data Preparation
* Prepare your dataset, the short audios packed by a single file, whose file structure should be as shown below:
```
Your-audio-file
├───Character_name_1
├   ├───c1_1.wav
├   ├───...
├   ├───...
├   └───c1_n.wav
├───Character_name_2
├   ├───c2_1.wav
├   ├───...
├   ├───...
├   └───c2_n.wav
├───...
├
└───Character_name_n
    ├───cn_1.wav
    ├───...
    ├───...
    └───cn_n.wav
```  
It is better to have aduio quality >=2s, <=10s

* Process the audio data(GPU memory of 12GP, otherwise scale down `whisper_size` to `medium` or `small`)
```
   python scripts/denoise_audio.py
   python scripts/short_audio_transcribe.py --languages "{CJE}" --whisper_size large
   python scripts/resample.py
   ```

## Training
* Run `python finetune_speaker_v2.py -m ./OUTPUT_MODEL --max_epochs "{Epochs}" --drop_speaker_embed True` Do replace `{Epochs}` with your desired number of epochs(100+ recommend). 
* You can also continue training on previous stored checkpoint, the model checkpoints for those generator and discriminator should under `./OUTPUT_MODEL/` dir.
* To view training progress, open a new terminal and `cd` to the project root directory, run `tensorboard --logdir=./OUTPUT_MODEL`, then visit `localhost:6006` with your web browser.

## Inference
* Simply run `python VC_inference.py --model_dir ./OUTPUT_MODEL/G_latest.pth --share True`, which automatically create TTS Gradio Interface link contain all the customized characters voice option.

## Fine-tune dataset
* The dataset we use extract 10 speakers from the VCTK dataset, each speaker contains 50 audio files.

* Dataset: https://drive.google.com/drive/folders/1jI1bCpVZ1S_LNlRIdY9rPM5grS2Nl08F?usp=drive_link

## Play around with our fine-tuned model
VCTK 10 Speakers TTS: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)]([https://huggingface.co/spaces/Russ0sg/my_english_tts_app])

## Presentation slide
Presentation slide can be accessed through Canva [https://www.canva.com/design/DAGu86xQ3fs/aKR0OxxhOhZgFX9uAmEh9w/edit?utm_content=DAGu86xQ3fs&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton ](https://www.canva.com/design/DAGu86xQ3fs/aKR0OxxhOhZgFX9uAmEh9w/edit?utm_content=DAGu86xQ3fs&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) 
