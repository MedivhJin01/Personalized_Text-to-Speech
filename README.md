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
* Run `python finetune_speaker_v2.py -m ./OUTPUT_MODEL --max_epochs "{Epochs}" --drop_speaker_embed True`  
   Do replace `{Epochs}` with your desired number of epochs(100+ recommend). 
   You can also continue training on previous stored checkpoint, the model checkpoints for those generator and discriminator should under `./OUTPUT_MODEL/` dir.
   To view training progress, open a new terminal and `cd` to the project root directory, run `tensorboard --logdir=./OUTPUT_MODEL`, then visit `localhost:6006` with your web browser.

## Inference