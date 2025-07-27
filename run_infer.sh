#!/bin/bash

python inference.py \
  --model_ckpt src/model/check_points/cvae_tacotron2_trial1.pth \
  --tacotron_ckpt src/model/check_points/tacotron2_pretrained.pt \
  --text "Hello world" \
  --speaker_id p323 \
  --out_wav demo_result/demo.wav