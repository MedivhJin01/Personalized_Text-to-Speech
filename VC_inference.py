import os
import numpy as np
import torch
from torch import no_grad, LongTensor
import argparse
import commons
from mel_processing import spectrogram_torch
import utils
from models import SynthesizerTrn
import gradio as gr
import librosa
import webbrowser

from text import text_to_sequence
device = "cuda:0" if torch.cuda.is_available() else "cpu"
import logging
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

# --- All language selection code has been removed ---

def get_text(text, hps):
    # This function is now hardcoded for English text processing
    text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def create_tts_fn(model, hps, speaker_ids):
    # The 'language' argument has been removed from this function
    def tts_fn(text, speaker, speed):
        # Text is now always processed as English
        text = "[EN]" + text + "[EN]"
        speaker_id = speaker_ids[speaker]
        stn_tst = get_text(text, hps)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return "Success", (hps.data.sampling_rate, audio)
    return tts_fn

# --- VOICE CONVERSION FUNCTION HAS BEEN REMOVED ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="directory to your fine-tuned model")
    parser.add_argument("--config_dir", help="directory to your model config file")
    parser.add_argument("--share", action="store_true", help="make link public")

    args = parser.parse_args()
    hps = utils.get_hparams_from_file(args.config_dir)

    net_g = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.model_dir, net_g, None)
    speaker_ids = hps.speakers
    speakers = list(speaker_ids.keys())
    
    tts_fn = create_tts_fn(net_g, hps, speaker_ids)
    
    # --- The Gradio app is now simplified for English TTS only ---
    app = gr.Blocks()
    with app:
        gr.Markdown("# English Text-to-Speech")
        gr.Markdown("Enter English text, select a voice, and click Generate.")
        with gr.Row():
            with gr.Column():
                textbox = gr.TextArea(label="English Text",
                                      placeholder="Type your sentence here",
                                      value="Hello, this is a test of my custom voice.", elem_id="tts-input")
                char_dropdown = gr.Dropdown(choices=speakers, value=speakers[0], label='Character Voice')
                duration_slider = gr.Slider(minimum=0.1, maximum=5, value=1, step=0.1,
                                            label='Speed')
            with gr.Column():
                text_output = gr.Textbox(label="Message")
                audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                btn = gr.Button("Generate!", variant="primary")
                # Note: The 'language_dropdown' input has been removed here
                btn.click(tts_fn,
                          inputs=[textbox, char_dropdown, duration_slider],
                          outputs=[text_output, audio_output])

    # Open in browser and launch the app
    if not args.share:
        webbrowser.open("http://127.0.0.1:7860")
    app.launch(share=args.share)


