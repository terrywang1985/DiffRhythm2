import os
import sys

# Add eSpeak NG to PATH for phonemizer
espeak_path = r"C:\Program Files\eSpeak NG"
if os.path.exists(espeak_path):
    os.environ['PATH'] = espeak_path + os.pathsep + os.environ.get('PATH', '')
    os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = os.path.join(espeak_path, 'libespeak-ng.dll')

import torch
import torchaudio
import json
import re
import random
import gradio as gr
import pedalboard
import numpy as np
from tqdm import tqdm

from muq import MuQMuLan
from diffrhythm2.cfm import CFM
from diffrhythm2.backbones.dit import DiT
from bigvgan.model import Generator
from huggingface_hub import hf_hub_download

# Constants and Tokenizer (copied from inference.py)
STRUCT_INFO = {
    "[start]": 500,
    "[end]": 501,
    "[intro]": 502,
    "[verse]": 503,
    "[chorus]": 504,
    "[outro]": 505,
    "[inst]": 506,
    "[solo]": 507,
    "[bridge]": 508,
    "[hook]": 509,
    "[break]": 510,
    "[stop]": 511,
    "[space]": 512
}

class CNENTokenizer():
    def __init__(self):
        curr_path = os.path.abspath(__file__)
        vocab_path = os.path.join(os.path.dirname(curr_path), "g2p/g2p/vocab.json")
        with open(vocab_path, 'r', encoding='utf-8') as file:
            self.phone2id:dict = json.load(file)['vocab']
        self.id2phone = {v:k for (k, v) in self.phone2id.items()}
        from g2p.g2p_generation import chn_eng_g2p
        self.tokenizer = chn_eng_g2p
    def encode(self, text):
        phone, token = self.tokenizer(text)
        token = [x+1 for x in token]
        return token
    def decode(self, token):
        return "|".join([self.id2phone[x-1] for x in token])

# Global variables for models
diffrhythm2 = None
mulan = None
lrc_tokenizer = None
decoder = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models(repo_id="ASLP-lab/DiffRhythm2"):
    global diffrhythm2, mulan, lrc_tokenizer, decoder
    
    print(f"Loading models from {repo_id}...")
    
    diffrhythm2_ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
        local_dir="./ckpt",
    )
    diffrhythm2_config_path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        local_dir="./ckpt",
    )
    with open(diffrhythm2_config_path, 'r', encoding='utf-8') as f:
        model_config = json.load(f)

    model_config['use_flex_attn'] = False
    diffrhythm2 = CFM(
        transformer=DiT(**model_config),
        num_channels=model_config['mel_dim'],
        block_size=model_config['block_size'],
    ).to(device)

    from safetensors.torch import load_file
    ckpt = load_file(diffrhythm2_ckpt_path)
    diffrhythm2.load_state_dict(ckpt)
    
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir="./ckpt").to(device)
    lrc_tokenizer = CNENTokenizer()

    decoder_ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename="decoder.bin",
        local_dir="./ckpt",
    )
    decoder_config_path = hf_hub_download(
        repo_id=repo_id,
        filename="decoder.json",
        local_dir="./ckpt",
    )
    decoder = Generator(decoder_config_path, decoder_ckpt_path).to(device)
    
    if device.type != 'cpu':
        diffrhythm2 = diffrhythm2.half()
        decoder = decoder.half()
    
    print("Models loaded successfully!")

STRUCT_PATTERN = re.compile(r'^\[.*?\]$')

def parse_lyrics(lyrics: str):
    lyrics_with_time = []
    lyrics = lyrics.split("\n")
    get_start = False
    for line in lyrics:
        line = line.strip()
        if not line:
            continue
        struct_flag = STRUCT_PATTERN.match(line)
        if struct_flag:
            struct_idx = STRUCT_INFO.get(line.lower(), None)
            if struct_idx is not None:
                if struct_idx == STRUCT_INFO['[start]']:
                    get_start = True
                lyrics_with_time.append([struct_idx, STRUCT_INFO['[stop]']])
            else:
                continue
        else:
            tokens = lrc_tokenizer.encode(line.strip())
            tokens = tokens + [STRUCT_INFO['[stop]']]
            lyrics_with_time.append(tokens)
    if len(lyrics_with_time) != 0 and not get_start:
        lyrics_with_time = [[STRUCT_INFO['[start]'], STRUCT_INFO['[stop]']]] + lyrics_with_time
    return lyrics_with_time

def make_fake_stereo(audio, sampling_rate):
    left_channel = audio
    right_channel = audio.copy()
    right_channel = right_channel * 0.8
    delay_samples = int(0.01 * sampling_rate)
    right_channel = np.roll(right_channel, delay_samples)
    right_channel[:,:delay_samples] = 0
    stereo_audio = np.concatenate([left_channel, right_channel], axis=0)
    return stereo_audio

def generate_song(lyrics_text, style_prompt_input, style_audio_input, duration, steps, cfg_strength, fake_stereo):
    if diffrhythm2 is None:
        load_models()

    # Handle empty lyrics for instrumental generation
    if not lyrics_text.strip():
        lyrics_text = "[start]\n[inst]\n[end]"

    # Preprocess lyrics
    lyrics_token = parse_lyrics(lyrics_text)
    lyrics_token = torch.tensor(sum(lyrics_token, []), dtype=torch.long, device=device)

    # Preprocess style prompt
    if style_audio_input is not None:
        # Use audio prompt
        prompt_wav, sr = torchaudio.load(style_audio_input)
        prompt_wav = torchaudio.functional.resample(prompt_wav.to(device), sr, 24000)
        if prompt_wav.shape[1] > 24000 * 10:
            start = random.randint(0, prompt_wav.shape[1] - 24000 * 10)
            prompt_wav = prompt_wav[:, start:start+24000*10]
        prompt_wav = prompt_wav.mean(dim=0, keepdim=True)
        with torch.no_grad():
            style_prompt_embed = mulan(wavs = prompt_wav)
    else:
        # Use text prompt
        with torch.no_grad():
            style_prompt_embed = mulan(texts = [style_prompt_input])
    
    style_prompt_embed = style_prompt_embed.to(device).squeeze(0)
    if device.type != 'cpu':
        style_prompt_embed = style_prompt_embed.half()

    # Inference
    with torch.inference_mode():
        # Ensure style_prompt is in the correct dtype
        if device.type != 'cpu':
            style_prompt_embed = style_prompt_embed.half()
        else:
            style_prompt_embed = style_prompt_embed.float()

        latent = diffrhythm2.sample_block_cache(
            text=lyrics_token.unsqueeze(0),
            duration=int(duration * 5),
            style_prompt=style_prompt_embed.unsqueeze(0),
            steps=steps,
            cfg_strength=cfg_strength,
            process_bar=True,
        )
        latent = latent.transpose(1, 2)
        audio = decoder.decode_audio(latent, overlap=5, chunk_size=20)

        audio_np = audio.float().cpu().numpy().squeeze()[None, :]
        num_channels = 1
        if fake_stereo:
            audio_np = make_fake_stereo(audio_np, decoder.h.sampling_rate)
            num_channels = 2
        
        output_path = "generated_song.wav"
        with pedalboard.io.AudioFile(output_path, "w", decoder.h.sampling_rate, num_channels) as f:
            f.write(audio_np)
            
    return output_path

# Gradio UI
with gr.Blocks(title="DiffRhythm 2 Demo") as demo:
    gr.Markdown("# 🎵 DiffRhythm 2: Efficient Song Generation")
    gr.Markdown("输入歌词和风格描述（或上传参考音频），生成属于你的歌曲。**如果不填歌词，将生成纯音乐（Instrumental）。**")
    
    with gr.Row():
        with gr.Column():
            lyrics = gr.Textbox(
                label="歌词 (Lyrics)", 
                placeholder="输入歌词... (留空生成纯音乐)\n[start]\n[intro]\n[verse]\n...",
                lines=10
            )
            style_text = gr.Textbox(
                label="风格描述 (Style Text Prompt)", 
                placeholder="Pop, Male Vocal, Emotional",
                value="Pop, Male Vocal"
            )
            style_audio = gr.Audio(label="参考音频 (Optional Audio Prompt)", type="filepath")
            
            with gr.Accordion("高级设置 (Advanced Settings)", open=False):
                duration = gr.Slider(minimum=10, maximum=300, value=60, step=10, label="时长 (Duration in seconds)")
                steps = gr.Slider(minimum=1, maximum=50, value=16, step=1, label="步数 (Steps)")
                cfg = gr.Slider(minimum=1.0, maximum=5.0, value=2.0, step=0.1, label="CFG Strength")
                stereo = gr.Checkbox(label="伪立体声 (Fake Stereo)", value=True)
            
            generate_btn = gr.Button("生成歌曲 (Generate)", variant="primary")
            
        with gr.Column():
            output_audio = gr.Audio(label="生成的歌曲 (Generated Song)")

    generate_btn.click(
        fn=generate_song,
        inputs=[lyrics, style_text, style_audio, duration, steps, cfg, stereo],
        outputs=output_audio
    )

if __name__ == "__main__":
    demo.launch()
