# -*- coding: utf-8 -*-
import sys
import os
import torch
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "CosyVoice_repo"))
sys.path.insert(0, os.path.join(BASE_DIR, "CosyVoice_repo", "third_party", "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel

print("Loading model...")
model = AutoModel(model_dir=os.path.join(BASE_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B"), fp16=True)

# Audio sample from repo
prompt_wav = os.path.join(BASE_DIR, "CosyVoice_repo", "asset", "zero_shot_prompt.wav")
prompt_text = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"

# Russian TTS text
tts_text = "Привет! Это проверка синтеза русской речи с помощью модели Кози Войс три."

print("Synthesizing zero-shot...")
outputs = list(model.inference_zero_shot(tts_text, prompt_text, prompt_wav, text_frontend=False))
audio = torch.concat([o['tts_speech'] for o in outputs], dim=1)
sf.write("out_zero_shot.wav", audio.squeeze(0).cpu().numpy(), model.sample_rate)
print("Saved out_zero_shot.wav!")

import whisper
w = whisper.load_model('base')
res = w.transcribe('out_zero_shot.wav', language='ru')
print("Whisper recognized:", res['text'])
