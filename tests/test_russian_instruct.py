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
model_dir = os.path.join(BASE_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B")
model = AutoModel(model_dir=model_dir, fp16=True)

# Test 1: instruct2 with Russian language instruction
prompt_wav = os.path.join(BASE_DIR, "CosyVoice_repo", "asset", "zero_shot_prompt.wav")
instruct = "You are a helpful assistant. Please speak in Russian.<|endofprompt|>"
tts_text = "Привет! Это проверка синтеза речи на русском языке."

print("Running inference_instruct2...")
outputs = list(model.inference_instruct2(tts_text, instruct, prompt_wav, text_frontend=False))
audio = torch.concat([o['tts_speech'] for o in outputs], dim=1)
sf.write("test_instruct_output.wav", audio.squeeze(0).cpu().numpy(), model.sample_rate)
print("Saved test_instruct_output.wav!")

# Check with Whisper
import whisper
w = whisper.load_model('base')
res = w.transcribe('test_instruct_output.wav', language='ru')
with open('whisper_result.txt', 'w', encoding='utf-8') as f:
    f.write(res['text'])
print("Whisper result written to whisper_result.txt!")
