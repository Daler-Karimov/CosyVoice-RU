# -*- coding: utf-8 -*-
import sys
import os
import torch
import soundfile as sf
import whisper

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "CosyVoice_repo"))
sys.path.insert(0, os.path.join(BASE_DIR, "CosyVoice_repo", "third_party", "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel

print("Loading model...")
model_dir = os.path.join(BASE_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B")
model = AutoModel(model_dir=model_dir, fp16=True)

# Generate an English prompt wav using Piper to ensure clean English reference
from piper import PiperVoice
voice_en = PiperVoice.load("piper_voices/en_US-lessac-medium.onnx")
chunks = list(voice_en.synthesize("Welcome to the artificial intelligence demonstration."))
audio_en = torch.tensor(torch.concat([torch.from_numpy(c.audio_float_array) for c in chunks]))
sf.write("en_prompt.wav", audio_en.numpy(), 22050)

prompt_wav = "en_prompt.wav"
prompt_text = "You are a helpful assistant.<|endofprompt|>Welcome to the artificial intelligence demonstration."
tts_text = "Today we are testing zero-shot speech synthesis on a powerful graphics card."

print("Running zero_shot in English...")
outputs = list(model.inference_zero_shot(tts_text, prompt_text, prompt_wav, text_frontend=False))
audio = torch.concat([o['tts_speech'] for o in outputs], dim=1)
sf.write("test_pure_en.wav", audio.squeeze(0).cpu().numpy(), model.sample_rate)
print("Saved test_pure_en.wav!")

w = whisper.load_model('base')
res = w.transcribe('test_pure_en.wav', language='en')
print("Whisper recognized:", res['text'])
