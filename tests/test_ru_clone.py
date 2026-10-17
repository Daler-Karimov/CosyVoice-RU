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

# Prompt is Russian test.wav
prompt_wav = os.path.join(BASE_DIR, "test.wav")
prompt_text = "You are a helpful assistant.<|endofprompt|>Привет! Это проверочный образец голоса для клонирования."

# New Russian sentence to synthesize in that voice
tts_text = "Сегодня прекрасная погода, и мы тестируем клонирование голоса."

print("Running zero_shot with Russian prompt...")
outputs = list(model.inference_zero_shot(tts_text, prompt_text, prompt_wav, text_frontend=False))
audio = torch.concat([o['tts_speech'] for o in outputs], dim=1)
sf.write("test_ru_same_lang.wav", audio.squeeze(0).cpu().numpy(), model.sample_rate)
print("Saved test_ru_same_lang.wav!")

w = whisper.load_model('base')
res = w.transcribe('test_ru_same_lang.wav', language='ru')
with open('whisper_result_ru.txt', 'w', encoding='utf-8') as f:
    f.write(res['text'])
print("Whisper transcribed:", res['text'].encode('utf-8'))
