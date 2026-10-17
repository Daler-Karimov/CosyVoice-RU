import sys
import os
import torch
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "CosyVoice_repo"))
sys.path.insert(0, os.path.join(BASE_DIR, "CosyVoice_repo", "third_party", "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel

print("Loading CosyVoice 3.0...")
model = AutoModel(model_dir=os.path.join(BASE_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B"), fp16=True)

# Test 1: Cross-lingual with prompt audio
text_ru = "You are a helpful assistant.<|endofprompt|>Здравствуйте, это проверка синтеза русской речи."
prompt_audio = os.path.join(BASE_DIR, "CosyVoice_repo", "asset", "zero_shot_prompt.wav")
if not os.path.exists(prompt_audio):
    prompt_audio = os.path.join(BASE_DIR, "test.wav")

print("Generating with asset prompt audio:", prompt_audio)
outputs = list(model.inference_cross_lingual(text_ru, prompt_audio, text_frontend=False))
audio = torch.concat([o['tts_speech'] for o in outputs], dim=1)
sf.write("test_ru_cross.wav", audio.squeeze(0).cpu().numpy(), model.sample_rate)
print("Saved test_ru_cross.wav! Samples:", audio.shape)
