import os
import sys
import time
import socket
import soundfile as sf
import torch
import whisper
import gradio as gr

# Setup CosyVoice environment paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COSY_DIR = os.path.join(BASE_DIR, "CosyVoice_repo")
MATCHA_DIR = os.path.join(COSY_DIR, "third_party", "Matcha-TTS")
if COSY_DIR not in sys.path:
    sys.path.insert(0, COSY_DIR)
if MATCHA_DIR not in sys.path:
    sys.path.insert(0, MATCHA_DIR)

from cosyvoice.cli.cosyvoice import AutoModel

print("Загрузка модели распознавания речи Whisper (base)...")
whisper_model = whisper.load_model("base")

print("Загрузка модели CosyVoice 3.0 в память GPU...")
MODEL_DIR = os.path.join(BASE_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B")
if not os.path.exists(MODEL_DIR) or not os.path.exists(os.path.join(MODEL_DIR, "cosyvoice3.yaml")):
    print("Веса модели не найдены. Запуск автоматической загрузки...")
    import subprocess
    subprocess.run([sys.executable, os.path.join(BASE_DIR, "download_cosyvoice.py")], check=True)

cosy_model = AutoModel(model_dir=MODEL_DIR, fp16=True)
print("Модели успешно загружены и готовы к работе!")

SAMPLE_AUDIO = os.path.join(BASE_DIR, "test.wav") if os.path.exists(os.path.join(BASE_DIR, "test.wav")) else None

def prepare_prompt_audio(audio_path, max_sec=20.0):
    try:
        info = sf.info(audio_path)
        if info.duration > 25.0:
            data, sr = sf.read(audio_path)
            max_samples = int(max_sec * sr)
            trimmed = data[:max_samples]
            temp_path = os.path.join(BASE_DIR, "temp_prompt_trimmed.wav")
            sf.write(temp_path, trimmed, sr)
            gr.Info(f"Образец длился {info.duration:.1f} сек. Аудио автоматически обрезано до первых {max_sec:.0f} секунд.")
            return temp_path
        elif info.duration < 1.0:
            gr.Warning("Образец аудио слишком короткий (менее 1 секунды). Запишите хотя бы 3–5 секунд.")
    except Exception as e:
        print(f"Ошибка проверки аудио: {e}")
    return audio_path

def auto_transcribe_audio(audio_path):
    if not audio_path:
        return ""
    try:
        res = whisper_model.transcribe(audio_path)
        text = res.get("text", "").strip()
        lang = res.get("language", "ru")
        gr.Info(f"Текст из образца распознан ({lang}): «{text}»")
        return text
    except Exception as e:
        print(f"Ошибка Whisper: {e}")
        return ""

def clone_voice(tts_text, prompt_audio, prompt_text, speed):
    if not tts_text or not tts_text.strip():
        gr.Warning("Пожалуйста, введите текст для синтеза речи.")
        return None, "Ошибка: текст для синтеза пуст."
    
    if not prompt_audio:
        gr.Warning("Пожалуйста, загрузите или запишите образец голоса (3–15 секунд)!")
        return None, "Ошибка: не предоставлен образец голоса."

    start_time = time.time()
    processed_audio = prepare_prompt_audio(prompt_audio, max_sec=20.0)
    
    auto_transcribed_note = ""
    # If prompt text wasn't provided manually, transcribe it with Whisper for highest cloning quality
    if not prompt_text or not prompt_text.strip():
        try:
            print("Автоматическое распознавание текста образца через Whisper...")
            transcribe_res = whisper_model.transcribe(processed_audio)
            prompt_text = transcribe_res.get("text", "").strip()
            detected_lang = transcribe_res.get("language", "ru")
            print(f"Распознан текст образца ({detected_lang}): {prompt_text}")
            auto_transcribed_note = f"\n[Whisper] Текст образца распознан автоматически ({detected_lang}): «{prompt_text}»"
        except Exception as e:
            print(f"Ошибка Whisper: {e}")
            prompt_text = ""

    formatted_prompt = f"You are a helpful assistant.<|endofprompt|>{prompt_text.strip()}"
    tts_text_clean = tts_text.strip()

    try:
        if prompt_text and prompt_text.strip():
            outputs = list(cosy_model.inference_zero_shot(
                tts_text_clean, 
                formatted_prompt, 
                processed_audio, 
                speed=float(speed),
                text_frontend=False
            ))
        else:
            outputs = list(cosy_model.inference_cross_lingual(
                tts_text_clean, 
                processed_audio, 
                speed=float(speed),
                text_frontend=False
            ))

        if not outputs:
            return None, "Ошибка: модель не вернула аудио."

        audio_tensor = torch.concat([o['tts_speech'] for o in outputs], dim=1)
        audio_np = audio_tensor.squeeze(0).cpu().numpy()
        sr = cosy_model.sample_rate
        elapsed = time.time() - start_time
        duration = len(audio_np) / sr
        rtf = elapsed / max(duration, 0.01)

        status = (
            f"Успешно! Сгенерировано {duration:.2f} сек. речи за {elapsed:.2f} сек. "
            f"(RTF: {rtf:.2f}) на RTX 5080.{auto_transcribed_note}"
        )
        return (sr, audio_np), status
    except Exception as e:
        err_msg = str(e)
        print(f"Ошибка инференса: {err_msg}")
        gr.Warning(f"Ошибка при синтезе: {err_msg}")
        return None, f"Ошибка: {err_msg}"

with gr.Blocks(title="CosyVoice 3.0 — Клонирование голоса") as demo:
    gr.Markdown(
        """
        # CosyVoice 3.0 — Локальное клонирование голоса (Voice Cloning)
        Диффузионная нейросеть нового поколения для клонирования голоса с аппаратным ускорением на **NVIDIA GeForce RTX 5080**.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=5):
            tts_input = gr.Textbox(
                label="Текст, который нужно произнести",
                placeholder="Введите текст для озвучки (русский, английский и др.)...",
                value="Здравствуйте! Это проверка клонирования голоса в нейросети CosyVoice 3. Теперь речь звучит чисто и разборчиво.",
                lines=4
            )
            
            prompt_audio_input = gr.Audio(
                label="Образец голоса для клонирования (референс: 3–15 сек речи)",
                sources=["upload", "microphone"],
                type="filepath",
                value=SAMPLE_AUDIO
            )

            with gr.Accordion("Настройки и текст образца", open=False):
                with gr.Row():
                    prompt_text_input = gr.Textbox(
                        label="Текст, звучащий в образце голоса (опционально)",
                        placeholder="Если оставить пустым — нейросеть Whisper распознает текст образца автоматически!",
                        lines=2,
                        scale=4
                    )
                    transcribe_btn = gr.Button("Распознать", scale=1)
                
                speed_slider = gr.Slider(
                    label="Скорость речи",
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.05
                )

            generate_btn = gr.Button("Клонировать голос и сгенерировать речь", variant="primary", size="lg")

        with gr.Column(scale=5):
            audio_output = gr.Audio(
                label="Результат клонирования",
                type="numpy",
                interactive=False
            )
            status_box = gr.Textbox(
                label="Статус генерации",
                lines=4,
                interactive=False
            )
            
            gr.Markdown(
                """
                ### Советы для идеального результата:
                1. **Качество записи:** образец без фоновой музыки, посторонних шумов и сильного эха дает наилучшее сходство.
                2. **Длина образца:** оптимально **5–12 секунд** связной речи.
                3. **Автоматический Whisper:** модель автоматически распознает произнесенный текст в образце для точного акустического выравнивания.
                4. **Языки:** поддерживаются русский, английский, китайский, японский, корейский, французский, немецкий, испанский и итальянский.
                """
            )

    transcribe_btn.click(
        auto_transcribe_audio,
        inputs=[prompt_audio_input],
        outputs=[prompt_text_input]
    )

    generate_btn.click(
        clone_voice,
        inputs=[tts_input, prompt_audio_input, prompt_text_input, speed_slider],
        outputs=[audio_output, status_box]
    )

def find_free_port(start_port=7865):
    port = start_port
    while port < start_port + 50:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    return start_port

if __name__ == "__main__":
    free_port = find_free_port(7865)
    print("\n" + "="*50)
    print(f"CosyVoice 3.0 WebUI: http://127.0.0.1:{free_port}")
    print("="*50 + "\n")
    demo.queue().launch(
        server_name="127.0.0.1", 
        server_port=free_port, 
        theme=gr.themes.Soft(),
        inbrowser=True
    )
