import os
import sys

target_dir = os.path.join(os.path.dirname(__file__), "pretrained_models", "Fun-CosyVoice3-0.5B")
print(f"Загрузка модели Fun-CosyVoice3-0.5B-2512 в: {target_dir}")

os.makedirs(target_dir, exist_ok=True)

success = False
try:
    print("Попытка загрузки через Hugging Face...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        local_dir=target_dir,
        resume_download=True
    )
    print("Модель успешно загружена через Hugging Face!")
    success = True
except Exception as e:
    print(f"Ошибка загрузки через Hugging Face: {e}")

if not success:
    try:
        print("Попытка загрузки через ModelScope (зеркало)...")
        from modelscope import snapshot_download
        snapshot_download(
            "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
            local_dir=target_dir
        )
        print("Модель успешно загружена через ModelScope!")
        success = True
    except Exception as e:
        print(f"Ошибка загрузки через ModelScope: {e}")

if not success:
    print("Не удалось автоматически скачать модель. Проверьте интернет-соединение.")
    sys.exit(1)
