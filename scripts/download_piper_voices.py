import os
import requests

VOICES = {
    "ru_RU-irina-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx.json",
    },
    "ru_RU-denis-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx.json",
    },
    "ru_RU-dmitri-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx.json",
    },
    "en_US-lessac-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    }
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "piper_voices")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_file(url, target_path):
    if os.path.exists(target_path):
        print(f"Already exists: {os.path.basename(target_path)}")
        return
    print(f"Downloading {url} -> {os.path.basename(target_path)} ...")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(target_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    print(f"Saved: {os.path.basename(target_path)}")

if __name__ == "__main__":
    for voice_name, urls in VOICES.items():
        onnx_file = os.path.join(OUTPUT_DIR, f"{voice_name}.onnx")
        json_file = os.path.join(OUTPUT_DIR, f"{voice_name}.onnx.json")
        download_file(urls["onnx"], onnx_file)
        download_file(urls["json"], json_file)
    print("All Piper voices downloaded successfully!")
