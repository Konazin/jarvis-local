import time

import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

print("Carregando Qwen3-TTS...")
start = time.perf_counter()

model = Qwen3TTSModel.from_pretrained(
    MODEL,
    device_map="cpu",
    dtype=torch.float32,
    attn_implementation="sdpa",
)

print(f"Modelo carregado em {time.perf_counter() - start:.2f}s")

texts = [
    "Sistema online.",
    "Kona, o Spotify foi aberto.",
    "A memória está em setenta por cento.",
]

for i, text in enumerate(texts, start=1):
    print(f"\n--- Geração {i} ---")

    start = time.perf_counter()

    wavs, sr = model.generate_custom_voice(
        text=text,
        language="Portuguese",
        speaker="Ono_Anna",
    )

    elapsed = time.perf_counter() - start
    duration = len(wavs[0]) / sr
    rtf = elapsed / duration

    sf.write(f"/tmp/yuki-{i}.wav", wavs[0], sr)

    print(f"Tempo: {elapsed:.2f}s")
    print(f"Áudio: {duration:.2f}s")
    print(f"RTF: {rtf:.2f}")
