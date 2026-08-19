import time

import numpy as np
import soundfile as sf

from kokoro import KPipeline


TEXT = (
    "Kona, o sistema está online. "
    "A memória está em setenta por cento."
)

print("Carregando Kokoro...")
start = time.perf_counter()

pipeline = KPipeline(lang_code="p")

load_time = time.perf_counter() - start
print(f"Modelo carregado em {load_time:.2f}s")

for i in range(1, 4):
    print(f"\n--- Geração {i} ---")

    start = time.perf_counter()

    generator = pipeline(
        TEXT,
        voice="pf_dora",
        speed=1.0,
    )

    chunks = []

    for _, _, audio in generator:
        chunks.append(audio)

    generation_time = time.perf_counter() - start

    audio = np.concatenate(chunks)

    sample_rate = 24000
    audio_duration = len(audio) / sample_rate
    rtf = generation_time / audio_duration

    sf.write(
        f"/tmp/kokoro-{i}.wav",
        audio,
        sample_rate,
    )

    print(f"Geração: {generation_time:.2f}s")
    print(f"Áudio: {audio_duration:.2f}s")
    print(f"RTF: {rtf:.2f}")
