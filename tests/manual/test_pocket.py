import time

import scipy.io.wavfile
from pocket_tts import TTSModel


TEXT = (
    "Kona, o sistema está online. "
    "A memória está em setenta por cento."
)

print("Carregando Pocket TTS...")
start = time.perf_counter()

model = TTSModel.load_model(
    language="portuguese",
    quantize=True,
)

load_time = time.perf_counter() - start
print(f"Modelo carregado em {load_time:.2f}s")

print("Carregando voz...")
start = time.perf_counter()

voice = model.get_state_for_audio_prompt("rafael")

voice_time = time.perf_counter() - start
print(f"Voz carregada em {voice_time:.2f}s")

for i in range(1, 4):
    print(f"\n--- Geração {i} ---")

    start = time.perf_counter()

    audio = model.generate_audio(
        voice,
        TEXT,
    )

    generation_time = time.perf_counter() - start

    audio_np = audio.detach().cpu().numpy()
    duration = len(audio_np) / model.sample_rate
    rtf = generation_time / duration

    scipy.io.wavfile.write(
        f"/tmp/pocket-{i}.wav",
        model.sample_rate,
        audio_np,
    )

    print(f"Geração: {generation_time:.2f}s")
    print(f"Áudio: {duration:.2f}s")
    print(f"RTF: {rtf:.2f}")
