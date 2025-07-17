import os
import onnxruntime as ort
from kokoro_onnx import Kokoro
import numpy as np

privders = ort.get_available_providers()
print("Available providers:", privders)  # Make sure CUDAExecutionProvider is listed
print(f'Is CUDA available: {"CUDAExecutionProvider" in privders}')

kokoro = Kokoro(os.path.join(os.getcwd(), 'assets/kokoro-v1.0.onnx'),
                       os.path.join(os.getcwd(), 'assets/voices-v1.0.bin'))

def gen_tts(text: str, voice, speed: float, lang: str = "en-us", is_phonemes: bool = False):
    return kokoro.create(text, voice, speed=speed, lang=lang, is_phonemes=is_phonemes)

def blend_voices(voices: dict) -> np.ndarray:
    kokoro_voices = []
    for voice, strength in voices.items():
        if len(voices) == 1:
            return kokoro.get_voice_style(voice)
        kokoro_voices.append(kokoro.get_voice_style(voice) * strength)
    return np.add(*kokoro_voices)