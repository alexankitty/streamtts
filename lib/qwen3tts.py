import os
import torch
import soundfile as sf
import tempfile
from qwen_tts import Qwen3TTSModel

_model = None
_voice_clone_prompts = {}

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

def _get_model():
    global _model
    if _model is None:
        _model = Qwen3TTSModel.from_pretrained(
            MODEL_NAME,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
    return _model


def _get_voice_clone_prompt(voice_dir: str):
    if voice_dir in _voice_clone_prompts:
        return _voice_clone_prompts[voice_dir]

    clip_path = os.path.join(voice_dir, "voice_clip.wav")
    script_path = os.path.join(voice_dir, "voice_script.txt")

    with open(script_path, "r", encoding="utf-8") as f:
        ref_text = f.read().strip()

    model = _get_model()
    prompt = model.create_voice_clone_prompt(
        ref_audio=clip_path,
        ref_text=ref_text,
    )
    _voice_clone_prompts[voice_dir] = prompt
    return prompt


def can_use_qwen3tts(config: dict, voice_dir: str) -> bool:
    if not config.get("qwen3tts"):
        return False
    clip_path = os.path.join(voice_dir, "voice_clip.wav")
    script_path = os.path.join(voice_dir, "voice_script.txt")
    return os.path.isfile(clip_path) and os.path.isfile(script_path)


def generate(text: str, voice_dir: str, language: str = "Auto") -> bytes:
    model = _get_model()
    prompt = _get_voice_clone_prompt(voice_dir)

    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=prompt,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, wavs[0], sr)

    with open(tmp_path, "rb") as f:
        output = f.read()
    os.unlink(tmp_path)
    return output
