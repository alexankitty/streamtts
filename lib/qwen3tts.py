import os
import torch
import soundfile as sf
import tempfile
from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

_model = None

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
PROMPT_FILENAME = "voice_clone_prompt.pt"

def _get_model():
    global _model
    if _model is None:
        _model = Qwen3TTSModel.from_pretrained(
            MODEL_NAME,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
    return _model


def load_model():
    print(f"Loading Qwen3-TTS model: {MODEL_NAME}")
    _get_model()
    print("Qwen3-TTS model loaded.")


def _save_prompt(voice_dir: str, prompt_items):
    data = []
    for item in prompt_items:
        data.append({
            "ref_code": item.ref_code.cpu() if item.ref_code is not None else None,
            "ref_spk_embedding": item.ref_spk_embedding.cpu(),
            "x_vector_only_mode": item.x_vector_only_mode,
            "icl_mode": item.icl_mode,
            "ref_text": item.ref_text,
        })
    torch.save(data, os.path.join(voice_dir, PROMPT_FILENAME))


def _load_prompt(voice_dir: str):
    path = os.path.join(voice_dir, PROMPT_FILENAME)
    device = _get_model().device
    data = torch.load(path, map_location=device, weights_only=False)
    items = []
    for d in data:
        items.append(VoiceClonePromptItem(
            ref_code=d["ref_code"].to(device) if d["ref_code"] is not None else None,
            ref_spk_embedding=d["ref_spk_embedding"].to(device),
            x_vector_only_mode=d["x_vector_only_mode"],
            icl_mode=d["icl_mode"],
            ref_text=d["ref_text"],
        ))
    return items


def _get_voice_clone_prompt(voice_dir: str):
    prompt_path = os.path.join(voice_dir, PROMPT_FILENAME)
    if os.path.isfile(prompt_path):
        return _load_prompt(voice_dir)

    clip_path = os.path.join(voice_dir, "voice_clip.wav")
    script_path = os.path.join(voice_dir, "voice_script.txt")

    with open(script_path, "r", encoding="utf-8") as f:
        ref_text = f.read().strip()

    model = _get_model()
    prompt = model.create_voice_clone_prompt(
        ref_audio=clip_path,
        ref_text=ref_text,
    )
    _save_prompt(voice_dir, prompt)
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
