from scipy.io import wavfile
from misaki import en, espeak
import tempfile
import os
import random
from dotenv import load_dotenv
from hatesonar import Sonar
from lib.config import ModelConfig, loadConfig
from lib.kokoro import gen_tts, blend_voices
from rvc.modules.vc.modules import VC
from pydub import AudioSegment
from audio_separator.separator import Separator
from yt_dlp import YoutubeDL
import re

load_dotenv(".env")

separator = Separator()

separator.load_model(model_filename='vocals_mel_band_roformer.ckpt')
output_names = {
    "Vocals": "vocals_output",
    "Instrumental": "instrumental_output",
    "Other": "instrumental_output"
}

ydl_opts = {
    'format': 'bestaudio/best',
    # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
    'postprocessors': [{  # Extract audio using ffmpeg
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
    }],
    'outtmpl': '%(title)s.%(ext)s',
    'restrictfilenames': True,
    'cookiefile': os.path.join(os.getcwd(), 'cookies.txt')
}


# Misaki G2P with espeak-ng fallback
fallback = espeak.EspeakFallback(british=False)
g2p = en.G2P(trf=False, british=False, fallback=fallback)

# rvc
#rvc = RVCInference(device="cuda:0")
#rvc.set_params(f0up_key=8, f0method="rmvpe")
vc = VC()


# hate speech replacers
with open(os.path.join(os.getcwd(), "assets/unhateful-phrases.txt"), "r") as f:
    unhateful_phrases = f.readlines()
sonar = Sonar()
hatespeech_threshold = 0.3

def gen(text: str, voice: str = 'miku') -> bytes:
    config: ModelConfig = loadConfig(voice)
    if not config:
        return
    print(config)
    text = checkText(text)
    voice = voice.lower()

    modelPath = os.path.join(os.getcwd(), f'models/{voice}/model.pth')
    indexPath = os.path.join(os.getcwd(), f'models/{voice}/model.index')

    with tempfile.NamedTemporaryFile(delete=False) as tmp_output:
        output_path = tmp_output.name

    # Phonemize
    # phonemes, _ = g2p(text)

    blend = blend_voices(config['voices'])
    samples, sample_rate = gen_tts(text, blend, speed=config['speed'])
    wavfile.write(output_path, sample_rate, samples)

    vc.get_vc(modelPath)
    tgt_sr, audio_opt, times, _ = vc.vc_single(
            1,
            output_path,
            config['pitch'],
            'rmvpe',
            index_file=indexPath,
            filter_radius=config['filter_radius'],
            protect=config['protect'],
            index_rate=config['index_rate'],
      )
    wavfile.write(output_path, tgt_sr, audio_opt)

    with open(output_path, "rb") as file:
        output = file.read()
    os.unlink(output_path)
    return output

def checkText(text: str):
    if sonar.ping(text)['classes'][0]['confidence'] >= hatespeech_threshold:
        return random.choice(unhateful_phrases)
    return text

def replace_vocals(url: str, name: str, pitch: int):
    url = re.sub(r'list=\w+', '', url)
    model = f"./models/{name}/model.pth"
    model_index = f"./models/{name}/model.index"
    vc.get_vc(model)
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        info_with_audio_extension = dict(info)
        info_with_audio_extension['ext'] = 'wav'
        final_filename = ydl.prepare_filename(info_with_audio_extension)

    output_files = separator.separate(final_filename, output_names)

    tgt_sr, audio_opt, times, _ = vc.vc_single(
                1,
                os.path.join(os.getcwd() ,"vocals_output.wav"),
                pitch,
                'rmvpe',
                index_file=model_index,
                filter_radius=10,
                protect=0,
                index_rate=0.33
        )
    wavfile.write("vocals_output.wav", tgt_sr, audio_opt)

    sound1 = AudioSegment.from_file("vocals_output.wav")
    sound2 = AudioSegment.from_file("instrumental_output.wav")

    combined = sound1.overlay(sound2)

    os.unlink(final_filename)
    os.unlink("vocals_output.wav")
    os.unlink("instrumental_output.wav")

    with tempfile.NamedTemporaryFile(delete=False) as tmp_output:
        output_path = tmp_output.name
    
    combined.export(output_path, format='mp3', bitrate="150k")
    with open(output_path, "rb") as file:
        final_output = file.read()

    os.unlink(output_path)
    return final_output

def video_info(url: str):
    url = re.sub(r'list=\w+', '', url)
    with YoutubeDL() as ydl: 
        info_dict = ydl.extract_info(url, download=False)
        info = {
            "title": info_dict.get('title', None)
        }
        return info