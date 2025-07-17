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
load_dotenv(".env")


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
            index_rate=config['index_rate']
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

