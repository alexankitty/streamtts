import soundfile as sf
from misaki import en, espeak
import numpy as np
import tempfile
import os
from pydantic import BaseModel
import uvicorn
import random
from glob import glob
import json

from kokoro_onnx import Kokoro

from rvc_python.infer import RVCInference

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

from hatesonar import Sonar

# Misaki G2P with espeak-ng fallback
fallback = espeak.EspeakFallback(british=False)
g2p = en.G2P(trf=False, british=False, fallback=fallback)

# Kokoro
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

# rvc
rvc = RVCInference(device="cuda:0")
rvc.set_params(f0up_key=8, f0method="rmvpe")

# hate speech replacers
with open("unhateful-phrases.txt", "r") as f:
    unhateful_phrases = f.readlines()
sonar = Sonar()
hatespeech_threshold = 0.3

class TtsRequest(BaseModel):
    text: str
    voice: str

def setup_routes(app: FastAPI):
    @app.post("/tts")
    async def ttspost(request: TtsRequest):
        result = gen(request.text, request.voice)
        return Response(content=result, media_type="audio/wav")
    @app.get("/tts")
    async def ttsget(text: str = '', voice: str = ''):
        result = gen(text, voice)
        return Response(content=result, media_type="audio/wav")
    @app.get("/voices")
    async def voicesget():
        voices = glob("*", root_dir="models")
        return JSONResponse(content=voices)
    

def gen(text, voice):
    text = checkText(text)
    if not voice:
        voice = "miku"
    voice = voice.lower()
    with open(f"./models/{voice}/pitch", "r") as f:
        pitch = f.readline().strip()
    
    rvc.set_params(f0up_key=pitch)
    rvc.load_model(f"./models/{voice}/model.pth", index_path=f"./models/{voice}/model.index")

    # Phonemize
    phonemes, _ = g2p(text)

    # blends
    voice1: np.ndarray = kokoro.get_voice_style("af_kore")
    voice2: np.ndarray = kokoro.get_voice_style("am_adam")
    blend = np.add(voice1 * (50 / 100), voice2 * (50 / 100))

    # Create
    samples, sample_rate = kokoro.create(phonemes, blend, is_phonemes=True)

    # Save
    with tempfile.NamedTemporaryFile(delete=False) as tmp_output:
        output_path = tmp_output.name
    
    sf.write(output_path, samples, sample_rate, format="wav")
    rvc.infer_file(output_path, output_path)
    with open(output_path, "rb") as f:
        output = f.read()
    os.unlink(output_path)
    return output

def checkText(text):
    if sonar.ping(text)['classes'][0]['confidence'] >= hatespeech_threshold:
        return random.choice(unhateful_phrases)
    return text

def create_app():
    app = FastAPI()

    # Add CORS middleware
    origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    setup_routes(app)
    return app

if __name__ == '__main__':
    app = create_app()
    # Set up server options
    host = "0.0.0.0"
    print(f"Starting API server on {host}:8081")

    # Run the server
    uvicorn.run(app, host=host, port=8081)