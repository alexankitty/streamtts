# Stream TTS
Extremely hacky implementation of leveraging kokoro and rvc to create a TTS in a certain synthetic robot's voice.

# Setup?
Whatever steps are required for your system to handle GPU inference. Repo assumes cuda, but can easily run on CPU with some less script kiddie bs  
Use pyenv and set your python version to 3.10 (or UV, or whatever other python management fluff you prefer)
```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Version of numpy sets off pip because of kokoro-onnx's requirements. It can be ignored (probably)  
You'll also need to get an RVC model and update it accordingly.
You'll also need to get a kokoro onnx model. [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)

# Usage
To start the server
```sh
python tts.py
```

To perform a request (or you can request from any language you want)
```sh
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"text":"I am TTS generated via API endpoint."}' \
  http://localhost:8081/tts -o test.wav
```

# Flaws
Endpoint assumes english. You could make Japanese audio with a language detection library, but mixed inputs are scuffed because phonemes are a PITA

