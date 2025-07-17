from pydantic import BaseModel
from yaml import safe_load, dump
import os
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

defaultConfig = {
    "pitch": 0,
    "voices": {
        "af_kore": 0.5,
        "am_adam": 0.5
    },
    "speed": 1.0,
    "index_rate": 1,
    "filter_radius": 3,
    "protect": 0.33
}

class ModelConfig(BaseModel):
    pitch: int
    voices: dict
    speed: float
    index_rate: float
    filter_radius: float
    protect: float


def loadConfig(modelname: str) -> ModelConfig:
    if not os.path.exists(os.path.join(os.getcwd(), f'models/{modelname}')):
        return False
    path = os.path.join(os.getcwd(), f'models/{modelname}/config.yaml')
    try: 
        with open(path, "r") as file:
            return safe_load(file)
    except Exception as e:
        print("No config present, creating new config")
        parsedConfig = dump(defaultConfig, Dumper=Dumper)
        with open(path, "w") as file:
            file.write(parsedConfig)
        return defaultConfig