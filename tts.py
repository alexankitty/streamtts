from pydantic import BaseModel
import uvicorn
from glob import glob
from lib.gen import gen, replace_vocals, video_info

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse


class TtsRequest(BaseModel):
    text: str
    voice: str

class YtReplaceRequest(BaseModel):
    voice: str
    url: str
    pitch: int = 0

class VideoInfoRequest(BaseModel):
    url: str

def setup_routes(app: FastAPI):
    @app.post("/tts")
    async def ttspost(request: TtsRequest):
        result = gen(request.text, request.voice)
        if not result:
            raise HTTPException(status_code=400, detail="Failed to generate voice.")
        return Response(content=result, media_type="audio/wav")
    @app.get("/tts")
    async def ttsget(text: str = '', voice: str = ''):
        result = gen(text, voice)
        if not result:
            raise HTTPException(status_code=400, detail="Failed to generate voice.")
        return Response(content=result, media_type="audio/wav")
    
    @app.get("/voices")
    async def voicesget():
        voices = glob("*", root_dir="models")
        return JSONResponse(content=voices)
    
    @app.get("/replace_yt")
    async def ryt_get(voice: str, url: str, pitch: int = 0):
        result = replace_vocals(url, voice, pitch)
        if not result:
            raise HTTPException(status_code=400, detail="Fail to replace youtube video vocals.")
        return Response(content=result, media_type="audio/mp3")
    @app.post("/replace_yt")
    async def ryt_post(request: YtReplaceRequest):
        result = replace_vocals(request.url, request.voice, request.pitch)
        if not result:
            raise HTTPException(status_code=400, detail="Fail to replace youtube video vocals.")
        return Response(content=result, media_type="audio/mp3")
    
    @app.get("/ytinfo")
    async def ytinfo_get(url: str):
        info = video_info(url)
        return JSONResponse(content=info)
    @app.post("/ytinfo")
    async def ytinfo_post(request: VideoInfoRequest):
        info = video_info(request.url)
        return JSONResponse(content=info)

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