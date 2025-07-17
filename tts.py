from pydantic import BaseModel
import uvicorn
from glob import glob
from lib.gen import gen

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse


class TtsRequest(BaseModel):
    text: str
    voice: str

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