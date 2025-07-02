# api/simli_router.py (Corrected Version - Single API Key)

import os
import httpx
import asyncio
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np
import librosa
# NOTE: The 'main' import is now done inside the function to prevent circular imports

# --- Configuration ---
# MODIFIED: Now using a single SIMLI_API_KEY environment variable
SIMLI_API_KEY = os.getenv("SIMLI_API_KEY") 
SIMLI_API_SESSION_URL = "https://api.simli.ai/startAudioToVideoSession"

router = APIRouter()

# --- Pydantic Models ---
class SimliSessionRequest(BaseModel):
    face_id: str
    # api_key is no longer expected from the client request body

class TtsRequest(BaseModel):
    text: str
    voice_id: str
    temperature: float = 0.75
    exaggeration: float = 0.80
    cfg: float = 2.0
    speed: float = 1.0

# --- Real TTS Generator Function ---
async def generate_audio_chunks_from_chatterbox(
    text: str, voice: str, exaggeration: float, temperature: float, cfg: float,
    chunk_size: int, context_window: int, fade_duration: float
) -> AsyncGenerator[bytes, None]:
    
    # FIX 1: Import moved inside function to prevent circular dependency at startup.
    from main import chatterbox_tts_model, PREPARED_VOICE_CONDITIONALS, TARGET_SAMPLE_RATE

    if chatterbox_tts_model is None:
        print("Error: ChatterboxTTS model is not loaded.")
        return

    cached_conditionals = PREPARED_VOICE_CONDITIONALS.get(voice)
    if not cached_conditionals:
        raise ValueError(f"Voice '{voice}' not pre-prepared or not in cache.")

    chatterbox_tts_model.conds = cached_conditionals

    try:
        for audio_chunk_tensor, _ in await asyncio.to_thread(
            chatterbox_tts_model.generate_stream,
            text=text, audio_prompt_path=None, exaggeration=exaggeration,
            temperature=temperature, cfg_weight=cfg, chunk_size=chunk_size,
            context_window=context_window, fade_duration=fade_duration, print_metrics=False
        ):
            audio_data_float = audio_chunk_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            original_sr = chatterbox_tts_model.sr
            if original_sr != TARGET_SAMPLE_RATE:
                audio_data_float = librosa.resample(y=audio_data_float, orig_sr=original_sr, target_sr=TARGET_SAMPLE_RATE)

            # FIX 2: Convert float32 audio to PCM int16 format as required by Simli.
            audio_data_int16 = (audio_data_float * 32767).astype(np.int16)
            
            # Yield the converted int16 bytes
            yield audio_data_int16.tobytes()

    except Exception as e:
        print(f"Error during ChatterboxTTS generate_stream: {e}")
        raise

# --- API Endpoints ---
@router.post("/api/get_simli_session_token", summary="Get Simli Session Token")
async def get_simli_session_token(req_body: SimliSessionRequest):
    # MODIFIED: Check for SIMLI_API_KEY instead of SIMLI_API_KEY_SERVER
    if not SIMLI_API_KEY:
        raise HTTPException(status_code=500, detail="SIMLI_API_KEY is not configured on the backend.")
    
    metadata = {
        "faceId": req_body.face_id, 
        "isJPG": False, 
        "apiKey": SIMLI_API_KEY, # MODIFIED: Use SIMLI_API_KEY
        "syncAudio": True
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(SIMLI_API_SESSION_URL, json=metadata, timeout=10.0)
            response.raise_for_status()
            response_data = response.json()
            return {
                "token": response_data.get("session_token"),
                "websocket_uri": response_data.get("websocket_uri")
            }
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Could not connect to Simli API: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

@router.post("/api/get_tts_audio_for_simli_input", summary="Get TTS Audio Stream for Simli")
async def get_tts_audio_for_simli_input(req_body: TtsRequest):
    return StreamingResponse(
        generate_audio_chunks_from_chatterbox(
            text=req_body.text, 
            voice=req_body.voice_id,
            temperature=req_body.temperature,
            exaggeration=req_body.exaggeration,
            cfg=req_body.cfg,
            chunk_size=15, 
            context_window=250,
            fade_duration=0.03
        ),
        media_type="application/octet-stream"
    )
