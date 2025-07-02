# main.py (MODIFIED - Full Integration with Voice Cloning Tab)

# --- Standard Library Imports ---
import logging
import os
import json
import re
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncGenerator
import warnings
import io
import sys
import shutil
import tempfile
import asyncio
import subprocess
import base64
import wave # For Chatterbox audio processing
from pathlib import Path # Added for easier path handling

# --- SUPPRESS FUTUREWARNINGS TO CLEAN UP LOGS ---
warnings.filterwarnings("ignore", category=FutureWarning)
# --- END WARNING SUPPRESSION ---


# --- SET EVENT LOOP POLICY FOR WINDOWS (MUST BE VERY EARLY) ---
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        print("DEBUG: Set Asyncio event loop policy to WindowsProactorEventLoopPolicy.")
    except Exception as e:
        print(f"DEBUG: Could not set WindowsProactorEventLoopPolicy: {e}")
# --- END SET EVENT LOOP POLICY ---

# --- Import for loading .env file ---
from dotenv import load_dotenv
# --- END ---

# --- SETUP LOGGING ---
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s [%(levelname)s] (%(filename)s) %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
# --- END LOGGING SETUP ---

# --- Load environment variables from .env file ---
load_dotenv()
logger.info("Attempted to load environment variables from .env file.")
# --- END ---

# --- Third-Party Imports ---
import numpy as np
import httpx # Still used for LLM router, so keep
import torch
from torch import nn
from fastapi import (
    FastAPI, HTTPException, Request, UploadFile, File, Form, # Added Form for voice name
    status as fastapi_status
)
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# --- NEW IMPORTS FOR CHATTERBOX INTEGRATION ---
try:
    from chatterbox.tts import ChatterboxTTS, Conditionals # Added Conditionals import
    logger.info("ChatterboxTTS and Conditionals imported successfully.")
except ImportError as e:
    logger.error(f"ChatterboxTTS or Conditionals not found. This should not happen in the correct environment. Error: {e}")
    ChatterboxTTS = None
    Conditionals = None # Set to None if not imported
try:
    from pydub import AudioSegment # Added for robust audio file conversion
    logger.info("Pydub imported successfully.")
except ImportError:
    logger.error("Pydub not found. Please install it: pip install pydub")
    AudioSegment = None
try:
    import soundfile as sf
    logger.info("Soundfile imported successfully.")
except ImportError:
    logger.error("Soundfile not found. Please install it: pip install soundfile")
    sf = None
try:
    import librosa
    logger.info("Librosa imported successfully.")
except ImportError:
    logger.error("Librosa not found. Please install it: pip install librosa")
    librosa = None
# --- Audio Filtering Import ---
try:
    from scipy.signal import butter, lfilter
    logger.info("Scipy imported successfully for audio filtering.")
except ImportError:
    logger.warning("Scipy not found. Please install it for audio filtering: pip install scipy")
    butter, lfilter = None, None
# --- END NEW IMPORTS ---


# --- Import the LLM router ---
try:
    from llm_router import router as llm_api_router
    logger.info("LLM router imported successfully.")
except ImportError:
    logger.error("LLM router (llm_router.py) not found. LLM functionality will be disabled.")
    llm_api_router = None
# --- End Import LLM router ---

# --- Import the Simli router ---
try:
    # UPDATED: Changed import path to 'api.simli_router'
    from api.simli_router import router as simli_api_router
    logger.info("Simli router imported successfully.")
except ImportError as e: # Added 'as e' to log the specific error
    logger.error(f"Simli router (api/simli_router.py) not found. Simli functionality will be disabled. Error: {e}")
    simli_api_router = None
# --- End Import Simli router ---


# --- Whisper Import (STT is separate from TTS) ---
try:
    import whisper
    logger.info("Whisper imported successfully for STT.")
except ImportError:
    logger.error("Whisper library not found. Please install it: pip install -U openai-whisper")
    whisper = None
# --- End Imports ---

# --- DIRECT IMPORTS for TextSegmenter and SentenceChunkerForTTS ---
try:
    from text_segmenter import TextSegmenter
    from sentence_chunker import SentenceChunkerForTTS
    logger.info("Successfully imported TextSegmenter and SentenceChunkerForTTS.")
except ImportError:
    logger.warning("TextSegmenter or SentenceChunkerForTTS not found. Text chunking will use basic splitting.")
    TextSegmenter = None
    SentenceChunkerForTTS = None
# --- END DIRECT IMPORTS ---

# --- Constants ---
TARGET_SAMPLE_RATE = int(os.getenv("TARGET_SAMPLE_RATE", "16000"))
DEFAULT_SENTENCES_PER_TTS_CHUNK = int(os.getenv("DEFAULT_SENTENCES_PER_TTS_CHUNK", "1"))
VOICES_PATH = os.getenv("VOICES_PATH", "voices")
os.makedirs(VOICES_PATH, exist_ok=True)
ALL_VOICES: List[str] = [] # MODIFIED: Initialize as empty, will be populated dynamically
DEFAULT_TTS_VOICE = "default" # MODIFIED: Placeholder, will be set from loaded voices
DEFAULT_CHATTERBOX_TEMP = float(os.getenv("DEFAULT_CHATTERBOX_TEMP", "1.0"))
DEFAULT_CHATTERBOX_EXAGGERATION = float(os.getenv("DEFAULT_CHATTERBOX_EXAGGERATION", "0.25"))
DEFAULT_CHATTERBOX_CFG = float(os.getenv("DEFAULT_CHATTERBOX_CFG", "1.1"))
DEFAULT_CHATTERBOX_SPEED = float(os.getenv("DEFAULT_CHATTERBOX_SPEED", "1.0")) # Default speed is 1.0 (normal)
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "turbo")
WHISPER_SAMPLE_RATE = 16000 # Whisper requires 16kHz
TEMP_AUDIO_DIR = "temp_stt_audio_files" # Used for STT and also for voice cloning temp files
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
STREAM_TIMEOUT_SECONDS = 300
SIMLI_AUDIO_INPUT_SR = 16000
SIMLI_AUDIO_INPUT_FORMAT = "s16le"
SIMLI_AUDIO_INPUT_CHANNELS = 1
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")

# --- Simli API Keys (NEW - Re-added for /api/config endpoint) ---
SIMLI_API_KEY = os.getenv("SIMLI_API_KEY")
SIMLI_FACE_ID = os.getenv("SIMLI_FACE_ID")
# --- End Simli API Keys ---

# --- Chatterbox Streaming & Clipping Parameters ---
CHATTERBOX_STREAMING_CHUNK_SIZE = int(os.getenv("CHATTERBOX_STREAMING_CHUNK_SIZE", "15"))
CHATTERBOX_CONTEXT_WINDOW = int(os.getenv("CHATTERBOX_CONTEXT_WINDOW", "250"))
CHATTERBOX_FADE_DURATION = float(os.getenv("CHATTERBOX_FADE_DURATION", "0.05"))
AUDIO_TRIM_SENTENCE_START_MS = int(os.getenv("AUDIO_TRIM_SENTENCE_START_MS", "25"))
# --- End Chatterbox Streaming & Clipping Parameters ---


# --- Audio Filtering Parameters ---
ENABLE_AUDIO_FILTERING = os.getenv("ENABLE_AUDIO_FILTERING", "False").lower() == "true"
FILTER_CUTOFF_HZ = int(os.getenv("FILTER_CUTOFF_HZ", "10000")) # Cut off frequencies above 10kHz
FILTER_ORDER = int(os.getenv("FILTER_ORDER", "5")) # A reasonable filter order
# --- End Constants ---

# --- FFmpeg Check ---
FFMPEG_AVAILABLE = False
def check_ffmpeg():
    global FFMPEG_AVAILABLE
    try:
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        subprocess.run([FFMPEG_PATH, "-version"], capture_output=True, check=True, text=True, startupinfo=startupinfo)
        logger.info(f"ffmpeg found at '{FFMPEG_PATH}'.")
        FFMPEG_AVAILABLE = True
    except Exception as e:
        logger.error(f"'{FFMPEG_PATH}' not found or failed: {e}. Audio conversion for Simli will be disabled if FFMPEG_PATH is not correct or ffmpeg is not installed.")
        FFMPEG_AVAILABLE = False
# --- End FFmpeg Check ---

# --- Device Setup ---
tts_device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Selected device for AI models: '{tts_device}'")
# --- End Device Setup ---

# --- Global Model Variables (Used by simli_router via import) ---
whisper_model: Optional[Any] = None
chatterbox_tts_model: Optional[ChatterboxTTS] = None
PREPARED_VOICE_CONDITIONALS: Dict[str, Any] = {}
# --- END Global Model Variables ---

# --- Helper Functions (Remaining in main.py if used elsewhere) ---
def apply_low_pass_filter(data: np.ndarray, cutoff: int, fs: int, order: int) -> np.ndarray:
    """Applies a low-pass Butterworth filter to audio data."""
    if not butter or not lfilter:
        logger.warning("Scipy not installed, skipping audio filtering.")
        return data
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y.astype(np.float32)

async def convert_audio_with_ffmpeg(
    input_audio_bytes: bytes,
    input_format: str = "f32le",
    input_sample_rate: int = TARGET_SAMPLE_RATE,
    input_channels: int = 1,
    output_format: str = SIMLI_AUDIO_INPUT_FORMAT,
    output_sample_rate: int = SIMLI_AUDIO_INPUT_SR,
    output_channels: int = SIMLI_AUDIO_INPUT_CHANNELS
) -> Optional[bytes]:
    if not FFMPEG_AVAILABLE or not input_audio_bytes:
        logger.error("FFmpeg not available or no input audio bytes for conversion.")
        return None
    if output_format == "s16le": actual_output_codec = "pcm_s16le"
    elif output_format == "f32le": actual_output_codec = "pcm_f32le"
    else: logger.error(f"Unsupported output_format '{output_format}' for direct FFmpeg codec mapping."); return None
    ffmpeg_cmd = [ FFMPEG_PATH, '-f', input_format, '-ar', str(input_sample_rate), '-ac', str(input_channels), '-i', 'pipe:0', '-f', output_format, '-ar', str(output_sample_rate), '-ac', str(output_channels), '-acodec', actual_output_codec, '-hide_banner', '-loglevel', 'error', '-' ]
    try:
        def run_ffmpeg_sync():
            startupinfo_sync = None
            if os.name == 'nt': startupinfo_sync = subprocess.STARTUPINFO(); startupinfo_sync.dwFlags |= subprocess.STARTF_USESHOWWINDOW; startupinfo_sync.wShowWindow = subprocess.SW_HIDE
            process_result = subprocess.run(ffmpeg_cmd, input=input_audio_bytes, capture_output=True, check=False, startupinfo=startupinfo_sync)
            if process_result.returncode != 0: logger.error(f"FFmpeg (one-shot) conversion error (code {process_result.returncode}): {process_result.stderr.decode(errors='ignore').strip()}"); return None
            return process_result.stdout
        return await asyncio.to_thread(run_ffmpeg_sync)
    except Exception as e: logger.exception(f"Exception during FFmpeg (one-shot) audio conversion: {e}"); return None


# --- FastAPI App Setup & Startup ---
app = FastAPI(title="Unified Voice AI Server (with Chatterbox)", description="Provides integrated Chatterbox TTS, STT, LLM and Direct WebRTC Simli functionalities.")

@app.on_event("startup")
async def startup_event():
    """
    Handles the application's startup logic, including loading AI models.
    """
    global whisper_model, chatterbox_tts_model, PREPARED_VOICE_CONDITIONALS, ALL_VOICES, DEFAULT_TTS_VOICE
    check_ffmpeg()
    logger.info("--- Loading AI Models (FastAPI Startup) ---")

    # --- Load Chatterbox TTS Model ---
    if ChatterboxTTS and chatterbox_tts_model is None:
        try:
            logger.info("Loading ChatterboxTTS model...")
            model_instance = ChatterboxTTS.from_pretrained(device=tts_device)
            if model_instance:
                chatterbox_tts_model = model_instance
                logger.info(f"ChatterboxTTS model loaded successfully on device '{tts_device}'.")

                # --- Scan for and pre-prepare .pt voice conditionals ---
                if chatterbox_tts_model and Conditionals: # Ensure Conditionals is imported
                    logger.info(f"Scanning '{VOICES_PATH}' for .pt voice files...")
                    found_pt_files = list(Path(VOICES_PATH).glob("*.pt"))
                    
                    if not found_pt_files:
                        logger.warning(f"No .pt voice files found in '{VOICES_PATH}'. TTS may not work as expected.")
                        ALL_VOICES = [] # No voices found
                        DEFAULT_TTS_VOICE = "none_available"
                    else:
                        for pt_file_path in found_pt_files:
                            voice_name = pt_file_path.stem # Get name without extension
                            try:
                                logger.info(f"  - Loading voice from .pt: {voice_name}")
                                # Conditionals.load already maps to correct device
                                voice_conditionals = Conditionals.load(str(pt_file_path), map_location=tts_device)
                                PREPARED_VOICE_CONDITIONALS[voice_name] = voice_conditionals
                                ALL_VOICES.append(voice_name)
                                logger.info(f"  - Successfully loaded voice: {voice_name}")
                            except Exception as e:
                                logger.exception(f"  - Failed to load voice '{voice_name}' from '{pt_file_path}'.")
                        
                        if ALL_VOICES:
                            # Use the first loaded voice as default if available
                            ALL_VOICES.sort() # Keep voices sorted alphabetically
                            DEFAULT_TTS_VOICE = ALL_VOICES[0]
                            logger.info(f"Dynamically set default TTS voice to: '{DEFAULT_TTS_VOICE}'")
                        else:
                            DEFAULT_TTS_VOICE = "none_available"
                            logger.warning("No voices successfully loaded, default TTS voice set to 'none_available'.")

                    logger.info("Voice conditional loading complete.")

                    # --- Warm-up Run (using a dynamically loaded voice if available) ---
                    if DEFAULT_TTS_VOICE != "none_available" and DEFAULT_TTS_VOICE in PREPARED_VOICE_CONDITIONALS:
                        logger.info(f"Running TTS warm-up inference for voice '{DEFAULT_TTS_VOICE}'...")
                        try:
                            chatterbox_tts_model.conds = PREPARED_VOICE_CONDITIONALS[DEFAULT_TTS_VOICE]
                            async for _ in generate_audio_chunks_from_chatterbox(
                                text="Hello world. This is a warm-up phrase.",
                                voice=DEFAULT_TTS_VOICE,
                                exaggeration=DEFAULT_CHATTERBOX_EXAGGERATION,
                                temperature=DEFAULT_CHATTERBOX_TEMP,
                                cfg=DEFAULT_CHATTERBOX_CFG,
                                chunk_size=CHATTERBOX_STREAMING_CHUNK_SIZE,
                                context_window=CHATTERBOX_CONTEXT_WINDOW,
                                fade_duration=CHATTERBOX_FADE_DURATION
                            ):
                                pass
                            logger.info("TTS warm-up inference complete.")
                        except Exception as e:
                            logger.exception("An error occurred during TTS warm-up.")
                    else:
                        logger.warning("Skipping TTS warm-up: No default voice available or loaded.")
            else:
                logger.error("ChatterboxTTS model instance is None after from_pretrained.")
        except Exception as e:
            logger.exception("ChatterboxTTS model load failed.")

    # --- Load Whisper STT Model (Re-enabled) ---
    if whisper is not None and whisper_model is None:
        try:
            logger.info(f"Loading Whisper STT model ({WHISPER_MODEL_NAME})...")
            whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=tts_device)
            logger.info("Whisper STT model loaded successfully.")
        except Exception as e:
            logger.exception("Whisper load failed.")

    # --- Mount Static Files and Include Routers ---
    # MODIFIED: Serve 'static' as a general directory, and index_full.html
    app.mount("/static", StaticFiles(directory="static", html=True), name="static_assets")

    if llm_api_router:
        app.include_router(llm_api_router, prefix="/api/llm", tags=["LLM Interaction"])
    
    if simli_api_router:
        app.include_router(simli_api_router) # Simli router handles its own prefixes like /api/get_simli_session_token

# --- Endpoint to provide client-side configuration (Simli API keys) ---
@app.get("/api/config", summary="Get client-side configuration", tags=["Configuration"])
async def get_client_config():
    """
    Provides necessary client-side configuration, such as Simli API keys and available voices.
    """
    config_data = {
        "simli_api_key": SIMLI_API_KEY if SIMLI_API_KEY else None,
        "simli_face_id": SIMLI_FACE_ID if SIMLI_FACE_ID else None,
        "available_voices": ALL_VOICES, # Expose available voices
        "default_tts_voice": DEFAULT_TTS_VOICE, # Expose default voice
    }
    if not (SIMLI_API_KEY and SIMLI_FACE_ID):
        config_data["error"] = "Simli API keys are not configured on the server."
        logger.warning("SIMLI_API_KEY or SIMLI_FACE_ID not set in environment variables. Simli functionality will be limited or disabled.")
    
    return JSONResponse(status_code=200, content=config_data)


# --- Voice Cloning Endpoint ---
@app.post("/api/clone_voice", summary="Creates a .pt voice embedding from a WAV file", tags=["Voice Cloning"])
async def clone_voice_endpoint(
    audio_file: UploadFile = File(...),
    voice_name: str = Form(...) # Use Form for simple text fields in FormData
):
    if chatterbox_tts_model is None or Conditionals is None or AudioSegment is None:
        raise HTTPException(status_code=503, detail="Voice cloning service unavailable: Core dependencies (Chatterbox model, Conditionals, or Pydub) not loaded.")

    voices_dir = Path(VOICES_PATH) # Use the existing VOICES_PATH constant
    output_pt_name_clean = Path(voice_name).stem # Sanitize the voice name
    if not output_pt_name_clean:
        raise HTTPException(status_code=400, detail="Invalid voice name. Please provide a valid filename.")

    output_path = voices_dir / f"{output_pt_name_clean}.pt"

    if output_path.exists():
        raise HTTPException(status_code=409, detail=f"Voice file '{output_pt_name_clean}.pt' already exists. Choose a different name or delete the existing file.")

    temp_wav_path = None
    try:
        # Save the uploaded audio to a temporary file
        temp_dir = Path(TEMP_AUDIO_DIR)
        temp_dir.mkdir(exist_ok=True)
        temp_input_filename = f"temp_upload_{uuid.uuid4()}{Path(audio_file.filename).suffix}"
        temp_input_path = temp_dir / temp_input_filename
        
        # Write the uploaded file content to the temp file
        with open(temp_input_path, "wb") as f:
            f.write(await audio_file.read())
        logger.info(f"Uploaded audio saved temporarily to {temp_input_path}")

        # Use pydub to convert the uploaded audio to a standard 16kHz mono WAV for Chatterbox
        temp_wav_path = temp_dir / f"cloned_voice_{uuid.uuid4()}.wav"
        try:
            audio_segment = AudioSegment.from_file(str(temp_input_path))
            # Ensure it's 16kHz, mono, 16-bit PCM for Chatterbox compatibility
            audio_segment = audio_segment.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(1).set_sample_width(2) # 2 bytes = 16-bit
            audio_segment.export(str(temp_wav_path), format="wav")
            logger.info(f"Converted uploaded audio to WAV at {temp_wav_path} for cloning.")
        except Exception as e:
            logger.error(f"Failed to convert uploaded audio to WAV using pydub: {e}")
            raise HTTPException(status_code=400, detail=f"Unsupported audio format for cloning or conversion failed: {e}")

        logger.info(f"⚡ Creating voice embedding for '{output_pt_name_clean}' from '{temp_wav_path}'...")
        # Prepare conditionals (this is the core cloning step)
        await asyncio.to_thread(chatterbox_tts_model.prepare_conditionals, str(temp_wav_path))
        
        # Save the prepared conditionals to the .pt file
        chatterbox_tts_model.conds.save(output_path)
        logger.info(f"✅ Voice '{output_pt_name_clean}.pt' created successfully at {output_path}!")

        # After creation, load this new voice into our in-memory cache
        # and update the ALL_VOICES list so it's immediately available without restarting the server.
        new_conditionals = Conditionals.load(str(output_path), map_location=chatterbox_tts_model.device)
        PREPARED_VOICE_CONDITIONALS[output_pt_name_clean] = new_conditionals # Use clean name as key
        if output_pt_name_clean not in ALL_VOICES:
            ALL_VOICES.append(output_pt_name_clean)
            ALL_VOICES.sort() # Keep it sorted
        
        return JSONResponse(content={"message": f"Voice '{output_pt_name_clean}.pt' created successfully!", "voice_name": output_pt_name_clean, "path": str(output_path)})
    except Exception as e:
        logger.exception(f"Error during voice cloning: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create voice file: {e}")
    finally:
        if temp_wav_path and temp_wav_path.exists():
            try:
                os.remove(temp_wav_path)
                logger.debug(f"Removed temporary WAV file: {temp_wav_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary WAV file {temp_wav_path}: {e}")
        if temp_input_path and temp_input_path.exists(): # Clean up original uploaded file too
            try:
                os.remove(temp_input_path)
                logger.debug(f"Removed temporary input file: {temp_input_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary input file {temp_input_path}: {e}")


# --- Chatterbox Helper Functions for Streaming (Still needed for /api/tts/stream_direct) ---
async def generate_audio_chunks_from_chatterbox(
    text: str,
    voice: str,
    exaggeration: float,
    temperature: float,
    cfg: float,
    chunk_size: int, # ADDED: chunk_size parameter
    context_window: int, # ADDED: context_window parameter
    fade_duration: float # ADDED: fade_duration parameter
) -> AsyncGenerator[bytes, None]:
    """
    Generates audio chunks using ChatterboxTTS's streaming capabilities.
    Yields raw float32 audio bytes for each internal chunk.
    This version is used by /api/tts/stream_direct, yielding float32.
    """
    if chatterbox_tts_model is None:
        logger.error("ChatterboxTTS model is not loaded. Cannot generate streaming audio.")
        return

    cached_conditionals = PREPARED_VOICE_CONDITIONALS.get(voice)
    if not cached_conditionals:
        logger.error(f"Conditionals for voice '{voice}' not found in cache. Cannot proceed.")
        # Attempt to load it just-in-time if not cached, though pre-caching is preferred
        try:
            pt_file_path = Path(VOICES_PATH) / f"{voice}.pt"
            if pt_file_path.exists():
                logger.info(f"Attempting just-in-time load for voice '{voice}' from '{pt_file_path}'.")
                cached_conditionals = Conditionals.load(str(pt_file_path), map_location=chatterbox_tts_model.device)
                PREPARED_VOICE_CONDITIONALS[voice] = cached_conditionals # Cache it for future use
            else:
                raise ValueError(f"Voice '{voice}' .pt file not found at '{pt_file_path}'.")
        except Exception as e:
            logger.error(f"Failed just-in-time loading of voice '{voice}': {e}")
            raise ValueError(f"Voice '{voice}' not pre-prepared, not in cache, and failed just-in-time loading.")

    chatterbox_tts_model.conds = cached_conditionals

    try:
        # The generate_stream now returns a tuple of (tensor, metrics_dict)
        for audio_chunk_tensor, _ in await asyncio.to_thread(
            chatterbox_tts_model.generate_stream,
            text=text,
            audio_prompt_path=None,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg,
            chunk_size=chunk_size, # Use passed parameter
            context_window=context_window, # Use passed parameter
            fade_duration=fade_duration, # Use passed parameter
            print_metrics=False
        ):
            audio_data_float = audio_chunk_tensor.squeeze(0).cpu().numpy().astype(np.float32)

            original_sr = chatterbox_tts_model.sr
            if original_sr != TARGET_SAMPLE_RATE:
                audio_data_float = librosa.resample(y=audio_data_float, orig_sr=original_sr, target_sr=TARGET_SAMPLE_RATE)
            
            yield audio_data_float.tobytes()

    except Exception as e:
        logger.exception(f"Error during ChatterboxTTS generate_stream for text '{text[:50]}...': {e}")
        raise

# --- Pydantic Models (still needed for direct TTS endpoint) ---
class ChatterboxTTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None # MODIFIED: Default to None, will use DEFAULT_TTS_VOICE if not provided
    sentences_per_tts_chunk: Optional[int] = DEFAULT_SENTENCES_PER_TTS_CHUNK
    temperature: Optional[float] = DEFAULT_CHATTERBOX_TEMP
    exaggeration: Optional[float] = DEFAULT_CHATTERBOX_EXAGGERATION
    cfg: Optional[float] = DEFAULT_CHATTERBOX_CFG
    speed: Optional[float] = DEFAULT_CHATTERBOX_SPEED
    chunk_size: Optional[int] = CHATTERBOX_STREAMING_CHUNK_SIZE
    context_window: Optional[int] = CHATTERBOX_CONTEXT_WINDOW
    fade_duration: Optional[float] = CHATTERBOX_FADE_DURATION
    trim_start_ms: Optional[int] = AUDIO_TRIM_SENTENCE_START_MS


class STTResponse(BaseModel):
    text: str
    language: Optional[str] = None
    error: Optional[str] = None

# --- TTS and STT Endpoints ---

@app.post("/api/tts/stream_direct", tags=["TTS_Direct_Playback"], summary="Generates TTS audio stream via Chatterbox for direct playback.")
async def tts_stream_direct_endpoint(request_data: ChatterboxTTSRequest):
    request_id = str(uuid.uuid4())
    # MODIFIED: Use request_data.voice if provided, otherwise fall back to global DEFAULT_TTS_VOICE
    voice_to_use = request_data.voice if request_data.voice and request_data.voice in ALL_VOICES else DEFAULT_TTS_VOICE
    
    if voice_to_use == "none_available" or voice_to_use not in PREPARED_VOICE_CONDITIONALS:
        raise HTTPException(status_code=503, detail="TTS service unavailable: No valid voice selected or available.")

    request_data.voice = voice_to_use # Ensure the request_data has the resolved voice
    logger.info(f"[{request_id}] Direct Stream Request: Text: '{request_data.text[:50]}...' Voice: {voice_to_use} Speed: {request_data.speed}")

    if chatterbox_tts_model is None:
        raise HTTPException(status_code=503, detail="TTS service unavailable: Chatterbox model not loaded.")

    response_headers = {"X-Sample-Rate": str(TARGET_SAMPLE_RATE), "X-Audio-Format": "FLOAT32_PCM"}
    return StreamingResponse(audio_sentence_generator(request_data, request_id), media_type="application/octet-stream", headers=response_headers)


async def audio_sentence_generator(request_data: ChatterboxTTSRequest, request_id: str) -> AsyncGenerator[bytes, None]:
    """
    A true streaming generator. It processes text, then yields audio chunks
    as they are generated by the underlying TTS model, applying effects chunk-by-chunk.
    """
    try:
        text_to_speak = request_data.text
        if not text_to_speak.strip():
            logger.warning(f"[{request_id}] Received empty text. Nothing to synthesize.")
            return

        logger.info(f"[{request_id}] Generating audio for text: '{text_to_speak[:70]}...'")

        is_first_chunk = True
        async for audio_chunk_bytes in generate_audio_chunks_from_chatterbox(
            text=text_to_speak,
            voice=request_data.voice,
            exaggeration=request_data.exaggeration,
            temperature=request_data.temperature,
            cfg=request_data.cfg,
            chunk_size=request_data.chunk_size, # Use passed parameter
            context_window=request_data.context_window, # Use passed parameter
            fade_duration=request_data.fade_duration # Use passed parameter
        ):
            if not audio_chunk_bytes:
                continue

            # Convert bytes to numpy array for processing
            audio_chunk_np = np.frombuffer(audio_chunk_bytes, dtype=np.float32)

            # Apply low-pass filter to each chunk
            if ENABLE_AUDIO_FILTERING:
                audio_chunk_np = apply_low_pass_filter(audio_chunk_np, FILTER_CUTOFF_HZ, TARGET_SAMPLE_RATE, FILTER_ORDER)

            # --- Trim START of the very first chunk of the sentence ---
            if is_first_chunk and request_data.trim_start_ms > 0: # Use request_data.trim_start_ms
                samples_to_trim = int(TARGET_SAMPLE_RATE * (request_data.trim_start_ms / 1000.0))
                if samples_to_trim < len(audio_chunk_np):
                    audio_chunk_np = audio_chunk_np[samples_to_trim:]
                else: # If chunk is smaller than trim amount, send empty
                    audio_chunk_np = np.array([], dtype=np.float32)
                is_first_chunk = False
            
            # --- Apply time-stretching for speed control ---
            speed_rate = request_data.speed
            # A rate of 1.0 is normal speed. > 1.0 is faster, < 1.0 is slower.
            if speed_rate and speed_rate != 1.0 and librosa and audio_chunk_np.size > 0:
                try:
                    # Note: librosa.effects.time_stretch expects a numpy array
                    audio_chunk_np = librosa.effects.time_stretch(y=audio_chunk_np, rate=speed_rate)
                except Exception as e:
                    logger.warning(f"[{request_id}] Could not time-stretch audio chunk: {e}")


            # Yield the processed chunk if it has any data
            if audio_chunk_np.size > 0:
                yield audio_chunk_np.tobytes()

    except Exception as e:
        logger.exception(f"[{request_id}] Error in audio_sentence_generator: {e}")


@app.post("/api/stt/transcribe", response_model=STTResponse, summary="Transcribe audio to text", tags=["STT"])
async def stt_transcribe_endpoint(audio_file: UploadFile = File(...)):
    if whisper is None or whisper_model is None:
        return STTResponse(text="", error="STT service unavailable: Whisper model not loaded.")
    
    tmp_audio_file_path = None
    try:
        os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
        _, file_extension = os.path.splitext(audio_file.filename or ".wav")
        file_extension = file_extension or ".wav"

        with tempfile.NamedTemporaryFile(delete=False, dir=TEMP_AUDIO_DIR, suffix=file_extension) as tmp_file:
            shutil.copyfileobj(audio_file.file, tmp_file)
            tmp_audio_file_path = tmp_file.name
        
        # --- FIX: Sanitize audio input before passing to Whisper ---
        # Load the audio file using librosa, which handles many formats.
        # This resamples to 16kHz (required by Whisper) and converts to a float32 numpy array.
        logger.info(f"STT: Loading and resampling '{tmp_audio_file_path}' to {WHISPER_SAMPLE_RATE}Hz.")
        audio_np, _ = librosa.load(tmp_audio_file_path, sr=WHISPER_SAMPLE_RATE, mono=True)
        logger.info("STT: Resampling complete. Transcribing with Whisper...")

        # Pass the sanitized numpy array directly to the model
        result = whisper_model.transcribe(audio_np, fp16=(tts_device == "cuda"))
        
        transcribed_text = result["text"].strip() if isinstance(result, dict) and "text" in result else ""
        detected_language = result.get("language", "unknown") if isinstance(result, dict) else "unknown"
        logger.info(f"STT: Transcription successful. Language: {detected_language}")
        return STTResponse(text=transcribed_text, language=detected_language)

    except Exception as e:
        # Check for CUDA-specific errors to provide a more helpful message
        error_str = str(e)
        if "CUDA" in error_str:
            logger.exception("STT: A CUDA-related error occurred during transcription. This might be due to an issue with the GPU or the audio format.")
            return STTResponse(text="", error=f"GPU error during transcription: {error_str}")
        
        logger.exception("STT: Error during transcription")
        return STTResponse(text="", error=error_str)
        
    finally:
        if tmp_audio_file_path and os.path.exists(tmp_audio_file_path):
            try:
                os.remove(tmp_audio_file_path)
            except Exception:
                pass


# --- Root and Static File Serving ---
@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/", include_in_schema=False)
async def read_root_html_static(request: Request):
    # MODIFIED: Serve index_full.html instead of index.html
    static_index_path = os.path.join("static", "index_full.html")
    if os.path.exists(static_index_path):
        return FileResponse(static_index_path)
    logger.error("Could not find index_full.html in 'static/' directory.")
    raise HTTPException(status_code=404, detail="index_full.html not found")

# --- Uvicorn Server ---
if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            current_policy = asyncio.get_event_loop_policy()
            if not isinstance(current_policy, asyncio.WindowsProactorEventLoopPolicy):
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except Exception as e:
            logger.warning(f"Could not set WindowsProactorEventLoopPolicy in __main__: {e}")

    if chatterbox_tts_model is None:
        logger.critical("ChatterboxTTS Model failed to load. TTS will be unavailable.")
    if whisper_model is None:
        logger.warning("Whisper STT Model not loaded.")
    if not FFMPEG_AVAILABLE:
        logger.warning("FFmpeg not found.")
    if AudioSegment is None:
        logger.critical("Pydub not loaded. Audio conversion for voice cloning might fail.")

    port_to_use = int(os.getenv("PORT", "8000"))
    host_to_use = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting Unified FastAPI Server with Uvicorn on http://{host_to_use}:{port_to_use} ...")
    uvicorn.run("main:app", host=host_to_use, port=port_to_use, reload=True)
    logger.info("FastAPI Server Stopped.")
