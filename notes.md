notes:
https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4
https://examplefiles.org/files/video/mp4-example-video-download-full-hd-1920x1080.mp4
https://examplefiles.org/example-video-files/sample-mp4-files

http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4
http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4
http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4
http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4
http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4
http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4
http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4
http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4
http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/SubaruOutbackOnStreetAndDirt.mp4
http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4
http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/WeAreGoingOnBullrun.mp4
http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/WhatCarCanYouGetForAGrand.mp4

https://github.com/joshuatz/video-test-file-links?tab=readme-ov-file

    (chatterbox_env) C:\Projects\chatterboxstreaming> uvicorn main:app --reload
INFO:     Will watch for changes in these directories: ['C:\\Projects\\chatterboxstreaming']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [22104] using StatReload
DEBUG: Set Asyncio event loop policy to WindowsProactorEventLoopPolicy.
2025-06-29 08:17:53 [INFO] (main.py) Attempted to load environment variables from .env file.
2025-06-29 08:17:57 [INFO] (main.py) ChatterboxTTS imported successfully.
2025-06-29 08:17:57 [INFO] (main.py) Pydub imported successfully.
2025-06-29 08:17:57 [INFO] (main.py) Soundfile imported successfully.
2025-06-29 08:17:57 [INFO] (main.py) Librosa imported successfully.
2025-06-29 08:17:57 [INFO] (main.py) Scipy imported successfully for audio filtering.
2025-06-29 08:17:57 [INFO] (main.py) LLM router imported successfully.
2025-06-29 08:17:57 [INFO] (main.py) Simli router imported successfully.
2025-06-29 08:17:57 [INFO] (main.py) Whisper imported successfully for STT.
2025-06-29 08:17:57 [INFO] (main.py) Successfully imported TextSegmenter and SentenceChunkerForTTS.
2025-06-29 08:17:57 [INFO] (main.py) Selected device for AI models: 'cuda'
INFO:     Started server process [14100]
INFO:     Waiting for application startup.
2025-06-29 08:17:57 [INFO] (main.py) ffmpeg found at 'ffmpeg'.
2025-06-29 08:17:57 [INFO] (main.py) --- Loading AI Models (FastAPI Startup) ---
2025-06-29 08:17:57 [INFO] (main.py) Loading ChatterboxTTS model...
2025-06-29 08:18:01 [INFO] (flow.py) input frame rate=25
2025-06-29 08:18:02 [INFO] (main.py) ChatterboxTTS model loaded successfully on device 'cuda'.
2025-06-29 08:18:02 [INFO] (main.py) Pre-preparing voice conditionals...
2025-06-29 08:18:02 [INFO] (main.py)   - Preparing voice: alloy
C:\Users\Jim\miniconda3\envs\chatterbox_env\lib\site-packages\chatterbox\tts.py:195: UserWarning: PySoundFile failed. Trying audioread instead.
  s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
2025-06-29 08:18:03 [INFO] (main.py)   - Successfully prepared voice: alloy
2025-06-29 08:18:03 [INFO] (main.py) Voice conditional preparation complete.
2025-06-29 08:18:03 [INFO] (main.py) Running TTS warm-up inference for voice 'alloy'...
LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.
We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class (https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)
2025-06-29 08:18:06 [INFO] (main.py) TTS warm-up inference complete.
2025-06-29 08:18:06 [INFO] (main.py) Loading Whisper STT model (base.en)...
2025-06-29 08:18:07 [INFO] (main.py) Whisper STT model loaded successfully.
INFO:     Application startup complete.
INFO:     127.0.0.1:53045 - "GET / HTTP/1.1" 200 OK
INFO:     127.0.0.1:53045 - "GET /static/script.js HTTP/1.1" 304 Not Modified
INFO:     127.0.0.1:53046 - "GET / HTTP/1.1" 200 OK
INFO:     127.0.0.1:53046 - "GET /static/simli_webrtc.js HTTP/1.1" 200 OK
INFO:     127.0.0.1:53048 - "GET /static/script.js HTTP/1.1" 200 OK
INFO:     127.0.0.1:53049 - "GET /static/medbackground1.png HTTP/1.1" 200 OK
INFO:     127.0.0.1:53048 - "GET /api/config HTTP/1.1" 200 OK
INFO:     127.0.0.1:53048 - "GET /static/favicon.ico HTTP/1.1" 200 OK
INFO:     127.0.0.1:53050 - "GET / HTTP/1.1" 200 OK
INFO:     127.0.0.1:53050 - "GET /static/simli_webrtc.js HTTP/1.1" 200 OK
INFO:     127.0.0.1:53051 - "GET /static/script.js HTTP/1.1" 200 OK
INFO:     127.0.0.1:53052 - "GET /static/medbackground1.png HTTP/1.1" 200 OK
INFO:     127.0.0.1:53052 - "GET /api/config HTTP/1.1" 200 OK
INFO:     127.0.0.1:53052 - "GET /static/favicon.ico HTTP/1.1" 200 OK
2025-06-29 08:19:03 [INFO] (main.py) STT: Loading and resampling 'C:\Projects\chatterboxstreaming\temp_stt_audio_files\tmpbsr4_36f.webm' to 16000Hz.
C:\Projects\chatterboxstreaming\main.py:524: UserWarning: PySoundFile failed. Trying audioread instead.
  audio_np, _ = librosa.load(tmp_audio_file_path, sr=WHISPER_SAMPLE_RATE, mono=True)
2025-06-29 08:19:03 [INFO] (main.py) STT: Resampling complete. Transcribing with Whisper...
2025-06-29 08:19:04 [INFO] (main.py) STT: Transcription successful. Language: en
INFO:     127.0.0.1:53054 - "POST /api/stt/transcribe HTTP/1.1" 200 OK
INFO:     127.0.0.1:53064 - "GET /api/config HTTP/1.1" 200 OK
INFO:     127.0.0.1:53065 - "POST /api/llm/chat/stream HTTP/1.1" 200 OK
2025-06-29 08:19:27 [INFO] (llm_router.py) [c04f9411-96d6-4764-ab40-2ab2d0350a1c] LLM Router: Initiating sentence stream. Model: medgemma-4b-it.
2025-06-29 08:19:27 [INFO] (llm_router.py) [c04f9411-96d6-4764-ab40-2ab2d0350a1c] LLM Router: API Stream connected after 0.002s.
INFO:     127.0.0.1:53069 - "GET /static/doc2exp.mp4 HTTP/1.1" 206 Partial Content
2025-06-29 08:19:29 [INFO] (llm_router.py) [c04f9411-96d6-4764-ab40-2ab2d0350a1c] LLM Router: Stream processing finished.
2025-06-29 08:19:29 [INFO] (main.py) [c6028a2b-c692-4342-8e05-e3c473da0653] Direct Stream Request: Text: 'Hello, how can I help you today?...' Voice: alloy Speed: 1.0
INFO:     127.0.0.1:53065 - "POST /api/tts/stream_direct HTTP/1.1" 200 OK
2025-06-29 08:19:29 [INFO] (main.py) [c6028a2b-c692-4342-8e05-e3c473da0653] Generating audio for text: 'Hello, how can I help you today?...'
INFO:     127.0.0.1:53070 - "POST /api/llm/chat/stream HTTP/1.1" 200 OK
2025-06-29 08:19:47 [INFO] (llm_router.py) [9ae1e95f-e712-4bc0-b4b8-e8a28d5a5bef] LLM Router: Initiating sentence stream. Model: medgemma-4b-it.
2025-06-29 08:19:47 [INFO] (llm_router.py) [9ae1e95f-e712-4bc0-b4b8-e8a28d5a5bef] LLM Router: API Stream connected after 0.003s.
2025-06-29 08:19:48 [INFO] (llm_router.py) [9ae1e95f-e712-4bc0-b4b8-e8a28d5a5bef] LLM Router: Stream processing finished.
2025-06-29 08:19:48 [INFO] (main.py) [10b6725a-144e-43bc-a803-1f2c4e23320e] Direct Stream Request: Text: 'Hi there, what brings you here to me?...' Voice: alloy Speed: 1.2
INFO:     127.0.0.1:53070 - "POST /api/tts/stream_direct HTTP/1.1" 200 OK
2025-06-29 08:19:48 [INFO] (main.py) [10b6725a-144e-43bc-a803-1f2c4e23320e] Generating audio for text: 'Hi there, what brings you here to me?...'
INFO:     127.0.0.1:53072 - "POST /api/llm/chat/stream HTTP/1.1" 200 OK
2025-06-29 08:20:08 [INFO] (llm_router.py) [6ffa2136-ac28-4c8e-a92d-f820d2f649a1] LLM Router: Initiating sentence stream. Model: medgemma-4b-it.
2025-06-29 08:20:08 [INFO] (llm_router.py) [6ffa2136-ac28-4c8e-a92d-f820d2f649a1] LLM Router: API Stream connected after 0.012s.
2025-06-29 08:20:09 [INFO] (llm_router.py) [6ffa2136-ac28-4c8e-a92d-f820d2f649a1] LLM Router: Stream processing finished.
2025-06-29 08:20:09 [INFO] (main.py) [617b9f7f-f7c7-4822-852f-fde375aaee57] Direct Stream Request: Text: 'What symptoms are you experiencing right now?...' Voice: alloy Speed: 0.8
INFO:     127.0.0.1:53072 - "POST /api/tts/stream_direct HTTP/1.1" 200 OK
2025-06-29 08:20:09 [INFO] (main.py) [617b9f7f-f7c7-4822-852f-fde375aaee57] Generating audio for text: 'What symptoms are you experiencing right now?...'
