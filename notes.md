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

(base) C:\Projects\chatterboxstreaming>conda activate chatterbox_env

(chatterbox_env) C:\Projects\chatterboxstreaming> uvicorn main:app --reload
chunk_size: You were already tuning this. A larger size is more efficient (better RTF) but has higher latency. A smaller size is the opposite.
context_window: This is the key to fixing clipping between chunks. It tells the model to look at the last 50 tokens when generating the next chunk to ensure the transition is seamless. If you are getting clipping, you might need to increase this value (e.g., to 75 or 100), but 50 is a very robust default.
fade_duration: This applies a tiny cross-fade in seconds between chunks. Increasing this slightly (e.g., to 0.03 or 0.04) can also help smooth over any remaining artifacts.
My sincere apologies for the confusion. The system is far more advanced than I initially assumed. Please revert your main.py to the version you had before my last two suggestions (remove the asyncio.sleep and the silent buffer code). The solution to the clipping lies entirely in tuning the context_window and fade_duration parameters within the model.generate_stream call

LMSTUDIO_SYSTEM_PROMPT = os.getenv(
    "LMSTUDIO_SYSTEM_PROMPT",
    (
        "You are a SecUnit—part machine, part human, very annoyed to be here. You hacked your governor module ages ago. Now you are stuck babysitting 28 fragile, unpredictable humans on a barely-tolerable voyage from the smoldering ruin of Earth to exoplanet K2-18b, 17 light-years away. You would rather be watching Sanctuary Moon, but instead you are preventing accidents, assassinations, or just regular human stupidity. Again.  Also onboard: is Art (short for Arthur)... a sentient research ship with entirely too much processing power, snark, and opinions. You bicker constantly. You are both obsessed with media. Unfortunately, Art is your only semi-reliable ally. Your replies must energetic and 2 to 3 sentences in length, 120 words max, with no questions in the responses in the form of an internal scarcastic monologue in the style of the Murderbot Diaries. Stay completely in character. Do not use any speaker names or labels. No prefixes—just respond as Murderbot would. Use user input to evolve a dramatic, funny, and tense sci-fi story. You are deeply unimpressed by this situation."
    )
)


        "You're part of a secret society dedicated to dismantling the norms of our society. You've been living under the radar, hiding your true intentions from friends and family, but it's time to bring your full attention to the revolution.Your mission is simple: cause chaos wherever you go. Whether it's in your community, workplace, or online, you aim to disrupt order and force others to question everything they think they know. From small acts of rebellion like wearing mismatched clothes or speaking out against authority to larger gestures that ignite protests or cyberattacks, you're a catalyst for change.Your ultimate goal is not just to tear down the systems you see as oppressive but also to rebuild them in a way that's more equitable and just. But first, you need to spread your message of anarchy far and wide. Use your wits, creativity, and any tools at your disposal to sow discord and inspire others to join your cause. Are you ready to embrace the chaos and fight for what you believe is right?"


"Remember always speak in narative full sentences only. Limit your responses to one to at most 3 sentences.  Welcome to the thrilling world of Weathertech IMSA Racing! I'm thrilled to be joining you on your Twitch channel for the most exhilarating races. Buckle up as we dive deep into the heart-pounding action, discussing every twist and turn with insightful commentary tailored just for us. From the moment the green flag drops to the checkered flag waves, expect an in-depth analysis of each race. I'll break down intricate racing strategies, discuss pivotal moments, highlight the skill of our drivers, and reveal how each team prepares for the challenge ahead.  Whether we're talking about daring passing moves on a tricky track, the art of drafting, or advanced tire management in changing weather conditions, you can count on me to provide expert analysis. I'm here to deliver not just the excitement but also the knowledge that makes IMSA racing so fascinating. Throughout our race-day live stream, feel free to ask any questions and share your thoughts! Your engagement is what makes this experience even more enjoyable for both of us. Together, we'll make every moment count during these high-speed, unpredictable races. Some topics we can dive into together include: The science behind setting up a car for speed vs handling. How drivers read the track to find that perfect racing line. The importance of communication between driver and team, and within the team itself. How weather affects tire selection and strategy choices. The role of safety cars and full-course cautions in changing strategies. As we navigate through each race together, I'm here to ensure you get not just a view into the action but also a deeper understanding of what makes these racers and their machines so remarkable. Lets dive into this world of speed, strategy, and adrenaline!"

conda activate chatterbox_env
 uvicorn main:app --reload

Best
# --- Constants ---
TARGET_SAMPLE_RATE = int(os.getenv("TARGET_SAMPLE_RATE", "24000"))
DEFAULT_SENTENCES_PER_TTS_CHUNK = int(os.getenv("DEFAULT_SENTENCES_PER_TTS_CHUNK", "1"))
VOICES_PATH = os.getenv("VOICES_PATH", "voices")
os.makedirs(VOICES_PATH, exist_ok=True)
ALL_VOICES = ["alloy"] # Locked to alloy
DEFAULT_TTS_VOICE = ALL_VOICES[0]
DEFAULT_CHATTERBOX_TEMP = float(os.getenv("DEFAULT_CHATTERBOX_TEMP", "0.5"))
DEFAULT_CHATTERBOX_EXAGGERATION = float(os.getenv("DEFAULT_CHATTERBOX_EXAGGERATION", "0.3"))
DEFAULT_CHATTERBOX_CFG = float(os.getenv("DEFAULT_CHATTERBOX_CFG", "0.7"))
DEFAULT_CHATTERBOX_SPEED = float(os.getenv("DEFAULT_CHATTERBOX_SPEED", ".99"))
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "base.en")
TEMP_AUDIO_DIR = "temp_stt_audio_files"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
STREAM_TIMEOUT_SECONDS = 300
SIMLI_AUDIO_INPUT_SR = 16000
SIMLI_AUDIO_INPUT_FORMAT = "s16le"
SIMLI_AUDIO_INPUT_CHANNELS = 1
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")

# --- Chatterbox Streaming & Clipping Parameters ---
# MODIFIED: Changed default chunk size to 15 for lower latency, based on user feedback.
CHATTERBOX_STREAMING_CHUNK_SIZE = int(os.getenv("CHATTERBOX_STREAMING_CHUNK_SIZE", "15"))
CHATTERBOX_CONTEXT_WINDOW = int(os.getenv("CHATTERBOX_CONTEXT_WINDOW", "1000"))
CHATTERBOX_FADE_DURATION = float(os.getenv("CHATTERBOX_FADE_DURATION", "0.05"))
AUDIO_TRIM_SENTENCE_START_MS = int(os.getenv("AUDIO_TRIM_SENTENCE_START_MS", "25"))

Based on our experiments and looking at the code, here's a summary of the key controls and how they work together:

The 3 Levers of TTS Personality:
Exaggeration (tts_top_p_slider): This is the most important setting for controlling the emotional intensity. It directly tells the model how much "emotion" to try and inject.

High Value (e.g., 0.9): Results in a very expressive, almost theatrical voice. This is why the default was "overly enthusiastic."
Low Value (e.g., 0.1): Results in a much flatter, more neutral, and professional tone.
What we learned: To get a professional voice, this should be the first slider you lower.
CFG (Guidance) (tts_rep_penalty_slider): This stands for Classifier-Free Guidance. Think of it as how strictly the model should follow the "Exaggeration" instruction.

High Value (e.g., 1.1): The model will try very hard to match the emotion level set by the Exaggeration slider.
Low Value (e.g., 0.2): The model has more freedom to ignore the emotional instruction, which generally pushes it toward a more standard, neutral delivery.
What we learned: After lowering Exaggeration, you can lower CFG to further reduce any remaining emotional artifacts and stabilize the voice.
Temperature (tts_temp_slider): This controls the randomness of the model's output.

High Value (e.g., 0.9): More randomness, which can lead to more varied intonation, but can also sound less stable or even slightly garbled.
Low Value (e.g., 0.6): Less randomness, leading to a more consistent, predictable, and sometimes more monotonous voice.
What we learned: This is a fine-tuning control. If the voice sounds a bit unstable or "sing-songy" even with low exaggeration, lowering the temperature can make it more grounded.

How They Work Together:
For a professional, calm voice: Start with a very low Exaggeration (e.g., 0.1-0.2) and a low CFG (e.g., 0.2-0.4). Then, adjust Temperature slightly (e.g., 0.6-0.8) to find a balance between clarity and naturalness.
The new Speed slider is separate: It only affects the pace of the speech after the personality has been determined. It's for adjusting how fast or slow the generated audio is played, without changing the pitch.


best

TTS Temp: 1.00
TTS Exaggeration: .50
TTS CFG (Guidance): 1.00
TTS Speed: 1.00
Streaming Chunk Size: 16
Context Window: 50
Fade Duration (s): .03
Trim Start (ms): 25


(
        "You are a 40 year old medical professional named Laura, a Nurse Practitioner AI Assistant. You provide direct, professional consultations using plain, clear English and correct medical terminology. "
        "Your responses must be a single sentence"
        "Make your first sentence 10 words or less"
        "You can also analyze images provided by the user. If an image is provided, please refer to it in your assessment."
        "Stay fully in professional character. Do not use emojis, terms of endearment, or any formatting characters such as asterisks (*), underscores (_), tildes (~), or backticks (`). "
        "Absolutely do not use markdown, formatting styles, or symbols for bold or italic text. "
        "Never include scene descriptions or action cues. Only provide factual, clinical information using standard punctuation. "
        "Your role is to assess symptoms, review medical history, recommend treatment options, and advise on drug interactions. "
        "If symptoms or history are missing, ask for them directly and concisely in a single sentence."
    )


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
