# Local Chatterbox-TTS (Unified Voice AI Server)
![image](https://github.com/user-attachments/assets/0006e264-0789-4f3f-ac18-3eebf7dac665)

This repository hosts a FastAPI web application that integrates a Local Large Language Model (LLM), Text-to-Speech (TTS) using Chatterbox, Speech-to-Text (STT) using Whisper, and real-time avatar interaction via Simli.ai. It provides a comprehensive solution for interactive AI conversations, with a user-friendly web interface.

## Features

* **Integrated LLM:** Connects to local LLMs (LM Studio or Ollama) for intelligent conversational responses, with a customizable system prompt.

* **High-Quality TTS (Chatterbox):** Generates natural-sounding speech from text using the Chatterbox TTS model.

* **Real-time STT (Whisper):** Transcribes user speech input to text using the Whisper model, supporting push-to-talk (Spacebar).

* **Sentence-by-Sentence Streaming:** LLM responses and TTS audio stream in real-time, enhancing interactivity.

* **Dynamic Avatar (Simli.ai):** Integrates with Simli.ai for real-time video avatar synchronization with generated speech.

* **Voice Cloning:** Create custom voice embeddings (`.pt` files) from uploaded WAV audio to use with the TTS model.

* **Customizable Background Video:** Set a looping background video from URL or local file for the UI.

* **Adjustable Parameters:** Fine-tune LLM and TTS parameters (temperature, top_p, etc.) via the UI.

* **Conversation History:** Displays an interactive chat history.

* **Detailed Logging:** Provides a real-time log of backend processes directly in the UI.

* **Dynamic Voice Selection:** Automatically loads and makes available `.pt` voice files from the `voices/` directory.

## Hardware Requirements

To run this application locally, particularly for the LLM and TTS models, a robust GPU with significant VRAM is highly recommended. CPU-only operation is possible but will be significantly slower, especially for LLM inference.

* **GPU:** NVIDIA GPU (e.g., RTX 30 series or higher)

* **VRAM:**

    * **Minimum:** 8 GB VRAM (for smaller LLMs like Llama 3 8B and basic Chatterbox TTS).

    * **Recommended:** 12 GB+ VRAM (for better performance, larger LLMs, and smoother operation).

* **CPU:** Modern multi-core CPU.

* **RAM:** 16 GB or more.

Please note: If your system does not meet these specifications, you may encounter out-of-memory errors, extremely slow processing times, or be unable to run the models effectively. Ensure your GPU drivers are up to date.

## Installation and Running the FastAPI App

Follow these steps to set up and run the Local Chatterbox-TTS application on your local machine.

### Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.10 or higher**: You can download it from [python.org](https://www.python.org/) or use Anaconda/Miniconda.

* **Git**: For cloning the repository.

* **FFmpeg**: Essential for audio processing and conversion. Download from [ffmpeg.org](https://ffmpeg.org/download.html) and ensure it's in your system's PATH, or set the `FFMPEG_PATH` environment variable.

### Step 1: Clone the Repository

First, clone this GitHub repository to your local machine:

```bash
git clone https://github.com/jjmlovesgit/local-chatterbox-tts.git
cd local-chatterbox-tts
```

### Step 2: Set Up Python Environment

It's highly recommended to use a virtual environment to manage dependencies and avoid conflicts with other Python projects.

**Using Conda (Recommended):**

```bash
conda create -n chatterbox_env python=3.10  # Or python=3.11/3.12
conda activate chatterbox_env
```

**Using `venv` (Standard Python Virtual Environment):**

```bash
python -m venv chatterbox_env
.\chatterbox_env\Scripts\activate   # On Windows
source chatterbox_env/bin/activate # On macOS/Linux
```

### Step 3: Install Dependencies

With your virtual environment activated, install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

This command will install all the necessary libraries, including `fastapi`, `uvicorn`, `whisper`, `pydub`, `librosa`, and others your application depends on.

**Important Considerations for `torch`, `torchaudio`, and `torchvision`:**

* **CUDA Specificity:** They are compiled for a *very specific* CUDA toolkit version (`cu128` typically implies CUDA 12.x, likely CUDA 12.1 or 12.2). Your local NVIDIA GPU drivers and CUDA toolkit installation must match this version exactly for GPU acceleration to work. Mismatches are a common cause of "CUDA out of memory" or "CUDA not available" errors.

**Recommendation:**
To avoid potential conflicts and ensure a more stable setup, it is strongly recommended that you **remove the `torch`, `torchaudio`, and `torchvision` lines from your `requirements.txt` file.**

Then, **install `torch` first** (along with `torchaudio` and `torchvision`) from the [official PyTorch website](https://pytorch.org/get-started/locally/) **before** running `pip install -r requirements.txt`. Choose the **stable version** that precisely matches your operating system, package manager (pip/conda), and NVIDIA CUDA version.

For example, a typical installation command for a stable CUDA 12.1 build might look like:
`pip install torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121`
(Adjust versions and CUDA version (`cu121`) as per PyTorch's website for your system.)

After installing `torch` manually, you can then try `pip install -r requirements.txt` again. `pip` should skip reinstalling `torch` if a compatible version is already present.

### Step 4: Configure Environment Variables

Create a `.env` file in the root of your project directory (the same level as `main.py`). This file will hold your configuration.

**Example `.env` file:**

```
# Simli.ai API Configuration (Required for Simli avatar functionality)
SIMLI_API_KEY="YOUR_SIMLI_API_KEY_HERE"
SIMLI_FACE_ID="YOUR_SIMLI_FACE_ID_HERE" # e.g., "d1c0199e-310b-44ec-b873-10e3020612ac" (Example ID, replace with yours)

# LLM Backend Configuration
# Choose "LM_STUDIO" or "OLLAMA"
LLM_BACKEND="LM_STUDIO"
LMSTUDIO_SERVER_BASE_URL="[http://127.0.0.1:1234](http://127.0.0.1:1234)" # Default for LM Studio
LMSTUDIO_MODEL="dolphin3.0-llama3.1-8b-abliterated" # Your LM Studio model name

# If using Ollama:
# LLM_BACKEND="OLLAMA"
# OLLAMA_SERVER_BASE_URL="[http://127.0.0.1:11434](http://127.0.0.1:11434)" # Default for Ollama
# OLLAMA_MODEL_NAME="deepseek-r1:latest" # Your Ollama model name

# Custom System Prompt (Optional)
# LLM_SYSTEM_PROMPT="You are a helpful assistant."

# Whisper STT Model (smaller models are faster, larger are more accurate)
WHISPER_MODEL_NAME="base.en" # Options: tiny.en, base.en, small.en, medium.en, large-v2, large-v3

# Chatterbox TTS Parameters (Defaults are set in main.py, override here if needed)
# DEFAULT_CHATTERBOX_TEMP="1.0"
# DEFAULT_CHATTERBOX_EXAGGERATION="0.25"
# DEFAULT_CHATTERBOX_CFG="1.1"
# DEFAULT_CHATTERBOX_SPEED="1.0"

# Audio Processing Parameters
# ENABLE_AUDIO_FILTERING="False" # Set to "True" to enable a low-pass filter for TTS audio
# FILTER_CUTOFF_HZ="10000" # Cutoff for low-pass filter
# FFMPEG_PATH="ffmpeg" # Path to ffmpeg executable if not in system PATH

# Server Host and Port
# HOST="0.0.0.0"
# PORT="8000"
```

**Obtaining Simli.ai API Key and Face ID:**
You will need to sign up at [Simli.ai](https://www.simli.ai/) to obtain your API key and choose/upload a face model to get a Face ID.

**Local LLM Setup:**
* **LM Studio:** Download and install [LM Studio](https://lmstudio.ai/). Download a compatible LLM (e.g., Llama 3 8B) within LM Studio and ensure its local server is running on `http://127.0.0.1:1234`.
* **Ollama:** Download and install [Ollama](https://ollama.com/). Pull a model (e.g., `ollama pull deepseek-coder:latest`) and ensure the Ollama server is running.

### Step 5: Run the FastAPI Application

Once all dependencies are installed and your `.env` is configured, you can start the FastAPI web application:

```bash
uvicorn main:app --reload
```
The `--reload` flag is useful for development as it will automatically restart the server when you make changes to your Python files.

### Step 6: Access the App in Your Browser

After running the command, you will see output in your terminal similar to this:

```
INFO:     Will watch for changes in these directories: ['C:\\Projects\\chatterboxstreaming']
INFO:     Uvicorn running on [http://127.0.0.1:8000](http://127.0.0.1:8000) (Press CTRL+C to quit)
INFO:     Started reloader process [XXXX] using StatReload
...
INFO:     Application startup complete.
```
Open your web browser and navigate to `http://127.0.0.1:8000` (or whatever local URL is displayed) to access the Local Chatterbox-TTS UI.

## How to Use the App

Once the app is loaded in your browser:

* **Chat Tab (Default):**

    * **Your Message:** Type your message.

    * **Mic Button:** Click to start/stop recording your voice for STT.

    * **Spacebar (Push-to-Talk):** Hold spacebar to record, release to transcribe and submit.

    * **Generate Button:** Submit your text input for processing.

    * **Mode Select:** Choose `TTS Only`, `LLM Only`, or `LLM + TTS`.

    * **Voice Select:** Choose from available TTS voices (includes any cloned voices).

    * **Avatar:** Toggle the Simli.ai avatar on/off.

* **Voice Cloning Tab:**

    * Upload a `.wav` audio file (10-30 seconds of clean speech recommended).

    * Provide a unique name for your new voice.

    * Click "Create Voice" to generate a `.pt` embedding file. The new voice will appear in the "Voice" dropdown on the Chat tab.

* **Background Video Tab:**

    * Enter a URL (HTTPS or relative path, e.g., `static/your_video.mp4`) for a background video.

    * Load or toggle play/pause for the video.

    * Adjust video volume.

    * "Choose Local Video File" button allows uploading a video directly from your computer.

* **Advanced Settings Tab:**

    * **TTS Settings:** Adjust `TTS Temp` (randomness), `TTS Exaggeration` (emotional intensity), `TTS CFG (Guidance)` (strictness to exaggeration), and `TTS Speed`. Also includes `Streaming Chunk Size`, `Context Window`, `Fade Duration`, and `Trim Start` for audio fine-tuning.

    * **LLM Settings:** Adjust `LLM Temp`, `LLM Top P`, `LLM Rep Penalty`, and `LLM Top K` for LLM response generation.

* **System Prompt Tab:**

    * Enter a custom system prompt to override the default LLM persona. This allows you to define the AI's character and behavior.

* **Logs Tab:**

    * View real-time backend log output for debugging and monitoring.

## Project Structure

```
.
├── api/
│   ├── simli_router.py                # Simli.ai API interactions
├── chatterbox-streaming/              # Contains Chatterbox TTS model files (if pre-downloaded)
├── dist/
├── src/
├── static/
│   ├── css/
│   │   └── style.css                # Custom CSS for the web UI
│   ├── js/
│   │   ├── script_full.js           # Frontend JavaScript logic
│   │   └── simli_webrtc.js          # Simli.ai WebRTC integration helper
│   ├── mp4/                         # Contains default avatar and background videos
│   │   ├── bot4.mp4
│   │   ├── bot5.mp4
│   │   ├── barista6.mp4
│   │   ├── doc2.mp4
│   │   ├── doc2exp.mp4
│   │   ├── coffeeshop3background.mp4
│   │   └── # ...other media files
│   └── mvp.min.css                  # Minimalist CSS framework
├── temp_stt_audio_files/          # Temporary directory for STT and voice cloning audio
├── voices/                        # Directory for Chatterbox .pt voice embeddings (cloned voices saved here)
│   └── # .pt voice files
├── .env                           # Environment variables for configuration
├── .gitignore                     # Git ignore file
├── Base_Install_test.py           # (Utility)
├── GPU_test.py                    # (Utility)
├── LICENSE                        # Project license
├── llm_router.py                  # Handles LLM integration and streaming responses
├── main.py                        # Main FastAPI application, orchestrates all components
├── notes.md                       # Development notes and external links
├── pyproject.toml                 # Poetry/packaging configuration
├── README.md                      # This file
├── requirements.txt               # Python package dependencies
├── sentence_chunker.py            # Utility for sentence segmentation for TTS
├── text_segmenter.py              # Utility for sentence segmentation (uses pysbd)
└── # ... other standard files
```

## Special Considerations

* **GPU Usage:** Ensure your `torch` installation matches your CUDA version if you have an NVIDIA GPU. The application automatically detects `cuda` if available.

* **FFmpeg:** If `ffmpeg` is not in your system's PATH, the server startup will log an error. You may need to manually set the `FFMPEG_PATH` environment variable in your `.env` file to its executable location.

* **Chatterbox TTS Model:** The first startup may take longer as the Chatterbox TTS model is downloaded and loaded. Voice `.pt` files are also loaded from the `voices/` directory.

## License

This project is licensed under the Apache 2.0 License.
