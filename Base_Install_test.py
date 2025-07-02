import torch
import torchaudio as ta
# This is the correct import. We import the standard class.
from chatterbox.tts import ChatterboxTTS
import os
import time

# --- 1. Setup and Model Loading ---
print("--- Final Chatterbox GPU Test ---")

# Select CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Final check to ensure we're on the GPU
if not torch.cuda.is_available():
    print("FATAL ERROR: PyTorch cannot detect CUDA. Exiting.")
    exit()

# Load the pre-trained model using the standard class name
print("Loading Chatterbox model...")
try:
    model = ChatterboxTTS.from_pretrained(device=device)
    print("Model loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load the model. The installation may still have issues: {e}")
    exit()


# --- 2. Generate Audio Stream with Your Cloned Voice ---
text_to_stream = "After all of that, this audio was finally generated on my own GPU."
audio_prompt_path = "voices/alloy.wav"

print(f"\nStarting generation with voice from: {audio_prompt_path}")

if not os.path.exists(audio_prompt_path):
    print(f"FATAL ERROR: Cannot find the voice file at '{audio_prompt_path}'.")
else:
    try:
        audio_chunks = []
        # The .generate_stream() method exists on the ChatterboxTTS object.
        for audio_chunk, metrics in model.generate_stream(
            text_to_stream, 
            audio_prompt_path=audio_prompt_path
        ):
            print(f"  > Generated chunk {metrics.chunk_count}...")
            audio_chunks.append(audio_chunk)

        # Concatenate all chunks into one final audio tensor
        final_audio = torch.cat(audio_chunks, dim=-1)

        # Save the final output file
        output_filename = "Base_Install_test.wav"
        ta.save(output_filename, final_audio, model.sr)
        
        print("\n--- VICTORY! ---")
        print(f"Successfully generated and saved audio to: {output_filename}")

    except Exception as e:
        print(f"\n--- ERROR DURING GENERATION ---")
        print(f"The model loaded, but failed during audio generation. This is likely a runtime bug.")
        print(f"Error details: {e}")