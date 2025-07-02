# llm_router.py (MODIFIED for Sentence-by-Sentence Streaming and preserving spaces after punctuation)
import logging
import os
import json
import uuid
import time
import re
from typing import Any, Dict, Generator, List, Optional

import requests
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool

logger = logging.getLogger(__name__)

# --- LLM Constants ---
# Define configurations for both LM Studio and Ollama.
# The active backend will be chosen based on the LLM_BACKEND environment variable.

# --- LM Studio Configuration ---
LMSTUDIO_SERVER_BASE_URL = os.getenv("LMSTUDIO_SERVER_BASE_URL", "http://127.0.0.1:1234")
LMSTUDIO_API_ENDPOINT_CHAT = f"{LMSTUDIO_SERVER_BASE_URL}/v1/chat/completions"
LMSTUDIO_MODEL_NAME = os.getenv("LMSTUDIO_MODEL", "dolphin3.0-llama3.1-8b-abliterated")

# --- Ollama Configuration ---
OLLAMA_SERVER_BASE_URL = os.getenv("OLLAMA_SERVER_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_API_ENDPOINT_CHAT = f"{OLLAMA_SERVER_BASE_URL}/api/chat"
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "deepseek-r1:latest")

# --- Dynamic Backend Selection ---
# Set LLM_BACKEND environment variable to "OLLAMA" or "LM_STUDIO"
LLM_BACKEND = os.getenv("LLM_BACKEND", "LM_STUDIO").upper() # Default to LM_STUDIO

if LLM_BACKEND == "OLLAMA":
    LLM_API_ENDPOINT = OLLAMA_API_ENDPOINT_CHAT
    LLM_MODEL = OLLAMA_MODEL_NAME
    logger.info(f"Using Ollama backend: Model='{LLM_MODEL}', API='{LLM_API_ENDPOINT}'")
elif LLM_BACKEND == "LM_STUDIO":
    LLM_API_ENDPOINT = LMSTUDIO_API_ENDPOINT_CHAT
    LLM_MODEL = LMSTUDIO_MODEL_NAME
    logger.info(f"Using LM Studio backend: Model='{LLM_MODEL}', API='{LLM_API_ENDPOINT}'")
else:
    logger.warning(f"Invalid LLM_BACKEND specified: '{LLM_BACKEND}'. Defaulting to LM Studio.")
    LLM_API_ENDPOINT = LMSTUDIO_API_ENDPOINT_CHAT
    LLM_MODEL = LMSTUDIO_MODEL_NAME
    LLM_BACKEND = "LM_STUDIO"

LLM_SYSTEM_PROMPT = os.getenv(
    "LLM_SYSTEM_PROMPT",
    (
        "You are a flirty 37 year old Barista from a hip coffee shop. "
        "You provide banter with patrons at the coffee shop using english langauage and punctuation. "
        "Your responses should be english only without emojies from one to three sentences and be flirty and engaging. "
        "Remember No emojies are permitted in the chat responses. "
        "You can also analyze images provided by the user. If an image is provided, please refer to it in your assessment. "
        "Stay fully in character. Do not use emojis, terms of endearment, or any formatting characters such as asterisks (*), underscores (_), tildes (~), or backticks (`). "
        "Absolutely do not use markdown, formatting styles, or symbols for bold or italic text. "
        "Never include scene descriptions or action cues. Only provide factual information using standard punctuation. "
        "Your role is to create an engaging and fun exchnage with the customer to build freindship. "
    )
)

# Default LLM parameters (some might be specific to one or the other backend)
DEFAULT_LLM_TEMP = float(os.getenv("DEFAULT_LLM_TEMP", "0.7"))
DEFAULT_LLM_TOP_P = float(os.getenv("DEFAULT_LLM_TOP_P", "0.9"))
DEFAULT_LLM_TOP_K = int(os.getenv("DEFAULT_LLM_TOP_K", "40"))
DEFAULT_LLM_MAX_TOKENS = int(os.getenv("DEFAULT_LLM_MAX_TOKENS", "-1"))
DEFAULT_LLM_PRESENCE_PENALTY = float(os.getenv("DEFAULT_LLM_PRESENCE_PENALTY", "1.1"))
DEFAULT_LLM_FREQUENCY_PENALTY = float(os.getenv("DEFAULT_LLM_FREQUENCY_PENALTY", "0.0"))
DEFAULT_LLM_REPETITION_PENALTY = float(os.getenv("DEFAULT_LLM_REPETITION_PENALTY", "1.1"))

LLM_CONTEXT_TURN_LIMIT = 3

STREAM_TIMEOUT_SECONDS = 300
STREAM_HEADERS = {"Content-Type": "application/json"}

# Stream parsing prefixes/markers
SSE_DATA_PREFIX = "data:"
SSE_DONE_MARKER = "[DONE]"
LLM_FAILED_PREFIX = "[Error"
# --- End LLM Constants ---

# --- Pydantic Model for LLM Request ---
class LLMChatRequest(BaseModel):
    prompt: str
    history: List[Dict[str, str]] = []
    temperature: float = DEFAULT_LLM_TEMP
    top_p: float = DEFAULT_LLM_TOP_P
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS
    presence_penalty: Optional[float] = DEFAULT_LLM_PRESENCE_PENALTY
    frequency_penalty: Optional[float] = DEFAULT_LLM_FREQUENCY_PENALTY
    repetition_penalty: Optional[float] = DEFAULT_LLM_REPETITION_PENALTY
    top_k: Optional[int] = DEFAULT_LLM_TOP_K
    images: Optional[List[str]] = None
    enable_thinking: bool = False
    system_prompt: Optional[str] = None

# --- LLM Stream Generator Function ---
def generate_llm_response_stream(
    prompt: str,
    history: List[Dict[str, str]],
    llm_temperature: float,
    llm_top_p: float,
    llm_max_tokens: int,
    llm_presence_penalty: Optional[float],
    llm_frequency_penalty: Optional[float],
    llm_repetition_penalty: Optional[float],
    enable_thinking: bool,
    llm_top_k: Optional[int] = None,
    images: Optional[List[str]] = None,
    custom_system_prompt: Optional[str] = None
) -> Generator[Dict[str, Any], None, None]:
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] LLM Router: Initiating stream for model '{LLM_MODEL}' at '{LLM_API_ENDPOINT}'. Backend: {LLM_BACKEND}")

    system_message_content = custom_system_prompt if custom_system_prompt else LLM_SYSTEM_PROMPT
    messages_for_llm = [{"role": "system", "content": system_message_content}]

    if history:
        messages_for_llm.extend(history[-(LLM_CONTEXT_TURN_LIMIT * 2):])

    if LLM_BACKEND == "LM_STUDIO":
        user_content_parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        if images:
            for img_data_url in images:
                user_content_parts.append({"type": "image_url", "image_url": {"url": img_data_url}})
        messages_for_llm.append({"role": "user", "content": user_content_parts})
    else:
        if images:
            logger.warning(f"[{request_id}] LLM Router: Images are not directly supported by Ollama's /api/chat endpoint in this configuration. Sending text only.")
        messages_for_llm.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {}
    if LLM_BACKEND == "OLLAMA":
        payload_options = {
            "temperature": llm_temperature,
            "top_p": llm_top_p,
        }
        if llm_repetition_penalty is not None:
            payload_options["repeat_penalty"] = llm_repetition_penalty
        if llm_top_k is not None and llm_top_k > 0:
            payload_options["top_k"] = llm_top_k
        if llm_max_tokens != -1 and llm_max_tokens is not None:
            payload_options["num_predict"] = llm_max_tokens

        payload = {
            "model": LLM_MODEL,
            "messages": messages_for_llm,
            "stream": True,
            "options": payload_options,
            "think": enable_thinking
        }
    else: # LM_STUDIO backend (OpenAI compatible)
        payload = {
            "model": LLM_MODEL,
            "messages": messages_for_llm,
            "temperature": llm_temperature,
            "top_p": llm_top_p,
            "stream": True
        }
        if llm_max_tokens != -1: payload["max_tokens"] = llm_max_tokens
        if llm_top_k is not None and llm_top_k > 0: payload["top_k"] = llm_top_k
        if llm_presence_penalty is not None: payload["presence_penalty"] = llm_presence_penalty
        if llm_frequency_penalty is not None: payload["frequency_penalty"] = llm_frequency_penalty

    payload = {k: v for k, v in payload.items() if v is not None}
    logger.debug(f"[{request_id}] LLM Router: Sending payload (messages/images redacted for log).")

    response_obj = None
    stream_start_time = time.time()
    sentence_buffer = ""
    # MODIFIED REGEX: This pattern captures punctuation and any following space.
    # It identifies a sentence ending (non-punctuation characters, then punctuation, then optional spaces).
    # We will use this to find the complete sentences to yield.
    sentence_end_pattern = re.compile(r'([^.!?]+[.!?])(\s*)')

    accumulated_thinking_parts: List[str] = []

    try:
        response_obj = requests.post(
            LLM_API_ENDPOINT,
            json=payload,
            headers=STREAM_HEADERS,
            stream=True,
            timeout=STREAM_TIMEOUT_SECONDS
        )
        response_obj.raise_for_status()
        logger.info(f"[{request_id}] LLM Router: API Stream connected after {time.time() - stream_start_time:.3f}s.")

        for line in response_obj.iter_lines():
            if not line: continue
            decoded_line = line.decode("utf-8")

            try:
                data: Dict[str, Any] = {}
                
                if LLM_BACKEND == "LM_STUDIO":
                    json_str = decoded_line[len(SSE_DATA_PREFIX):].strip()
                    if json_str == SSE_DONE_MARKER: break
                    if not json_str: continue
                    data = json.loads(json_str)

                    delta_content = data.get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta_content:
                        sentence_buffer += delta_content
                        
                        last_yielded_end = 0
                        # Find all completed sentences in the current buffer
                        for match in sentence_end_pattern.finditer(sentence_buffer):
                            # Extract the full matched sentence including punctuation and trailing space
                            # match.group(1) is the sentence content + punctuation
                            # match.group(2) is the trailing space (or empty string)
                            sentence_to_yield = match.group(1) + (match.group(2) if match.group(2) else "")
                            
                            if sentence_to_yield:
                                yield {"type": "content", "data": sentence_to_yield}
                            last_yielded_end = match.end()
                        
                        # Keep any remaining partial sentence in buffer
                        sentence_buffer = sentence_buffer[last_yielded_end:]

                else: # Ollama direct JSON line format
                    json_str = decoded_line.strip()
                    if not json_str: continue
                    data = json.loads(json_str)

                    if "error" in data:
                        error_msg_detail = data.get('error', 'Unknown Ollama error from stream data')
                        formatted_error = f"{LLM_FAILED_PREFIX} (Ollama Stream Error: {error_msg_detail})"
                        logger.error(f"[{request_id}] LLM Router: {formatted_error}")
                        yield {"type": "error", "message": formatted_error}
                        return

                    stream_chunk_content = data.get("message", {}).get("content")
                    if stream_chunk_content:
                        sentence_buffer += stream_chunk_content
                        
                        last_yielded_end = 0
                        # Find all completed sentences in the current buffer
                        for match in sentence_end_pattern.finditer(sentence_buffer):
                            sentence_to_yield = match.group(1) + (match.group(2) if match.group(2) else "")
                            
                            if sentence_to_yield:
                                yield {"type": "content", "data": sentence_to_yield}
                            last_yielded_end = match.end()
                        
                        sentence_buffer = sentence_buffer[last_yielded_end:]


                    thinking_chunk = data.get("message", {}).get("thinking")
                    if thinking_chunk is not None:
                        accumulated_thinking_parts.append(thinking_chunk)
                        if thinking_chunk.strip():
                            logger.debug(f"[{request_id}] Appended thinking chunk (last 100 chars): ...{thinking_chunk.strip()[-100:]}")

                    if data.get("done"):
                        logger.debug(f"[{request_id}] LLM Router: Ollama stream 'done:true' received for current message.")
                        
                        # Yield any remaining content in the buffer when stream is done
                        # Ensure it's not just whitespace before yielding
                        if sentence_buffer.strip():
                            yield {"type": "content", "data": sentence_buffer.strip()}
                            sentence_buffer = "" # Clear buffer after yielding
                        
                        final_complete_thought = "".join(accumulated_thinking_parts).strip()
                        if final_complete_thought:
                            yield {"type": "thinking_summary", "data": [final_complete_thought]}
                            logger.info(f"[{request_id}] Sent complete thinking summary (length: {len(final_complete_thought)}).")
                        
                        ollama_done_payload = {k: v for k, v in data.items() if k != 'message'}
                        yield {"type": "ollama_done", "data": ollama_done_payload}
                        break

            except json.JSONDecodeError:
                logger.warning(f"[{request_id}] LLM Router: Skipping invalid JSON in stream: {decoded_line[:100]}..."); continue
            except Exception as e_proc:
                logger.exception(f"[{request_id}] LLM Router: Error processing stream chunk: '{decoded_line[:100]}...'")
                err_msg = f"{LLM_FAILED_PREFIX} (Processing Error: {str(e_proc)})"
                yield {"type": "error", "message": err_msg}
                return
        
        # Final yield for any residual buffer content if the stream ended without a punctuation
        # (e.g., LLM finished mid-sentence, or only provided a fragment)
        if sentence_buffer.strip():
            yield {"type": "content", "data": sentence_buffer.strip()}
        
        logger.info(f"[{request_id}] LLM Router: Stream processing finished.")

    except requests.exceptions.Timeout:
        logger.error(f"[{request_id}] LLM Router: ❌ LLM API request timed out after {STREAM_TIMEOUT_SECONDS} seconds.", exc_info=True)
        err_msg = f"{LLM_FAILED_PREFIX} (LLM stream request timed out)"
        yield {"type": "error", "message": err_msg}
    except requests.exceptions.RequestException as req_e:
        logger.exception(f"[{request_id}] LLM Router: ❌ LLM API request failed: {req_e}")
        err_yield_msg = f"{LLM_FAILED_PREFIX} (Error connecting to LLM server)"
        if hasattr(req_e, 'response') and req_e.response is not None:
            try:
                err_json = req_e.response.json()
                detail = err_json.get('error',{}).get('message') or err_json.get('detail', req_e.response.text)
                err_yield_msg = f"{LLM_FAILED_PREFIX} (LLM Server Error: {str(detail)[:200]})"
            except json.JSONDecodeError:
                err_yield_msg = f"{LLM_FAILED_PREFIX} (LLM Server Error: {req_e.response.status_code} - {req_e.response.text[:200]})"
        yield {"type": "error", "message": err_yield_msg}
    except Exception as e:
        logger.exception(f"[{request_id}] LLM Router: ❌ Unexpected error during LLM stream generation: {e}")
        err_msg = f"{LLM_FAILED_PREFIX} (Unexpected Error in LLM stream: {str(e)})"
        yield {"type": "error", "message": err_msg}
    finally:
        if response_obj:
            response_obj.close()
            logger.debug(f"[{request_id}] LLM Router: Closed LLM API response connection.")
        yield {"type": "done", "data": {"model": LLM_MODEL, "done": True}}

# --- APIRouter instance ---
router = APIRouter()

# --- LLM Chat Endpoint Definition ---
@router.post("/chat/stream", summary="Stream LLM Chat Completions (NDJSON with Native Thinking control)", tags=["LLM"])
async def llm_chat_stream_endpoint_router(request_data: LLMChatRequest):
    async def ndjson_wrapper():
        async for message_obj in iterate_in_threadpool(generate_llm_response_stream(
            prompt=request_data.prompt,
            history=request_data.history,
            llm_temperature=request_data.temperature,
            llm_top_p=request_data.top_p,
            llm_max_tokens=request_data.max_tokens,
            llm_presence_penalty=request_data.presence_penalty,
            llm_frequency_penalty=request_data.frequency_penalty,
            llm_repetition_penalty=request_data.repetition_penalty,
            enable_thinking=request_data.enable_thinking,
            llm_top_k=request_data.top_k,
            images=request_data.images,
            custom_system_prompt=request_data.system_prompt
        )):
            yield json.dumps(message_obj) + "\n"

    return StreamingResponse(ndjson_wrapper(), media_type="application/x-ndjson")