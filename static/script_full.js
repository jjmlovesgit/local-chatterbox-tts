// static/script_full.js (MODIFIED - V1.3 - Corrected Simli API URL initialization)
console.log("--- script_full.js (MODIFIED - V1.3 - Corrected Simli API URL initialization): Script execution STARTED. ---");
const TARGET_SAMPLE_RATE = 16000;

// --- Global Variables ---
let appForm, textInput, generateButton, modeSelect, chatHistoryDisplay, ttsVoiceSelect;
let avatarSelect, avatarSelectLabel;
let legacyAudioPlayer;
let simliVideoElement, simliPlaceholderVideo, simliAudioElement, simliStopButton;
let expLipsyncVideo; 
let simliWebRTCLog; 

// STT Variables
let micButton, sttStatus;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

// AudioContext & Direct TTS Variables
let audioContext = null;
let directTTSAudioBufferQueue = []; // This is the 'audio bucket' for all continuous audio chunks
let nextDirectTTSAudioStartTime = 0; // Tracks when the next audio segment should start playing
let isPlayingContinuousAudio = false; // Flag to manage the continuous playback loop

let audioContextInitialized = false; // Flag for audio context
let backgroundVideoPlayed = false; // NEW: Flag to track if background video has started playing

// LLM & Chat Variables
let chatHistory = [];
let currentLLMStreamController = null;

// Voice Cloner Tab elements
let clonerInputWav, clonerOutputName, createVoiceButton, clonerStatusMessage;

// --- GLOBAL VARIABLES FOR AVATAR CAROUSEL ---
// Define the sets of idle and talking videos for each avatar the appear in the order below
const avatarVideoSets = [
    { idle: 'static/mp4/bot4.mp4', talking: 'static/mp4/bot4.mp4' },
    { idle: "static/mp4/bot5.mp4", talking: "static/mp4/bot5.mp4"}, 
    { idle: "static/mp4/barista6.mp4", talking: "static/mp4/barista6.mp4" }, 
    { idle: "static/mp4/hipstermark.mp4", talking: "static/mp4/hipstermark.mp4"}, 
    { idle: "static/mp4/palmer.mp4", talking: "static/mp4/palmer.mp4"}, 
    { idle: "static/mp4/archer.mp4", talking: "static/mp4/archer.mp4"}, 
    { idle: "static/mp4/lena.mp4", talking: "static/mp4/lena.mp4"}, 
    { idle: "static/mp4/doc2.mp4", talking: "static/mp4/doc2exp.mp4"}, // Corrected typo in static path for doc2exp.mp4
];
let currentAvatarIndex = 0; // Index of the currently displayed avatar video set
let prevAvatarButton, nextAvatarButton; // References to the navigation buttons
let backgroundVideoUrlInput;
let loadBackgroundVideoButton;
let toggleBackgroundVideoButton; // NEW: Toggle play/pause button
let videoBackgroundElement;
let backgroundVideoStatus;
let backgroundVideoVolumeSlider;
let backgroundVideoVolumeValue;
let chooseLocalBackgroundVideoButton; // NEW: Button to trigger local file input
let localBackgroundVideoInput; // NEW: Hidden file input for local video
let systemPromptInput; // NEW: Textarea for custom system prompt
// --- End Global Variables ---


function appendSimliStatusLog(message) {
    if (simliWebRTCLog) {
        simliWebRTCLog.textContent += message + "\n";
        simliWebRTCLog.scrollTop = simliWebRTCLog.scrollHeight;
    }
    console.log(`[Simli Log] ${message}`); // Always console log for debugging
}

// --- Chat History Functions ---
function renderChatHistory() {
    if (!chatHistoryDisplay) { console.warn("Chat history display element not found."); return null; }
    chatHistoryDisplay.innerHTML = '';
    let lastAssistantContentDiv = null;
    chatHistory.forEach(msg => {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('chat-message');
        msgDiv.classList.add(msg.role === 'user' ? 'user-message' : 'assistant-message');
        const messageContentWrapper = document.createElement('div');
        messageContentWrapper.classList.add('message-content-wrapper');
        const strong = document.createElement('strong');
        strong.textContent = msg.role === 'user' ? 'You:' : 'Assistant:';
        messageContentWrapper.appendChild(strong);
        const contentDiv = document.createElement('div');
        if (msg.role === 'assistant' && msg.isStreaming) {
            contentDiv.classList.add('streaming-llm-content');
        }
        contentDiv.textContent = msg.content;
        messageContentWrapper.appendChild(contentDiv);
        msgDiv.appendChild(messageContentWrapper);
        chatHistoryDisplay.appendChild(msgDiv);
        if (msg.role === 'assistant') {
            lastAssistantContentDiv = contentDiv;
        }
    });
    if (chatHistoryDisplay) chatHistoryDisplay.scrollTop = chatHistoryDisplay.scrollHeight;
    return lastAssistantContentDiv;
}
function addUserMessageToChat(text) {
    appendSimliStatusLog(`[Chat] Adding user message: "${text.substring(0,30)}..."`);
    chatHistory.push({ role: 'user', content: text, isStreaming: false });
    return renderChatHistory();
}
function addAssistantMessageToChat(text, isStreaming = false) {
    appendSimliStatusLog(`[Chat] Adding/Updating assistant message: "${text.substring(0,30)}...", streaming: ${isStreaming}`);
    let assistantMessageInHistory;
    if (chatHistory.length > 0 &&
        chatHistory[chatHistory.length - 1].role === 'assistant' &&
        (chatHistory[chatHistory.length - 1].isStreaming || chatHistory[chatHistory.length - 1].content === "Thinking...")) {
        assistantMessageInHistory = chatHistory[chatHistory.length - 1];
        assistantMessageInHistory.content = text;
        assistantMessageInHistory.isStreaming = isStreaming;
    } else {
        assistantMessageInHistory = { role: 'assistant', content: text, isStreaming: isStreaming };
        chatHistory.push(assistantMessageInHistory);
    }
    return renderChatHistory();
}
function finalizeAssistantMessage() {
    appendSimliStatusLog("[Chat] Finalizing assistant message.");
    if (chatHistory.length > 0 && chatHistory[chatHistory.length - 1].role === 'assistant') {
        chatHistory[chatHistory.length - 1].isStreaming = false;
    }
    renderChatHistory();
}
// --- End Chat History Functions ---


// --- AudioContext & Direct TTS Functions ---
async function ensureAudioContext() {
    if (audioContextInitialized) return true;

    if (!audioContext || audioContext.state === 'closed') {
        appendSimliStatusLog("Attempting to initialize AudioContext...");
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: TARGET_SAMPLE_RATE });
            appendSimliStatusLog(`AudioContext created. State: ${audioContext.state}, Sample Rate: ${audioContext.sampleRate}`);
        } catch (e) {
             console.error("Error creating AudioContext:", e);
             appendSimliStatusLog(`Error creating AudioContext: ${e.message}. Please allow audio playback permissions if prompted.`);
             return false;
        }
    }

    if (audioContext.state === 'suspended') {
        appendSimliStatusLog("AudioContext state is suspended. Attempting to resume...");
        try {
            await audioContext.resume();
            appendSimliStatusLog(`AudioContext resume attempt finished. Current state: ${audioContext.state}`);
        }
        catch (e) {
            console.error("Error resuming AudioContext:", e);
            appendSimliStatusLog(`Error resuming AudioContext: ${e.message}.`);
            return false;
        }
    }

    const isRunning = audioContext.state === 'running';
    if (isRunning) {
        audioContextInitialized = true;
        appendSimliStatusLog("AudioContext now running and first-gesture listeners removed.");
    }
    return isRunning;
}

async function tryInitAudioAndVideo() {
    await ensureAudioContext();
    if (!backgroundVideoPlayed) {
        const videoBackground = document.getElementById('video-background');
        if (videoBackground) {
            try {
                await videoBackground.play();
                backgroundVideoPlayed = true; // Set flag to true on successful play
                appendSimliStatusLog("Background video started playing successfully.");
                // Remove listeners for background video after it starts
                document.removeEventListener('click', tryInitAudioAndVideo);
                document.removeEventListener('keydown', tryInitAudioAndVideo);
            } catch (error) {
                if (error.name !== 'AbortError') {
                    console.error("Error playing background video on user gesture:", error);
                    appendSimliStatusLog(`Failed to play background video: ${error.message}.`);
                } else {
                    // console.log("Background video play AbortError (normal on rapid interaction):", error);
                }
            }
        }
    }
    // Remove initial listeners for audio context once both are attempted
    if (audioContextInitialized && backgroundVideoPlayed) {
        document.removeEventListener('click', tryInitAudioAndVideo);
        document.removeEventListener('keydown', tryInitAudioAndVideo);
        appendSimliStatusLog("All initial user-gesture media playback setup complete. Listeners removed.");
    }
}


// --- CONTINUOUS AUDIO PLAYBACK HANDLER (The 'Bucket Player') ---
async function startContinuousAudioPlayback() {
    if (isPlayingContinuousAudio || directTTSAudioBufferQueue.length === 0 || !audioContext || audioContext.state !== 'running') {
        return;
    }

    isPlayingContinuousAudio = true;
    appendSimliStatusLog("Starting continuous playback loop.");

    let playbackCompletionPromise = new Promise(resolve => {
        let playedSourcesCount = 0;
        let totalSourcesToPlay = directTTSAudioBufferQueue.length;

        const scheduleNextBuffer = async () => {
            if (directTTSAudioBufferQueue.length > 0 && audioContext.state === 'running') {
                const audioBuffer = directTTSAudioBufferQueue.shift();

                if (audioBuffer && audioBuffer.length > 0) {
                    const source = audioContext.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(audioContext.destination);

                    source.onended = () => {
                        playedSourcesCount++;
                        if (playedSourcesCount >= totalSourcesToPlay && directTTSAudioBufferQueue.length === 0) {
                            isPlayingContinuousAudio = false;
                            resolve();
                        } else {
                            totalSourcesToPlay = directTTSAudioBufferQueue.length + playedSourcesCount;
                            scheduleNextBuffer();
                        }
                    };

                    try {
                        source.start(nextDirectTTSAudioStartTime);
                        nextDirectTTSAudioStartTime += audioBuffer.duration;
                    } catch (e) {
                        console.error("Error scheduling audio source:", e);
                        isPlayingContinuousAudio = false;
                        resolve();
                        return;
                    }
                } else {
                    scheduleNextBuffer();
                }
                await new Promise(resolve => setTimeout(resolve, 10));
            } else {
                if (playedSourcesCount === 0 && totalSourcesToPlay === 0) {
                    isPlayingContinuousAudio = false;
                    resolve();
                } else if (!isPlayingContinuousAudio) {
                     resolve();
                }
            }
        };

        scheduleNextBuffer();
    });

    return playbackCompletionPromise;
}
// --- END CONTINUOUS AUDIO PLAYBACK HANDLER ---

// The direct TTS function, acting as a 'Producer' for the audio bucket
async function speakTextDirectly(text, voice) {
    appendSimliStatusLog("Direct TTS Function CALLED.");
    if (!await ensureAudioContext()){
        appendSimliStatusLog(`AudioContext not running. Aborting TTS.`);
        return false;
    }
    appendSimliStatusLog("AudioContext IS READY. Fetching and buffering for: " + text.substring(0, 30) + "...");

    const ttsPayload = {
        text: text,
        voice: voice,
        temperature: parseFloat(document.getElementById('tts_temp_slider').value),
        exaggeration: parseFloat(document.getElementById('tts_top_p_slider').value),
        cfg: parseFloat(document.getElementById('tts_rep_penalty_slider').value),
        speed: parseFloat(document.getElementById('tts_speed_slider').value),
        chunk_size: parseInt(document.getElementById('streaming_chunk_size_slider').value, 10), // Use chunk_size
        context_window: parseInt(document.getElementById('context_window_slider').value, 10), // Use context_window
        fade_duration: parseFloat(document.getElementById('fade_duration_slider').value), // Use fade_duration
        trim_start_ms: parseInt(document.getElementById('trim_start_ms_slider').value, 10), // Use trim_start_ms
    };

    let resolveAudioQueuePromise;
    let audioQueuePromise = new Promise(resolve => {
        resolveAudioQueuePromise = resolve;
    });

    try {
        const response = await fetch('/api/tts/stream_direct', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(ttsPayload) });
        if (!response.ok) throw new Error(`Direct TTS API Error: ${response.status} ${response.statusText}`);

        const reader = response.body.getReader();
        const responseSampleRateHeader = response.headers.get('X-Sample-Rate');
        const streamSampleRate = responseSampleRateHeader ? parseInt(responseSampleRateHeader, 10) : TARGET_SAMPLE_RATE;

        let lastPlaybackPromise = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                appendSimliStatusLog(`Finished reading all audio chunks for sentence: "${text.substring(0, 20)}..."`);
                if (lastPlaybackPromise) {
                    await lastPlaybackPromise;
                }
                resolveAudioQueuePromise();
                break;
            }
            if (value && value.byteLength > 0) {
                const float32Array = new Float32Array(value.buffer, value.byteOffset, value.byteLength / Float32Array.BYTES_PER_ELEMENT);
                if (float32Array.length === 0) continue;

                const buffer = audioContext.createBuffer(1, float32Array.length, streamSampleRate);
                buffer.copyToChannel(float32Array, 0);

                directTTSAudioBufferQueue.push(buffer);
                const newPlaybackPromise = startContinuousAudioPlayback();
                if (newPlaybackPromise) {
                    lastPlaybackPromise = newPlaybackPromise;
                }
            }
        }
        return audioQueuePromise;

    } catch (error) {
        console.error('[App/speakTextDirectly] Direct TTS Request failed:', error);
        appendSimliStatusLog(`Direct TTS Error: ${error.message}`);
        resolveAudioQueuePromise();
        return false;
    }
}
// --- End AudioContext & Direct TTS Functions ---


// --- Simli UI & Session Functions ---
function resetSimliUI() {
    appendSimliStatusLog("Resetting Simli UI elements.");
    if(generateButton) { generateButton.disabled = false; generateButton.textContent = 'Generate'; }
    if(simliStopButton) simliStopButton.style.display = 'none';

    if(simliVideoElement) {
        if (simliVideoElement.srcObject) {
            const stream = simliVideoElement.srcObject;
            stream.getTracks().forEach(track => track.stop());
            simliVideoElement.srcObject = null;
        }
        simliVideoElement.style.display = 'none';
        simliVideoElement.style.borderColor = 'lime';
    }
    if (simliPlaceholderVideo) {
        // Ensure the correct placeholder video for the current avatar is displayed
        updateAvatarVideos(); // This will set the correct src and display it if necessary
        simliPlaceholderVideo.style.display = 'block'; // Ensure placeholder is shown on reset
        simliPlaceholderVideo.play().catch(e => {});
        simliPlaceholderVideo.style.borderColor = 'var(--color-link)'; // Default border color
    }
    if (expLipsyncVideo) {
        expLipsyncVideo.pause();
        expLipsyncVideo.style.display = 'none';
        expLipsyncVideo.loop = false;
        expLipsyncVideo.currentTime = 0; // Reset time
    }

    if(simliAudioElement && simliAudioElement.srcObject) {
        (simliAudioElement.srcObject).getTracks().forEach(track => track.stop());
        simliAudioElement.srcObject = null;
    }

    if (avatarSelectLabel) avatarSelectLabel.textContent = 'Use Simli Avatar: No';
    if (avatarSelect) avatarSelect.value = 'false';
}

function stopSimliSessionApp() {
    appendSimliStatusLog("stopSimliSessionApp called.");
    if (typeof SimliWebRTC !== 'undefined') {
        appendSimliStatusLog("Calling SimliWebRTC.stopSession().");
        SimliWebRTC.stopSession();
    } else {
        appendSimliStatusLog("SimliWebRTC is not defined. Cannot call stopSession().");
    }

    // Immediately clear audio context and queues regardless of SimliWebRTC status
    if (audioContext && audioContext.state === 'running') {
        audioContext.close().then(() => {
            audioContext = null;
            audioContextInitialized = false;
            appendSimliStatusLog("AudioContext closed after session stop.");
        }).catch(e => {
            console.error("Error closing AudioContext:", e);
            appendSimliStatusLog(`Error closing AudioContext: ${e.message}`);
        });
    }
    directTTSAudioBufferQueue = [];
    isPlayingContinuousAudio = false;
    nextDirectTTSAudioStartTime = 0;

    resetSimliUI();
    appendSimliStatusLog("Simli UI elements reset.");
}
// --- End Simli UI & Session Functions ---

// --- AVATAR VIDEO CAROUSEL ---
function updateAvatarVideos() {
    const currentSet = avatarVideoSets[currentAvatarIndex];
    if (simliPlaceholderVideo) {
        simliPlaceholderVideo.src = currentSet.idle;
        simliPlaceholderVideo.load(); // Explicitly load the new video source
        // Only play if currently visible, otherwise it will be played when made visible
        if (simliPlaceholderVideo.style.display === 'block') {
            simliPlaceholderVideo.play().catch(e => console.error("Error playing placeholder video:", e));
        }
        simliPlaceholderVideo.currentTime = 0; // Reset video to start
    }
    if (expLipsyncVideo) {
        expLipsyncVideo.src = currentSet.talking;
        expLipsyncVideo.load(); // Explicitly load the new video source
        expLipsyncVideo.pause(); // Ensure it's paused initially
        expLipsyncVideo.currentTime = 0; // Reset video to start
    }
    appendSimliStatusLog(`Avatar video set updated to: ${currentSet.idle.split('/').pop()}`);
}
// --- END NEW FUNCTION ---


// --- Typewriter Function ---
function typeText(element, text, speed = 20) {
    let i = 0;
    function typing() {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            i++;
            if (chatHistoryDisplay) {
                chatHistoryDisplay.scrollTop = chatHistoryDisplay.scrollHeight;
            }
            setTimeout(typing, speed);
        }
    }
    typing();
}

// --- LLM Functions ---
async function fetchLLMResponse(promptText, sentenceCallback) {
    appendSimliStatusLog(`[LLM] Called with prompt: "${promptText.substring(0,30)}..."`);
    let lastAssistantContentDiv = addAssistantMessageToChat("Thinking...", true);
    let accumulatedLLMText = "";
    let isFirstSentencePart = true;

    // Get the custom system prompt from the new textarea
    const customSystemPrompt = systemPromptInput ? systemPromptInput.value.trim() : "";

    // Construct the messages array. If a custom prompt is provided, use it.
    // Otherwise, the backend will use its default.
    const messages = [];
    if (customSystemPrompt) {
        messages.push({"role": "system", "content": customSystemPrompt});
    } else {
        // If no custom prompt, ensure the backend uses its default by not sending a system message here
        // or by sending a placeholder that the backend understands as "use default".
    }


    const historyForAPI = chatHistory.slice(0, -2)
        .filter(msg => msg.content !== "Thinking..." && !msg.isStreaming)
        .map(msg => ({ role: msg.role, content: msg.content }));

    messages.push(...historyForAPI); // Add chat history
    messages.push({"role": "user", "content": promptText}); // Add current user prompt

    const llmPayload = {
        prompt: promptText, // This will be the user's current input
        history: messages, // Send the full constructed message history including system prompt
        temperature: parseFloat(document.getElementById('llm_temp_slider').value),
        top_p: parseFloat(document.getElementById('llm_top_p_slider').value),
        presence_penalty: parseFloat(document.getElementById('llm_rep_penalty_slider').value),
        top_k: parseInt(document.getElementById('llm_top_k_slider').value, 10),
        system_prompt: customSystemPrompt, // NEW: Pass the custom system prompt to the backend
        enable_thinking: document.getElementById('enable_thinking_select').value === 'true' // Pass enable_thinking flag
    };
    currentLLMStreamController = new AbortController();

    try {
        appendSimliStatusLog(`[LLM] Fetching from /api/llm/chat/stream...`);
        const response = await fetch('/api/llm/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(llmPayload),
            signal: currentLLMStreamController.signal
        });
        appendSimliStatusLog(`[LLM] Response status: ${response.status}`);
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`LLM API Error: ${response.status} ${errorText || response.statusText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let initialThinkingRemoved = false;

        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                appendSimliStatusLog("[LLM] Stream finished.");
                break;
            }
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (line.trim() === '') continue;
                try {
                    const parsed = JSON.parse(line);

                    if (parsed.type === 'content' && parsed.data) {
                        // If thinking was previously displayed, clear it before showing content
                        if (!initialThinkingRemoved && lastAssistantContentDiv.textContent === "Thinking...") {
                            lastAssistantContentDiv.textContent = "";
                            initialThinkingRemoved = true;
                        } else if (initialThinkingRemoved && lastAssistantContentDiv.textContent.startsWith("Thinking...")) {
                             // If thinking was being streamed, clear it now that content is coming
                             lastAssistantContentDiv.textContent = "";
                        }


                        let textForTyping = parsed.data; // Use parsed.data directly
                        if (!isFirstSentencePart) {
                            textForTyping = ' ' + parsed.data; // Add space if not first part
                        }

                        typeText(lastAssistantContentDiv, textForTyping);

                        if (sentenceCallback) {
                            await sentenceCallback(parsed.data, ttsVoiceSelect.value);
                        }

                        accumulatedLLMText += parsed.data;
                        chatHistory[chatHistory.length - 1].content = accumulatedLLMText;
                        chatHistory[chatHistory.length - 1].isStreaming = true;

                        isFirstSentencePart = false;

                    } else if (parsed.type === 'thinking_summary' && parsed.data && parsed.data.length > 0) {
                        // Handle LLM thinking output
                        const thinkingText = parsed.data[0]; // Assuming data is an array with thinking text
                        appendSimliStatusLog(`[LLM Thinking] ${thinkingText}`);
                        // Optionally, you can also display this in a specific UI element
                        // For example, if you want it in the chat history temporarily:
                        // addAssistantMessageToChat(`Thinking: ${thinkingText}`, false);
                        // Or in a dedicated thinking log area:
                        // if (llmThinkingLogElement) llmThinkingLogElement.textContent = thinkingText;

                        // If "Thinking..." is still in chat history, replace it with the actual thinking or clear it
                        if (lastAssistantContentDiv.textContent === "Thinking...") {
                            lastAssistantContentDiv.textContent = `Thinking: ${thinkingText.substring(0, 50)}...`; // Show truncated thinking
                            initialThinkingRemoved = true; // Mark as thinking displayed
                        }

                    } else if (parsed.type === 'ollama_done' || parsed.type === 'done') {
                        finalizeAssistantMessage();
                    } else if (parsed.type === 'error') {
                        throw new Error(parsed.message);
                    }
                } catch (e) {
                    console.error("Error parsing NDJSON line:", e, "Line:", line);
                    appendSimliStatusLog(`[LLM Parsing Error] ${e.message}. Line: ${line.substring(0, 100)}...`);
                }
            }
        }

    } catch (error) {
        console.error('LLM Request failed:', error);
        const errorMsg = error.name === 'AbortError' ? "(LLM request cancelled)" : `(LLM Error: ${error.message})`;
        addAssistantMessageToChat(errorMsg, false);
    } finally {
        finalizeAssistantMessage();
        currentLLMStreamController = null;
    }
}
// --- End LLM Functions ---


// --- UI Initialization Functions ---
async function initializeTTSVoicesUI() { // Made async
    const currentTtsVoiceSelect = document.getElementById('tts_voice_dd');
    if (!currentTtsVoiceSelect) {
        console.warn("TTS Voice select dropdown (tts_voice_dd) not found.");
        return;
    }

    try {
        appendSimliStatusLog("Fetching available voices from backend /api/config...");
        const configResponse = await fetch("/api/config");
        if (!configResponse.ok) {
            const errorText = await configResponse.text();
            throw new Error(`Failed to fetch config: ${configResponse.status} - ${errorResponseText || configResponse.statusText}`);
        }
        const config = await configResponse.json();
        const availableVoices = config.available_voices || [];
        const defaultTtsVoice = config.default_tts_voice;

        // Clear existing options (if any, e.g., hardcoded ones)
        currentTtsVoiceSelect.innerHTML = '';

        if (availableVoices.length === 0) {
            appendSimliStatusLog("No voices found from backend. Please add .pt files to the voices/ directory.");
            const option = document.createElement('option');
            option.value = "no_voices_found";
            option.textContent = "No Voices Found";
            currentTtsVoiceSelect.appendChild(option);
            currentTtsVoiceSelect.disabled = true; // Disable if no voices
            return;
        }

        availableVoices.forEach(voice => {
            const option = document.createElement('option');
            option.value = voice;
            option.textContent = voice.charAt(0).toUpperCase() + voice.slice(1); // Capitalize first letter
            currentTtsVoiceSelect.appendChild(option);
        });

        // Set the default voice from the backend if it exists and is in the list
        if (defaultTtsVoice && availableVoices.includes(defaultTtsVoice)) {
            currentTtsVoiceSelect.value = defaultTtsVoice;
            appendSimliStatusLog(`Default TTS voice set to: ${defaultTtsVoice}`);
        } else if (availableVoices.length > 0) {
            // Otherwise, select the first available voice
            currentTtsVoiceSelect.value = availableVoices[0];
            appendSimliStatusLog(`Default TTS voice set to first available: ${availableVoices[0]}`);
        }
        currentTtsVoiceSelect.disabled = false; // Enable if voices are loaded

    } catch (error) {
        console.error("Error fetching or initializing TTS voices:", error);
        appendSimliStatusLog(`ERROR: Failed to load TTS voices: ${error.message}`);
        // Fallback to a disabled state or show an error option
        currentTtsVoiceSelect.innerHTML = '';
        const option = document.createElement('option');
        option.value = "load_error";
        option.textContent = "Error Loading Voices";
        currentTtsVoiceSelect.appendChild(option);
        currentTtsVoiceSelect.disabled = true;
    }
}
function initializeSlidersUI() {
    const sliderConfigs = [
        { id: 'llm_temp_slider', valueId: 'llm_temp_value', isFloat: true, precision: 2 },
        { id: 'llm_top_p_slider', valueId: 'llm_top_p_value', isFloat: true, precision: 2 },
        { id: 'llm_rep_penalty_slider', valueId: 'llm_rep_penalty_value', isFloat: true, precision: 2 },
        { id: 'llm_top_k_slider', valueId: 'llm_top_k_value', isFloat: false },
        { id: 'tts_temp_slider', valueId: 'tts_temp_value', isFloat: true, precision: 2 },
        { id: 'tts_top_p_slider', valueId: 'tts_top_p_value', isFloat: true, precision: 2 },
        { id: 'tts_rep_penalty_slider', valueId: 'tts_rep_penalty_value', isFloat: true, precision: 2 },
        { id: 'tts_speed_slider', valueId: 'tts_speed_value', isFloat: true, precision: 2 },
        { id: 'streaming_chunk_size_slider', valueId: 'streaming_chunk_size_value', isFloat: false },
        { id: 'context_window_slider', valueId: 'context_window_value', isFloat: false },
        { id: 'fade_duration_slider', valueId: 'fade_duration_value', isFloat: true, precision: 2 },
        { id: 'trim_start_ms_slider', valueId: 'trim_start_ms_value', isFloat: false }
    ];
    sliderConfigs.forEach(config => {
        const slider = document.getElementById(config.id); const valueSpan = document.getElementById(config.valueId);
        if (!slider || !valueSpan) { console.warn(`Slider or value span not found: ${config.id}/${config.valueId}`); return; }
        const updateValueDisplay = () => { const val = parseFloat(slider.value); valueSpan.textContent = config.isFloat ? val.toFixed(config.precision || 1) : slider.value; };
        slider.addEventListener('input', updateValueDisplay); updateValueDisplay();
    });
}
// --- End UI Initialization Functions ---


// --- STT (Microphone) Functions ---
function updateSttStatus(message = "", isError = false) {
    if (sttStatus) {
        sttStatus.textContent = message;
        sttStatus.style.color = isError ? 'var(--color-error)' : 'var(--color-secondary-text)';
    }
}

async function startRecording() {
    if (!await ensureAudioContext()) {
        updateSttStatus("Audio system not ready. Please click page first.", true);
        return;
    }

    if (isRecording || !micButton) return;
    if (generateButton && generateButton.disabled) {
        updateSttStatus("Still processing previous request.", true);
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioChunks = [];
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            if (mediaRecorder.stream) {
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }

            if (audioChunks.length === 0) {
                updateSttStatus("No audio recorded.", true);
                resetRecordingState();
                return;
            }

            updateSttStatus("Processing...");
            if (micButton) {
                micButton.classList.add('processing');
                micButton.disabled = true;
            }

            const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
            const formData = new FormData();
            const fileExtension = mediaRecorder.mimeType.split('/')[1].split(';')[0] || 'webm';
            const filename = `recording.${fileExtension}`;
            formData.append('audio_file', audioBlob, filename);

            try {
                const response = await fetch('/api/stt/transcribe', {
                    method: 'POST',
                    body: formData,
                });
                const result = await response.json();
                if (!response.ok || result.error) {
                    throw new Error(result.error || `Server error: ${response.status}`);
                }
                const transcribedText = result.text || "";
                if (textInput) {
                     textInput.value = textInput.value ? `${textInput.value} ${transcribedText}`.trim() : transcribedText;
                }
                const shouldSubmit = transcribedText.trim() !== "";
                resetRecordingState();
                if (shouldSubmit) {
                    updateSttStatus("");
                    if(appForm) {
                        if(generateButton) generateButton.disabled = false;
                        appForm.requestSubmit();
                    }
                } else {
                    updateSttStatus("Transcription empty.", true);
                }
            } catch (error) {
                console.error("STT Transcription Error:", error);
                updateSttStatus(`Error: ${error.message}`, true);
                resetRecordingState();
            }
        };

        mediaRecorder.start();
        isRecording = true;
        if (micButton) {
            micButton.textContent = "Stop";
            micButton.classList.add('recording');
        }
        updateSttStatus("Recording...");

    } catch (error) {
        console.error("Microphone Access Error:", error);
        updateSttStatus(`Mic access error: ${error.message}`, true); // Generic error message for simplicity
        resetRecordingState();
    }
}

function stopRecording() {
    if (!isRecording || !mediaRecorder || mediaRecorder.state === 'inactive') return;
    mediaRecorder.stop();
}

function resetRecordingState() {
    isRecording = false;
    if (mediaRecorder) {
        if (mediaRecorder.stream) {
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
        mediaRecorder = null;
    }
    audioChunks = [];
    if (micButton) {
        micButton.textContent = "Mic";
        micButton.classList.remove('recording', 'processing');
        micButton.disabled = false;
    }
    if (!sttStatus.textContent.startsWith("Error:") && !sttStatus.textContent.startsWith("No audio recorded.")) {
        updateSttStatus("");
    }
}

function handleMicClick() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}
// --- END STT (Microphone) Functions ---

// --- Voice Cloner Functions ---
async function handleCreateVoice() {
    if (!clonerInputWav || !clonerInputWav.files || clonerInputWav.files.length === 0) {
        clonerStatusMessage.textContent = "Please select a WAV file to upload.";
        clonerStatusMessage.className = "voice-cloner-output error";
        return;
    }
    const audioFile = clonerInputWav.files[0];

    const voiceName = clonerOutputName.value.trim();
    if (!voiceName) {
        clonerStatusMessage.textContent = "Please enter a name for the new voice.";
        clonerStatusMessage.className = "voice-cloner-output error";
        return;
    }

    // Disable button and show processing message
    createVoiceButton.disabled = true;
    createVoiceButton.textContent = "Creating Voice...";
    clonerStatusMessage.textContent = "Processing voice for cloning...";
    clonerStatusMessage.className = "voice-cloner-output"; // Reset class

    const formData = new FormData();
    formData.append('audio_file', audioFile);
    formData.append('voice_name', voiceName);

    try {
        const response = await fetch('/api/clone_voice', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.detail || `Server error: ${response.status}`);
        }

        clonerStatusMessage.textContent = `Success: ${result.message}`;
        clonerStatusMessage.className = "voice-cloner-output success";
        clonerOutputName.value = ''; // Clear input on success
        clonerInputWav.value = ''; // Clear file input on success
        
        // Refresh the TTS voice dropdown after successful cloning
        await initializeTTSVoicesUI(); // Re-fetch and update voices

    } catch (error) {
        console.error("Voice Cloning Error:", error);
        clonerStatusMessage.textContent = `Error: ${error.message}`;
        clonerStatusMessage.className = "voice-cloner-output error";
    } finally {
        createVoiceButton.disabled = false;
        createVoiceButton.textContent = "Create Voice (.pt) 🎶";
    }
}
// --- End Voice Cloner Functions ---

// --- Tab Switching Functions ---
function showTab(tabId) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tabContent => {
        tabContent.classList.add('hidden');
    });

    // Deactivate all tab buttons
    document.querySelectorAll('.tab-button').forEach(tabButton => {
        tabButton.classList.remove('active');
    });

    // Show the selected tab content
    const selectedTabContent = document.getElementById(tabId);
    if (selectedTabContent) {
        selectedTabContent.classList.remove('hidden');
    }

    // Activate the clicked tab button
    document.querySelector(`.tab-button[data-tab="${tabId}"]`).classList.add('active');

    // Special handling for logs tab to ensure it's always scrolled to bottom
    if (tabId === 'logs-tab' && simliWebRTCLog) {
        simliWebRTCLog.scrollTop = simliWebRTCLog.scrollHeight;
    }
}
// --- End Tab Switching Functions ---


// --- Main App Initialization ---
async function initializeApp() {
    let simliApiKeyClient = null;
    let simliFaceIdClient = null;

    try {
        appendSimliStatusLog("Attempting to fetch Simli and TTS configuration from backend...");
        const configResponse = await fetch("/api/config");
        if (!configResponse.ok) {
            const errorText = await configResponse.text();
            throw new Error(`Failed to fetch config: ${configResponse.status} - ${errorResponseText || configResponse.statusText}`);
        }
        const config = await configResponse.json();
        simliApiKeyClient = config.simli_api_key;
        simliFaceIdClient = config.simli_face_id;
        appendSimliStatusLog("Successfully fetched client configuration from backend.");

        if (!simliApiKeyClient || !simliFaceIdClient) {
            console.error("Simli API keys or Face ID were not provided by the backend configuration.");
            appendSimliStatusLog("ERROR: Simli API keys/Face ID missing from backend config. Simli functionality may be disabled.");
            // Do NOT use alert here, it blocks the UI.
            // alert("Simli API keys or Face ID not configured on the server. Simli will be disabled.");
            if (avatarSelect) {
                avatarSelect.value = 'false';
                avatarSelect.disabled = true;
            }
            if (avatarSelectLabel) {
                avatarSelectLabel.textContent = 'Use Simli Avatar: (Config N/A)';
            }
        }

    } catch (error) {
        console.error("Could not fetch config from backend:", error);
        appendSimliStatusLog("ERROR: Could not fetch client config. Simli functionality may be disabled: " + error.message);
        // Do NOT use alert here.
        // alert("Could not load app configuration from the server. Simli will be disabled.");
        if (avatarSelect) {
            avatarSelect.value = 'false';
            avatarSelect.disabled = true;
        }
        if (avatarSelectLabel) {
            avatarSelectLabel.textContent = 'Use Simli Avatar: (Config N/A)';
        }
    }

    // Call initializeTTSVoicesUI after potentially fetching config
    await initializeTTSVoicesUI(); // Now awaitable

    if (typeof SimliWebRTC !== 'undefined' && avatarSelect && !avatarSelect.disabled) {
        SimliWebRTC.init({
            SIMLI_API_KEY_CLIENT: simliApiKeyClient,
            SIMLI_FACE_ID_CLIENT: simliFaceIdClient,
            // REMOVED: No longer override simliApiSessionUrl here.
            // It's now correctly set as a default in simli_webrtc.js itself.
            // simliApiSessionUrl: 'https://api.simli.ai/startAudioToVideoSession', // <-- REMOVED THIS LINE
            simliWsUrl: 'wss://api.simli.ai/startWebRTCSession',
            uiCallbacks: {
                appendSimliStatusLog: appendSimliStatusLog,
                // Removed UI updates for these, now only console/simli-log
                updateIceGatheringState: (state) => { appendSimliStatusLog(`[WebRTC State] ICE Gathering: ${state}`); },
                updateIceConnectionState: (state) => { appendSimliStatusLog(`[WebRTC State] ICE Connection: ${state}`); },
                updateSignalingState: (state) => { appendSimliStatusLog(`[WebRTC State] Signaling: ${state}`); },
                updateOfferSdp: (sdp) => { appendSimliStatusLog(`[WebRTC SDP] Offer: ${sdp}`); },
                updateAnswerSdp: (sdp) => { appendSimliStatusLog(`[WebRTC SDP] Answer: ${sdp}`); },

                setSimliVideoSrcObject: (stream) => {
                    if(simliVideoElement && simliPlaceholderVideo) {
                        appendSimliStatusLog("Simli video stream received. Applying & starting poll.");
                        simliVideoElement.srcObject = stream;
                        simliVideoElement.play().catch(e => appendSimliStatusLog(`Error playing Simli stream: ${e.message}`));

                        const checkReadyState = () => {
                            if (simliVideoElement.srcObject === stream && simliVideoElement.style.display === 'none') {
                                if (simliVideoElement.readyState >= 3) {
                                     appendSimliStatusLog(`Simli WEBRTC video readyState is ${simliVideoElement.readyState}. Displaying it.`);
                                     simliVideoElement.style.display = 'block';
                                     simliPlaceholderVideo.style.display = 'none';
                                } else {
                                     setTimeout(checkReadyState, 100);
                                }
                            }
                        };
                        setTimeout(checkReadyState, 100);
                    }
                },
                setSimliAudioSrcObject: (stream) => { if(simliAudioElement) simliAudioElement.srcObject = stream; },
                onSessionStoppedByServer: () => {
                    appendSimliStatusLog("Simli session stopped by server. Updating UI.");
                    resetSimliUI();
                },
                getTtsParams: () => ({
                    // Removed voice parameter as it's passed separately to sendTextForTTS directly
                    temperature: parseFloat(document.getElementById('tts_temp_slider').value),
                    exaggeration: parseFloat(document.getElementById('tts_top_p_slider').value),
                    cfg: parseFloat(document.getElementById('tts_rep_penalty_slider').value),
                    speed: parseFloat(document.getElementById('tts_speed_slider').value),
                    streaming_chunk_size: parseInt(document.getElementById('streaming_chunk_size_slider').value, 10),
                    context_window: parseInt(document.getElementById('context_window_slider').value, 10),
                    fade_duration: parseFloat(document.getElementById('fade_duration_slider').value),
                    trim_start_ms: parseInt(document.getElementById('trim_start_ms_slider').value, 10),
                }),
                onConnectionEstablished: () => {
                    appendSimliStatusLog("Simli connection established callback.");
                    if (avatarSelectLabel) avatarSelectLabel.textContent = 'Use Simli Avatar: Yes (Connected)';
                    if (simliPlaceholderVideo && simliVideoElement.style.display === 'none') {
                        simliPlaceholderVideo.style.borderColor = 'lime';
                    }
                    if(generateButton && generateButton.textContent === 'Connecting Simli...') {
                        generateButton.disabled = false;
                        generateButton.textContent = 'Generate';
                    }
               },
                onConnectionFailed: () => {
                    appendSimliStatusLog("Simli connection failed callback.");
                    if (avatarSelectLabel) avatarSelectLabel.textContent = 'Use Simli Avatar: Yes (Error)';
                    if (simliPlaceholderVideo) {
                        simliPlaceholderVideo.style.borderColor = 'red';
                    }
                    resetSimliUI();
               },
                onAudioProcessingComplete: (success) => {
                     appendSimliStatusLog(`Simli audio processing finished. Success: ${success}`);
                    if (avatarSelect && avatarSelect.value === 'true') {
                        const targetBorderElement = (simliVideoElement.style.display === 'block') ? simliVideoElement : simliPlaceholderVideo;
                        if (success) {
                            if (avatarSelectLabel) avatarSelectLabel.textContent = 'Use Simli Avatar: Yes (Connected)';
                            if (targetBorderElement) targetBorderElement.style.borderColor = 'lime';
                        } else {
                            if (avatarSelectLabel) avatarSelectLabel.textContent = 'Use Simli Avatar: Yes (Audio Error)';
                            if (targetBorderElement) targetBorderElement.style.borderColor = 'orange';
                        }
                    }
                }
            }
        });
    } else if (!avatarSelect || avatarSelect.disabled) {
        console.warn("SimliWebRTC not loaded or Avatar select disabled by config fetch failure.");
    } else {
        console.error("SimliWebRTC helper script not loaded!");
        if (avatarSelect) avatarSelect.disabled = true;
    }
}


// --- DOMContentLoaded - Main Execution ---
document.addEventListener('DOMContentLoaded', () => {
    // Get All Elements
    // Removed specific UI elements for WebRTC status
    // iceGatheringStateSpan = document.getElementById("ice-gathering-state");
    // iceConnectionStateSpan = document.getElementById("ice-connection-state");
    // signalingStateSpan = document.getElementById("signaling-state");
    simliWebRTCLog = document.getElementById("simli-log"); // This remains for general log output
    // Removed specific UI elements for SDP display
    // offerSdpPre = document.getElementById("offer-sdp");
    // answerSdpPre = document.getElementById("answer-sdp");

    appForm = document.getElementById('app-form');
    textInput = document.getElementById('text_input');
    generateButton = document.getElementById('generate-button');
    modeSelect = document.getElementById('mode_select');
    chatHistoryDisplay = document.getElementById('chat-history-display');
    ttsVoiceSelect = document.getElementById('tts_voice_dd');
    avatarSelect = document.getElementById('avatar_select');
    avatarSelectLabel = document.querySelector("label[for='avatar_select']");
    legacyAudioPlayer = document.getElementById('legacy-audio-player');
    simliVideoElement = document.getElementById("simli-video-element");
    simliPlaceholderVideo = document.getElementById("simli-placeholder-video");
    simliAudioElement = document.getElementById("simli-audio-element");
    simliStopButton = document.getElementById("simli_stop_button");
    micButton = document.getElementById("mic-button");
    sttStatus = document.getElementById("stt-status");
    expLipsyncVideo = document.getElementById("exp-lipsync-video");

    // --- NEW: Get references to avatar navigation buttons ---
    prevAvatarButton = document.getElementById("prev-avatar");
    nextAvatarButton = document.getElementById("next-avatar");
    // --- END NEW ---

    // Voice Cloner tab elements
    clonerInputWav = document.getElementById('cloner_input_wav');
    clonerOutputName = document.getElementById('cloner_output_name');
    createVoiceButton = document.getElementById('create-voice-button');
    clonerStatusMessage = document.getElementById('cloner-status-message');

    // --- NEW: Get references to background video elements ---
    backgroundVideoUrlInput = document.getElementById('backgroundVideoUrl');
    loadBackgroundVideoButton = document.getElementById('loadBackgroundVideoButton');
    toggleBackgroundVideoButton = document.getElementById('toggleBackgroundVideoButton'); // NEW
    videoBackgroundElement = document.getElementById('video-background');
    backgroundVideoStatus = document.getElementById('backgroundVideoStatus');
    backgroundVideoVolumeSlider = document.getElementById('backgroundVideoVolume');
    backgroundVideoVolumeValue = document.getElementById('backgroundVideoVolumeValue');
    chooseLocalBackgroundVideoButton = document.getElementById('chooseLocalBackgroundVideoButton'); // NEW
    localBackgroundVideoInput = document.getElementById('localBackgroundVideoInput'); // NEW
    // --- END NEW ---

    // --- NEW: Get reference to system prompt input ---
    systemPromptInput = document.getElementById('systemPromptInput'); // NEW
    // --- END NEW ---

    // Event listener for the "Load Background Video" button
    if (loadBackgroundVideoButton) {
        loadBackgroundVideoButton.addEventListener('click', () => {
            const url = backgroundVideoUrlInput.value.trim();
            console.log("Load Background Video button clicked. URL:", url);

            if (url) {
                // Set the new video source
                videoBackgroundElement.src = url;
                videoBackgroundElement.load(); // Reload the video element to pick up the new source
                
                // Set initial volume from slider or default
                if (backgroundVideoVolumeSlider) {
                    videoBackgroundElement.volume = parseFloat(backgroundVideoVolumeSlider.value);
                } else {
                    videoBackgroundElement.volume = 0.2; // Default if slider not found
                }

                videoBackgroundElement.play().then(() => {
                    backgroundVideoPlayed = true;
                    backgroundVideoStatus.textContent = 'Background video loaded and playing successfully!';
                    backgroundVideoStatus.style.color = 'var(--color-link)';
                    console.log("Background video started playing successfully.");
                    if (toggleBackgroundVideoButton) { // Update button text on successful play
                        toggleBackgroundVideoButton.textContent = 'Pause Background Video';
                    }
                }).catch(error => {
                    console.error('Autoplay of background video failed:', error);
                    backgroundVideoStatus.textContent = 'Video loaded, but autoplay might be blocked. You may need to interact with the page or ensure the video is muted.';
                    backgroundVideoStatus.style.color = 'var(--color-error)';
                    if (toggleBackgroundVideoButton) { // Set to Play if autoplay failed
                        toggleBackgroundVideoButton.textContent = 'Play Background Video';
                    }
                });
            } else {
                backgroundVideoStatus.textContent = 'Please enter a video URL.';
                backgroundVideoStatus.style.color = 'var(--color-error)';
                console.warn("No URL entered for background video.");
            }
        });

        // Optional: Add an event listener to clear status when input changes
        backgroundVideoUrlInput.addEventListener('input', () => {
            backgroundVideoStatus.textContent = ''; // Clear status message
        });

        // Event listener for background video volume slider
        if (backgroundVideoVolumeSlider && backgroundVideoVolumeValue) {
            const updateVolumeDisplay = () => {
                const val = parseFloat(backgroundVideoVolumeSlider.value);
                backgroundVideoVolumeValue.textContent = val.toFixed(2);
                if (videoBackgroundElement) {
                    videoBackgroundElement.volume = val;
                }
            };
            backgroundVideoVolumeSlider.addEventListener('input', updateVolumeDisplay);
            updateVolumeDisplay(); // Set initial display and volume
        }

        // Event listener for the toggle background video button
        if (toggleBackgroundVideoButton && videoBackgroundElement) {
            toggleBackgroundVideoButton.addEventListener('click', () => {
                console.log("Toggle Background Video button clicked."); // Diagnostic log added here
                if (videoBackgroundElement.paused) {
                    videoBackgroundElement.play().then(() => {
                        toggleBackgroundVideoButton.textContent = 'Pause Background Video';
                        backgroundVideoStatus.textContent = 'Background video playing.';
                        backgroundVideoStatus.style.color = 'var(--color-link)';
                        console.log("Background video resumed.");
                    }).catch(error => {
                        console.error('Failed to resume background video:', error);
                        backgroundVideoStatus.textContent = 'Failed to play video. Autoplay might be blocked.';
                        backgroundVideoStatus.style.color = 'var(--color-error)';
                    });
                } else {
                    videoBackgroundElement.pause();
                    toggleBackgroundVideoButton.textContent = 'Play Background Video';
                    backgroundVideoStatus.textContent = 'Background video paused.';
                    backgroundVideoStatus.style.color = 'var(--color-secondary-text)';
                    console.log("Background video paused.");
                }
            });

            videoBackgroundElement.addEventListener('play', () => {
                if (toggleBackgroundVideoButton) toggleBackgroundVideoButton.textContent = 'Pause Background Video';
            });
            videoBackgroundElement.addEventListener('pause', () => {
                if (toggleBackgroundVideoButton) toggleBackgroundVideoButton.textContent = 'Play Background Video';
            });
            if (!videoBackgroundElement.paused) {
                toggleBackgroundVideoButton.textContent = 'Pause Background Video';
            } else {
                toggleBackgroundVideoButton.textContent = 'Play Background Video';
            }
        } else {
            console.error("toggleBackgroundVideoButton or videoBackgroundElement not found. Background video play/pause feature might not be fully hooked up.");
        }
    } else {
        console.error("loadBackgroundVideoButton element not found. Background video feature might not be fully hooked up.");
    }

    // Local Background Video Picker Logic
    if (chooseLocalBackgroundVideoButton && localBackgroundVideoInput && videoBackgroundElement) {
        chooseLocalBackgroundVideoButton.addEventListener('click', () => {
            localBackgroundVideoInput.click(); // Trigger the hidden file input
        });

        localBackgroundVideoInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file && file.type.startsWith('video/')) {
                const videoUrl = URL.createObjectURL(file);
                videoBackgroundElement.src = videoUrl;
                videoBackgroundElement.load();

                // Set initial volume from slider or default
                if (backgroundVideoVolumeSlider) {
                    videoBackgroundElement.volume = parseFloat(backgroundVideoVolumeSlider.value);
                } else {
                    videoBackgroundElement.volume = 0.2; // Default if slider not found
                }

                videoBackgroundElement.play().then(() => {
                    backgroundVideoPlayed = true;
                    backgroundVideoStatus.textContent = `Local video "${file.name}" loaded and playing successfully!`;
                    backgroundVideoStatus.style.color = 'var(--color-link)';
                    console.log(`Local background video "${file.name}" started playing.`);
                    if (toggleBackgroundVideoButton) {
                        toggleBackgroundVideoButton.textContent = 'Pause Background Video';
                    }
                }).catch(error => {
                    console.error('Autoplay of local background video failed:', error);
                    backgroundVideoStatus.textContent = `Local video "${file.name}" loaded, but autoplay might be blocked.`;
                    backgroundVideoStatus.style.color = 'var(--color-error)';
                    if (toggleBackgroundVideoButton) {
                        toggleBackgroundVideoButton.textContent = 'Play Background Video';
                    }
                });
            } else {
                backgroundVideoStatus.textContent = 'Please select a valid video file.';
                backgroundVideoStatus.style.color = 'var(--color-error)';
                console.warn("No valid video file selected for local background video.");
            }
        });
    } else {
        console.error("Local background video picker elements not found. Local video feature might not be fully hooked up.");
    }
    // --- END ---

    initializeSlidersUI();
    renderChatHistory();

    // This function will play background video and initialize audio context
    document.addEventListener('click', tryInitAudioAndVideo);
    document.addEventListener('keydown', tryInitAudioAndVideo);
    appendSimliStatusLog("AudioContext and Background Video need a user gesture (click/key) to start.");

    initializeApp(); // This now handles initializeTTSVoicesUI internally

    // Set initial avatar video and attach carousel event listeners
    if (prevAvatarButton) {
        prevAvatarButton.addEventListener('click', () => {
            currentAvatarIndex = (currentAvatarIndex - 1 + avatarVideoSets.length) % avatarVideoSets.length;
            updateAvatarVideos();
        });
    }
    if (nextAvatarButton) {
        nextAvatarButton.addEventListener('click', () => {
            currentAvatarIndex = (currentAvatarIndex + 1) % avatarVideoSets.length;
            updateAvatarVideos();
        });
    }


    if(simliStopButton) {
        simliStopButton.onclick = () => {
            appendSimliStatusLog("Stop Simli button clicked.");
            stopSimliSessionApp();
        };
        simliStopButton.style.display = 'none';
    }

    // Event listener for tab buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.dataset.tab;
            showTab(tabId);
        });
    });

    // Voice Cloner Form Submission
    const voiceClonerForm = document.getElementById('voice-cloner-form');
    if (voiceClonerForm) {
        voiceClonerForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            await handleCreateVoice();
        });
    }


    if (avatarSelect && !avatarSelect.disabled) {
        avatarSelect.addEventListener('change', async (event) => {
            const useAvatar = event.target.value === 'true';
            const currentSimliStopButton = document.getElementById('simli_stop_button');

            if (useAvatar) {
                currentSimliStopButton.style.display = 'inline-block';
                appendSimliStatusLog("Avatar toggled ON. Attempting SimliWebRTC.startSession().");
                if (avatarSelectLabel) avatarSelectLabel.textContent = 'Use Simli Avatar: Yes (Connecting...)';

                if (expLipsyncVideo) {
                    expLipsyncVideo.pause();
                    expLipsyncVideo.style.display = 'none';
                    expLipsyncVideo.loop = false;
                }

                try {
                    if (typeof SimliWebRTC !== 'undefined') await SimliWebRTC.startSession();
                } catch (error) {
                    appendSimliStatusLog(`Error during SimliWebRTC.startSession() from toggle: ${error.message}.`);
                }
            } else {
                appendSimliStatusLog("Avatar toggled OFF. Calling stopSimliSessionApp().");
                stopSimliSessionApp();
            }
        });
    }

    if (micButton) {
        micButton.addEventListener('click', handleMicClick);
    } else {
        console.warn("Mic button not found!");
    }

    document.addEventListener('keydown', (event) => {
        const isTextInputFocused = ['TEXTAREA', 'INPUT'].includes(event.target.tagName.toUpperCase());
        const isButtonOrSelectFocused = ['BUTTON', 'SELECT'].includes(event.target.tagName.toUpperCase());

        // Only trigger mic on spacebar if the chat tab is active AND a text input/button is NOT focused
        const isChatTabActive = !document.getElementById('chat-tab').classList.contains('hidden');
        if (event.code === 'Space' && isChatTabActive) {
            if (!isTextInputFocused && !isButtonOrSelectFocused) {
                event.preventDefault();
                if (!isRecording) {
                    startRecording();
                }
            }
        }
    });

    document.addEventListener('keyup', (event) => {
        const isTextInputFocused = ['TEXTAREA', 'INPUT'].includes(event.target.tagName.toUpperCase());
        const isButtonOrSelectFocused = ['BUTTON', 'SELECT'].includes(event.target.tagName.toUpperCase());

        const isChatTabActive = !document.getElementById('chat-tab').classList.contains('hidden');
        if (event.code === 'Space' && isChatTabActive) {
            if (isRecording && !isTextInputFocused && !isButtonOrSelectFocused) {
                event.preventDefault();
                stopRecording();
            }
        }
    });

    // Initial tab display
    showTab('chat-tab'); // Default to chat tab on load

    const initialUseAvatar = avatarSelect ? avatarSelect.value === 'true' : false;
    const simliMediaContainer = document.getElementById('simli-media-container');
    if (simliMediaContainer) simliMediaContainer.style.display = 'block';
    resetSimliUI(); // This call will now correctly initiate the first avatar video display
    if (initialUseAvatar) {
        appendSimliStatusLog("Initial UI setup - Simli Avatar is ON. Auto-connecting.");
        avatarSelect.value = 'true';
        setTimeout(() => avatarSelect.dispatchEvent(new Event('change')), 100);
    } else {
        appendSimliStatusLog("Initial UI setup - Simli Avatar is OFF.");
        // If Simli is OFF, ensure the placeholder is explicitly visible
        if (simliPlaceholderVideo && simliVideoElement.style.display === 'none') {
            simliPlaceholderVideo.style.display = 'block';
        }
        if (expLipsyncVideo) expLipsyncVideo.style.display = 'none';
    }


    if (appForm) {
        appForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            // Ensure audio context and background video play on first interaction from form submit as well
            await tryInitAudioAndVideo();

            if (!await ensureAudioContext()) { // Re-check after trying to play video
                 alert("Audio system not ready. Please click the page or press a key first.");
                 return;
            }

            const useAvatar = avatarSelect ? avatarSelect.value === 'true' : false;
            const currentMode = modeSelect ? modeSelect.value : 'tts_only';
            let userText = textInput ? textInput.value.trim() : "";

            if (!userText && (currentMode.includes('tts') || currentMode.includes('llm'))) {
                alert("Please enter text or a prompt.");
                if(generateButton) { generateButton.disabled = false; generateButton.textContent = 'Generate';}
                return;
            }

            // Check if ttsVoiceSelect has a valid selection, especially for tts modes
            if ((currentMode.includes('tts') || currentMode.includes('llm_tts')) && ttsVoiceSelect.value === "no_voices_found") {
                alert("No voices are available. Please ensure .pt files are in the voices/ directory and the backend is running correctly.");
                if(generateButton) { generateButton.disabled = false; generateButton.textContent = 'Generate';}
                return;
            }


            if (generateButton) {
                generateButton.disabled = true;
                generateButton.textContent = 'Processing...';
            }

            if (userText) {
                addUserMessageToChat(userText);
                if(currentMode === 'tts_only') textInput.value = '';
            }

            const ttsFunction = useAvatar ? SimliWebRTC.sendTextForTTS : speakTextDirectly;

            try {
                let audioPlaybackCompletionPromise = Promise.resolve();

                if (currentMode === 'tts_only') {
                    if (generateButton) generateButton.textContent = 'Synthesizing...';
                    if (!useAvatar && expLipsyncVideo && simliPlaceholderVideo) {
                        simliPlaceholderVideo.style.display = 'none';
                        expLipsyncVideo.style.display = 'block';
                        expLipsyncVideo.currentTime = 0;
                        expLipsyncVideo.loop = true;
                        expLipsyncVideo.play().catch(e => console.error("Error playing exp.mp4 for TTS only:", e));
                    }
                    audioPlaybackCompletionPromise = ttsFunction(userText, ttsVoiceSelect.value);
                } else if (currentMode === 'llm_only') {
                    if (generateButton) generateButton.textContent = 'Thinking...';
                    await fetchLLMResponse(userText, null);
                    textInput.value = '';
                } else if (currentMode === 'llm_tts') {
                    if (generateButton) generateButton.textContent = 'Thinking & Synthesizing...';
                    if (!useAvatar && expLipsyncVideo && simliPlaceholderVideo) {
                        simliPlaceholderVideo.style.display = 'none';
                        expLipsyncVideo.style.display = 'block';
                        expLipsyncVideo.currentTime = 0;
                        expLipsyncVideo.loop = true;
                        expLipsyncVideo.play().catch(e => console.error("Error playing exp.mp4 for LLM+TTS:", e));
                    }
                    audioPlaybackCompletionPromise = fetchLLMResponse(userText, ttsFunction);
                    textInput.value = '';
                }

                await audioPlaybackCompletionPromise;

            } catch (e) {
                console.error("App: Form Submit Error:", e);
                appendSimliStatusLog(`An error occurred: ${e.message}`);
                addAssistantMessageToChat(`An error occurred: ${e.message}`, false);
            } finally {
                const finalUseAvatar = avatarSelect ? avatarSelect.value === 'true' : false;
                const finalCurrentMode = modeSelect ? modeSelect.value : 'tts_only';
                if (!finalUseAvatar && (finalCurrentMode.includes('tts'))) {
                    if (expLipsyncVideo) {
                        expLipsyncVideo.pause();
                        expLipsyncVideo.style.display = 'none';
                        expLipsyncVideo.loop = false;
                    }
                    if (simliPlaceholderVideo) {
                        simliPlaceholderVideo.style.display = 'block';
                    }
                    appendSimliStatusLog("exp.mp4 hidden, placeholder shown as audio playback completed.");
                }
                console.log("DEBUG: Form submit handler finished. Button re-enabling will be handled by setInterval.");
            }
        });
    }

    appendSimliStatusLog("Application UI Initialized. Select options and enter text.");

    setInterval(() => {
        const isContinuousAudioPlaying = isPlayingContinuousAudio || directTTSAudioBufferQueue.length > 0;

        // Only re-enable generate button if current tab is chat
        const isChatTabActive = !document.getElementById('chat-tab').classList.contains('hidden');
        if (isChatTabActive && !currentLLMStreamController && generateButton && generateButton.disabled) {
            if (!isContinuousAudioPlaying) {
                generateButton.disabled = false;
                generateButton.textContent = 'Generate';
                appendSimliStatusLog("Generate button re-enabled as all audio processed/played and LLM stream finished.");
            }
        }
        // Ensure logs tab is always scrolled to bottom if active
        if (document.getElementById('logs-tab') && !document.getElementById('logs-tab').classList.contains('hidden') && simliWebRTCLog) {
            simliWebRTCLog.scrollTop = simliWebRTCLog.scrollHeight;
        }
    }, 500);
});
// --- End DOMContentLoaded ---
