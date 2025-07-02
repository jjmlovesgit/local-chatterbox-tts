// static/simli_webrtc.js (Corrected for true sentence-by-sentence streaming, ADDED RESET LOGGING)

const SimliWebRTC = (() => {
    let rtcPeerConnection = null;
    let simliSignalingWS = null;
    let localStream = null;
    let currentSimliSessionToken = null;
    
    let simliAudioDataQueue = [];
    let isSendingAudioToSimli = false;

    // --- FIX: New variables to manage sentence-by-sentence completion ---
    let currentUpstreamFetchDone = false; 
    let currentAudioSendCompletionResolver = null; // Resolves the promise in sendTextForTTS

    let config = {
        SIMLI_API_KEY_CLIENT: '',
        SIMLI_FACE_ID_CLIENT: '',
        // MODIFIED: Point to your backend's endpoint for session token
        simliApiSessionUrl: '/api/get_simli_session_token', 
        simliWsUrl: 'wss://api.simli.ai/startWebRTCSession',
        uiCallbacks: {
            appendSimliStatusLog: (msg) => console.log('[SimliWebRTC-DefaultLog]', msg),
            updateIceGatheringState: (state) => {}, // Default empty functions
            updateIceConnectionState: (state) => {},
            updateSignalingState: (state) => {},
            updateOfferSdp: (sdp) => {},
            updateAnswerSdp: (sdp) => {},
            onConnectionEstablished: () => {},
            onConnectionFailed: () => {},
            onSessionStoppedByServer: () => {},
            onAudioProcessingComplete: (success) => {} 
        }
    };

    function _resetConnectionState() {
        config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Attempting to reset connection state...");

        // 1. Stop local media tracks
        if (localStream) {
            config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Stopping local media tracks.");
            localStream.getTracks().forEach(track => track.stop());
            localStream = null;
        } else {
            config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: No local stream to stop.");
        }

        // 2. Close WebSocket connection
        if (simliSignalingWS) {
            config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: Closing WebSocket (current state: ${simliSignalingWS.readyState}).`);
            simliSignalingWS.onopen = null;
            simliSignalingWS.onmessage = null;
            simliSignalingWS.onerror = null;
            simliSignalingWS.onclose = null;
            if (simliSignalingWS.readyState === WebSocket.OPEN || simliSignalingWS.readyState === WebSocket.CONNECTING) {
                 simliSignalingWS.close(1000, "User initiated disconnect"); // 1000 is normal closure
                 config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: WebSocket close initiated.");
            }
        } else {
            config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: No WebSocket to close.");
        }
        simliSignalingWS = null;

        // 3. Close RTCPeerConnection
        if (rtcPeerConnection) {
            config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: Closing RTCPeerConnection (signalingState: ${rtcPeerConnection.signalingState}, iceConnectionState: ${rtcPeerConnection.iceConnectionState}).`);
            try {
                // Attempt to stop all transceivers/tracks within the peer connection for clean shutdown
                if (rtcPeerConnection.getTransceivers) { 
                    rtcPeerConnection.getTransceivers().forEach(t => { 
                        if(t.stop) {
                            t.stop(); 
                            config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: Stopped transceiver: ${t.mid || 'no mid'}`);
                        }
                        if (t.sender && t.sender.track) {
                            t.sender.track.stop();
                            config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: Stopped sender track: ${t.sender.track.kind}`);
                        }
                        if (t.receiver && t.receiver.track) {
                            t.receiver.track.stop();
                            config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: Stopped receiver track: ${t.receiver.track.kind}`);
                        }
                    }); 
                }
            } catch (e) { config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: Error stopping transceivers: ${e.message}`); }
            
            rtcPeerConnection.onicecandidate = null;
            rtcPeerConnection.onicegatheringstatechange = null;
            rtcPeerConnection.oniceconnectionstatechange = null;
            rtcPeerConnection.onsignalingstatechange = null;
            rtcPeerConnection.ontrack = null;
            // Only call close if it's not already in a closed state
            if (rtcPeerConnection.signalingState !== 'closed') {
                rtcPeerConnection.close();
                config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: RTCPeerConnection close initiated.");
            }
        } else {
            config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: No RTCPeerConnection to close.");
        }
        rtcPeerConnection = null;

        // 4. Reset internal state variables
        currentSimliSessionToken = null;
        simliAudioDataQueue = [];
        isSendingAudioToSimli = false;
        currentUpstreamFetchDone = false;
        if (currentAudioSendCompletionResolver) {
            config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Resolving pending audio send promise (likely due to reset).");
            currentAudioSendCompletionResolver(false); // Resolve any pending promises as failed
            currentAudioSendCompletionResolver = null;
        }
        config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Connection state reset complete.");
    }

    // This function remains the same
    function _createSimliPeerConnectionInternal() {
        config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Creating RTCPeerConnection...");
        const pcConfig = { sdpSemantics: "unified-plan", iceServers: [{ urls: ["stun:stun.l.google.com:19302"] }] };
        let pc = new RTCPeerConnection(pcConfig);

        pc.onicegatheringstatechange = () => {
            if (!pc) return;
            config.uiCallbacks.updateIceGatheringState(pc.iceGatheringState);
        };
        pc.oniceconnectionstatechange = () => {
            if (!pc) return;
            config.uiCallbacks.updateIceConnectionState(pc.iceConnectionState);
            if (pc.iceConnectionState === 'disconnected' || pc.iceConnectionState === 'failed' || pc.iceConnectionState === 'closed') {
                config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: ICE Connection State changed to: ${pc.iceConnectionState}.`);
                // Only trigger onConnectionFailed if the PC isn't already explicitly closed
                if (rtcPeerConnection && rtcPeerConnection.signalingState !== 'closed') { 
                    config.uiCallbacks.onConnectionFailed(); 
                }
            } else if (pc.iceConnectionState === 'connected' || pc.iceConnectionState === 'completed') {
                config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: ICE Connection State changed to: ${pc.iceConnectionState}. Connection established.`);
                config.uiCallbacks.onConnectionEstablished();
                if (!isSendingAudioToSimli && simliAudioDataQueue.length > 0) {
                    config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: ICE connected, starting queued audio send.");
                    _sendQueuedAudioToSimliInternal();
                }
            }
        };
        pc.onsignalingstatechange = () => {
            if (!pc) return;
            config.uiCallbacks.updateSignalingState(pc.signalingState);
            config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: Signaling State changed to: ${pc.signalingState}.`);
        };
        pc.ontrack = (evt) => {
            config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: Received track of kind: ${evt.track.kind}.`);
            let targetStream = evt.streams && evt.streams[0] ? evt.streams[0] : new MediaStream([evt.track]);
            if (evt.track.kind === 'video') config.uiCallbacks.setSimliVideoSrcObject(targetStream);
            else if (evt.track.kind === 'audio') config.uiCallbacks.setSimliAudioSrcObject(targetStream);
        };
        return pc;
    }
    
    // Unchanged
    function _openWebSocketAndSignalInternal(pc) {
        return new Promise((resolve, reject) => {
            config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Opening Signaling WebSocket...");
            simliSignalingWS = new WebSocket(config.simliWsUrl);
            let promiseHandled = false;

            simliSignalingWS.onopen = () => {
                config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: WebSocket opened.");
                if (!pc || !pc.localDescription || !currentSimliSessionToken) {
                    if (!promiseHandled) { promiseHandled = true; reject(new Error("PC, localDescription, or session token missing for WS send."));}
                    return;
                }
                
                const offer = pc.localDescription;
                simliSignalingWS.send(JSON.stringify({ sdp: offer.sdp, type: offer.type }));
                config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Sent SDP offer via WebSocket.");
                config.uiCallbacks.updateOfferSdp(pc.localDescription.sdp); // Update UI callback
                
                setTimeout(() => {
                    if (simliSignalingWS && simliSignalingWS.readyState === WebSocket.OPEN) {
                        simliSignalingWS.send(currentSimliSessionToken);
                        config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Sent session token via WebSocket.");
                    } else if (!promiseHandled) {
                        promiseHandled = true; reject(new Error("WebSocket not open for sending token."));
                    }
                }, 100);
            };
            simliSignalingWS.onmessage = async (evt) => {
                config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: WebSocket message received: ${evt.data.substring(0, 50)}...`);
                if (evt.data === "START") {
                    config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Received 'START' signal.");
                    if (!promiseHandled) { promiseHandled = true; resolve(); }
                    await _sendQueuedAudioToSimliInternal(); 
                    return;
                }
                if (evt.data === "STOP") {
                    config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Received 'STOP' signal from server. Resetting connection.");
                    _resetConnectionState(); 
                    config.uiCallbacks.onSessionStoppedByServer(); 
                    return;
                }
                try {
                    const message = JSON.parse(evt.data);
                    if (message.type === 'answer' && pc.signalingState !== 'closed') {
                        config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Received SDP answer.");
                        await pc.setRemoteDescription(new RTCSessionDescription(message));
                        config.uiCallbacks.updateAnswerSdp(message.sdp); // Update UI callback
                    } else if (message.candidate && pc.signalingState !== 'closed') {
                        config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Received ICE candidate.");
                        await pc.addIceCandidate(new RTCIceCandidate(message.candidate));
                    }
                } catch (e) { 
                    config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: Non-JSON WebSocket message or error parsing: ${e.message}`);
                    /* Non-JSON messages */ 
                }
            };
            simliSignalingWS.onerror = (errEvent) => {
                console.error("SimliWebRTC: Signaling WebSocket error:", errEvent);
                config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: Signaling WebSocket error: ${errEvent.message || errEvent.type}`);
                if (!promiseHandled) { promiseHandled = true; reject(new Error("WebSocket error")); }
            };
            simliSignalingWS.onclose = (closeEvent) => {
                config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: WebSocket closed. Code: ${closeEvent.code}, Reason: ${closeEvent.reason || 'N/A'}, Clean: ${closeEvent.wasClean}.`);
            };
        });
    }

    // Unchanged
    async function _negotiateAndConnectSimliInternal(pc) {
        if (!pc) throw new Error("PeerConnection not available for negotiation.");
        config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Initiating SDP negotiation.");
        
        const offer = await pc.createOffer({ offerToReceiveAudio: true, offerToReceiveVideo: true });
        await pc.setLocalDescription(offer);
        config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Created and set local SDP offer.");
        
        // Simplified ICE check
        config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Waiting for ICE candidates (briefly).");
        await new Promise(resolve => setTimeout(resolve, 1500)); 

        await _openWebSocketAndSignalInternal(pc);
    }

    async function _fetchAndSendTTSAudioToSimliInternal(text, voice) {
        currentUpstreamFetchDone = false;
        if (!text || !text.trim()) {
            config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: No text for TTS fetch. Resolving as complete.");
            if (currentAudioSendCompletionResolver) {
                currentAudioSendCompletionResolver(true); // Successfully did nothing
                currentAudioSendCompletionResolver = null;
            }
            return;
        }

        const ttsParams = config.uiCallbacks.getTtsParams();
        const payload = { text, voice_id: voice, ...ttsParams };
        config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: Fetching TTS audio stream for Simli input for text: "${text.substring(0, 30)}..." Voice: ${voice}`);

        try {
            const response = await fetch('/api/get_tts_audio_for_simli_input', { 
                method: 'POST', 
                headers: { 'Content-Type': 'application/json' }, 
                body: JSON.stringify(payload) 
            });

            if (!response.ok || !response.body) { 
                const errorText = await response.text(); 
                throw new Error(`TTS stream fetch failed: ${response.status} ${errorText}`); 
            }

            const reader = response.body.getReader();
            config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Started streaming TTS audio from backend.");

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    currentUpstreamFetchDone = true;
                    config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Finished fetching all TTS audio chunks from backend.");
                    if (!isSendingAudioToSimli) {
                        config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: All TTS fetched, send queue idle. Attempting to finalize audio send.");
                        _sendQueuedAudioToSimliInternal(); // Ensure finalization if queue becomes empty
                    }
                    break;
                }
                if (value && value.byteLength > 0) {
                    simliAudioDataQueue.push(value);
                    if (!isSendingAudioToSimli) {
                        config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: New audio chunk received, starting send queue.");
                        _sendQueuedAudioToSimliInternal();
                    }
                }
            }
        } catch (e) {
            console.error(`SimliWebRTC: Error fetching/streaming TTS: ${e.message}`);
            config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: Error fetching/streaming TTS: ${e.message}.`);
            currentUpstreamFetchDone = true;
            if (currentAudioSendCompletionResolver) {
                currentAudioSendCompletionResolver(false); // Signal failure
                currentAudioSendCompletionResolver = null;
            }
        }
    }

    async function _sendQueuedAudioToSimliInternal() {
        if (isSendingAudioToSimli) return;
        
        // --- FIX: Check for completion and resolve the promise ---
        if (simliAudioDataQueue.length === 0) {
            if (currentUpstreamFetchDone) {
                config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Queue empty & upstream fetch complete. Audio sending finalized successfully.");
                if (currentAudioSendCompletionResolver) {
                    currentAudioSendCompletionResolver(true); // Signal success
                    currentAudioSendCompletionResolver = null;
                }
            }
            return;
        }

        const canSend = rtcPeerConnection && simliSignalingWS && simliSignalingWS.readyState === WebSocket.OPEN &&
                         (rtcPeerConnection.iceConnectionState === 'connected' || rtcPeerConnection.iceConnectionState === 'completed');
        
        if (!canSend) {
            config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Cannot send audio. Connection not ready.");
            return;
        }

        isSendingAudioToSimli = true;
        let sendLoopError = null;

        try {
            while (simliAudioDataQueue.length > 0) {
                let segment = simliAudioDataQueue.shift();
                if (!segment || segment.byteLength === 0) continue;
                if (!simliSignalingWS || simliSignalingWS.readyState !== WebSocket.OPEN) {
                    simliAudioDataQueue.unshift(segment); // Put it back if WS closed
                    config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: WebSocket closed during audio send loop. Stopping send.");
                    throw new Error("WebSocket closed during send");
                }
                
                config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: Sending audio segment of ${segment.byteLength} bytes.`);
                simliSignalingWS.send(segment);
                await new Promise(r => setTimeout(r, 20)); // Small delay to prevent overwhelming network/CPU
            }
        } catch (e) {
            console.error(`SimliWebRTC: Error during sending audio: ${e.message}`);
            config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: Error during sending audio: ${e.message}.`);
            sendLoopError = e;
        } finally {
            isSendingAudioToSimli = false;
            if (sendLoopError) {
                if (currentAudioSendCompletionResolver) {
                    config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Audio send loop ended with error. Resolving audio send promise as failed.");
                    currentAudioSendCompletionResolver(false);
                    currentAudioSendCompletionResolver = null;
                }
            } else if (simliAudioDataQueue.length === 0 && currentUpstreamFetchDone) {
                 if (currentAudioSendCompletionResolver) {
                    config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Audio send loop finished, queue empty, fetch done. Resolving audio send promise as success.");
                    currentAudioSendCompletionResolver(true);
                    currentAudioSendCompletionResolver = null;
                 }
            } else if (simliAudioDataQueue.length > 0) {
                config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Audio send loop finished, but queue still has data. Rescheduling send.");
                setTimeout(() => _sendQueuedAudioToSimliInternal(), 0); // Continue processing if more data arrived
            }
        }
    }
    
    function init(userConfig) {
        config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Initializing with provided config.");
        Object.assign(config, userConfig);
        // Deep merge uiCallbacks to preserve defaults if not overridden
        if (userConfig.uiCallbacks) {
            config.uiCallbacks = { ...config.uiCallbacks, ...userConfig.uiCallbacks };
        }
    }

    async function startSession() {
        if (rtcPeerConnection && rtcPeerConnection.signalingState !== 'closed') {
            config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Session already active. Not restarting.");
            return;
        }
        config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Starting new session...");
        _resetConnectionState(); // Ensure a clean slate

        try {
            config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Requesting session token from *your backend*.");
            // Pass faceId from client config, but the API key will be handled by your backend
            const payload = { 
                face_id: config.SIMLI_FACE_ID_CLIENT, // Use face ID from client config
                // Do NOT send the API key from the client here. Your backend will add SIMLI_API_KEY_SERVER.
            };
            
            // Call your backend's endpoint
            const response = await fetch(config.simliApiSessionUrl, { 
                method: "POST", 
                body: JSON.stringify(payload), 
                headers: { "Content-Type": "application/json" } 
            });
            const resJSON = await response.json();
            if (!response.ok || !resJSON.token) { // Your backend returns 'token'
                throw new Error(`Failed to get session token from backend: ${resJSON.detail || resJSON.message || response.statusText}`);
            }
            currentSimliSessionToken = resJSON.token; // Use the token returned by your backend
            
            // simliWsUrl is still hardcoded in config for now, but could also be returned by backend if dynamic
            
            config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: Session token received from backend: ${currentSimliSessionToken.substring(0, 10)}...`);
            
            rtcPeerConnection = _createSimliPeerConnectionInternal();
            
            try {
                config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Requesting local media (audio/video).");
                localStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
                config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Local media stream acquired.");
                localStream.getTracks().forEach(track => {
                    if (rtcPeerConnection) {
                        rtcPeerConnection.addTrack(track, localStream);
                        config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: Added local track: ${track.kind}`);
                    }
                });
            } catch (mediaErr) {
                config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: getUserMedia failed: ${mediaErr.message}. Proceeding with recvonly transceivers.`);
                console.warn(`SimliWebRTC: getUserMedia failed: ${mediaErr.message}. Proceeding with recvonly transceivers.`);
                // If getUserMedia fails (e.g., no webcam/mic), add recvonly transceivers to still receive video/audio
                if (rtcPeerConnection && rtcPeerConnection.addTransceiver) {
                    rtcPeerConnection.addTransceiver('audio', { direction: 'recvonly' });
                    rtcPeerConnection.addTransceiver('video', { direction: 'recvonly' });
                }
            }
            config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Negotiating SDP and connecting WebSocket.");
            await _negotiateAndConnectSimliInternal(rtcPeerConnection);
        } catch (e) {
            console.error(`CRITICAL Error during startSession: ${e.message}`);
            config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: CRITICAL Error during startSession: ${e.message}. Resetting connection.`);
            _resetConnectionState();
            config.uiCallbacks.onConnectionFailed();
            throw e;
        }
    }

    function stopSession() {
        config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: stopSession() called (User initiated).");
        _resetConnectionState();
    }

    // --- FIX: This function now returns a promise that resolves on completion ---
    async function sendTextForTTS(text, voice) {
        if (!isSessionReady()) {
            config.uiCallbacks.appendSimliStatusLog("SimliWebRTC: Session not ready for TTS. Aborting TTS send.");
            return Promise.resolve(false); 
        }

        const completionPromise = new Promise(resolve => {
            currentAudioSendCompletionResolver = resolve;
        });
        config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: sendTextForTTS initiated for "${text.substring(0,20)}..."`);

        // This kicks off the process. It does not need to be awaited here.
        _fetchAndSendTTSAudioToSimliInternal(text, voice);

        // Await the promise that will be resolved by the send loop.
        const success = await completionPromise;
        config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: sendTextForTTS completed with success: ${success}`);
        
        // This callback is for the main UI button state
        config.uiCallbacks.onAudioProcessingComplete(success);

        return success;
    }
    
    function isSessionReady() {
        const ready = rtcPeerConnection && simliSignalingWS && simliSignalingWS.readyState === WebSocket.OPEN &&
                      (rtcPeerConnection.iceConnectionState === 'connected' || rtcPeerConnection.iceConnectionState === 'completed');
        // config.uiCallbacks.appendSimliStatusLog(`SimliWebRTC: isSessionReady: ${ready}. PC state: ${rtcPeerConnection?.iceConnectionState}, WS state: ${simliSignalingWS?.readyState}`);
        return ready;
    }

    return {
        init,
        startSession,
        stopSession,
        sendTextForTTS,
        isSessionReady
    };
})();

