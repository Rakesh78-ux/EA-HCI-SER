/**
 * Audio Recording and Processing Module
 * Handles microphone access, recording, and WebSocket streaming
 */

class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioContext = null;
        this.analyser = null;
        this.microphone = null;
        this.stream = null;
        this.isRecording = false;
        this.isStreaming = false;
        this.recordedChunks = [];
        
        // WebSocket for real-time streaming
        this.websocket = null;
        this.streamProcessor = null;
        
        // Audio processing settings
        this.sampleRate = 22050;
        this.bufferSize = 4096;
        
        // Event callbacks
        this.onDataCallback = null;
        this.onLevelCallback = null;
        this.onErrorCallback = null;
        this.onStreamingCallback = null;
    }

    /**
     * Initialize audio context and request microphone access
     */
    async initialize() {
        try {
            // Request microphone access
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: this.sampleRate,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate
            });

            // Create analyser for audio level monitoring
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            
            // Connect microphone to analyser
            this.microphone = this.audioContext.createMediaStreamSource(this.stream);
            this.microphone.connect(this.analyser);

            // Initialize media recorder
            this.mediaRecorder = new MediaRecorder(this.stream, {
                mimeType: this.getSupportedMimeType()
            });

            this.setupMediaRecorderEvents();
            
            console.log('Audio recorder initialized successfully');
            return true;

        } catch (error) {
            console.error('Failed to initialize audio recorder:', error);
            if (this.onErrorCallback) {
                this.onErrorCallback('Microphone access denied or not available');
            }
            return false;
        }
    }

    /**
     * Get supported MIME type for MediaRecorder
     */
    getSupportedMimeType() {
        const types = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/mp4',
            'audio/wav'
        ];

        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) {
                return type;
            }
        }
        return 'audio/webm'; // fallback
    }

    /**
     * Setup MediaRecorder event handlers
     */
    setupMediaRecorderEvents() {
        this.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                this.recordedChunks.push(event.data);
            }
        };

        this.mediaRecorder.onstop = () => {
            if (this.recordedChunks.length > 0) {
                const blob = new Blob(this.recordedChunks, {
                    type: this.mediaRecorder.mimeType
                });
                
                if (this.onDataCallback) {
                    this.onDataCallback(blob);
                }
            }
        };

        this.mediaRecorder.onerror = (event) => {
            console.error('MediaRecorder error:', event.error);
            if (this.onErrorCallback) {
                this.onErrorCallback('Recording failed: ' + event.error.message);
            }
        };
    }

    /**
     * Start recording audio
     */
    startRecording() {
        try {
            if (!this.mediaRecorder || this.mediaRecorder.state !== 'inactive') {
                throw new Error('MediaRecorder not ready');
            }

            this.recordedChunks = [];
            this.mediaRecorder.start(100); // Collect data every 100ms
            this.isRecording = true;
            
            // Start audio level monitoring
            this.startLevelMonitoring();
            
            console.log('Recording started');
            return true;

        } catch (error) {
            console.error('Failed to start recording:', error);
            if (this.onErrorCallback) {
                this.onErrorCallback('Failed to start recording: ' + error.message);
            }
            return false;
        }
    }

    /**
     * Stop recording audio
     */
    stopRecording() {
        try {
            if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
                this.mediaRecorder.stop();
                this.isRecording = false;
                
                // Stop audio level monitoring
                this.stopLevelMonitoring();
                
                console.log('Recording stopped');
                return true;
            }
            return false;

        } catch (error) {
            console.error('Failed to stop recording:', error);
            if (this.onErrorCallback) {
                this.onErrorCallback('Failed to stop recording: ' + error.message);
            }
            return false;
        }
    }

    /**
     * Start real-time audio streaming via WebSocket
     */
    async startStreaming() {
        try {
            if (this.isStreaming) {
                return false;
            }

            // Connect WebSocket
            await this.connectWebSocket();
            
            if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
                throw new Error('WebSocket connection failed');
            }

            // Create audio processor for streaming
            this.createStreamProcessor();
            
            this.isStreaming = true;
            console.log('Audio streaming started');
            return true;

        } catch (error) {
            console.error('Failed to start streaming:', error);
            if (this.onErrorCallback) {
                this.onErrorCallback('Failed to start streaming: ' + error.message);
            }
            return false;
        }
    }

    /**
     * Stop real-time audio streaming
     */
    stopStreaming() {
        try {
            this.isStreaming = false;
            
            // Disconnect audio processor
            if (this.streamProcessor) {
                this.streamProcessor.disconnect();
                this.streamProcessor = null;
            }
            
            // Close WebSocket
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.close();
            }
            
            console.log('Audio streaming stopped');
            return true;

        } catch (error) {
            console.error('Failed to stop streaming:', error);
            return false;
        }
    }

    /**
     * Connect WebSocket for real-time streaming
     */
    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            try {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/audio`;
                
                this.websocket = new WebSocket(wsUrl);
                
                this.websocket.onopen = () => {
                    console.log('WebSocket connected');
                    resolve();
                };
                
                this.websocket.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        if (this.onStreamingCallback) {
                            this.onStreamingCallback(data);
                        }
                    } catch (error) {
                        console.error('Error parsing WebSocket message:', error);
                    }
                };
                
                this.websocket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    reject(error);
                };
                
                this.websocket.onclose = (event) => {
                    console.log('WebSocket disconnected:', event.code, event.reason);
                    this.isStreaming = false;
                };
                
                // Timeout for connection
                setTimeout(() => {
                    if (this.websocket.readyState !== WebSocket.OPEN) {
                        reject(new Error('WebSocket connection timeout'));
                    }
                }, 5000);

            } catch (error) {
                reject(error);
            }
        });
    }

    /**
     * Create audio processor for real-time streaming
     */
    createStreamProcessor() {
        try {
            // Use ScriptProcessorNode for older browsers or AudioWorklet for modern browsers
            if (this.audioContext.createScriptProcessor) {
                this.streamProcessor = this.audioContext.createScriptProcessor(this.bufferSize, 1, 1);
                
                this.streamProcessor.onaudioprocess = (event) => {
                    if (!this.isStreaming) return;
                    
                    const inputBuffer = event.inputBuffer.getChannelData(0);
                    this.sendAudioData(inputBuffer);
                };
                
                // Connect to audio graph
                this.microphone.connect(this.streamProcessor);
                this.streamProcessor.connect(this.audioContext.destination);
                
            } else {
                // Fallback for browsers without ScriptProcessorNode
                console.warn('ScriptProcessorNode not supported, using fallback method');
                this.createFallbackStreamProcessor();
            }

        } catch (error) {
            console.error('Failed to create stream processor:', error);
            throw error;
        }
    }

    /**
     * Fallback streaming method using MediaRecorder
     */
    createFallbackStreamProcessor() {
        const mediaRecorder = new MediaRecorder(this.stream, {
            mimeType: this.getSupportedMimeType()
        });

        let streamChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0 && this.isStreaming) {
                streamChunks.push(event.data);
                
                // Send accumulated chunks every few data events
                if (streamChunks.length >= 3) {
                    const blob = new Blob(streamChunks, { type: mediaRecorder.mimeType });
                    this.sendAudioBlob(blob);
                    streamChunks = [];
                }
            }
        };

        mediaRecorder.start(100); // Collect data every 100ms
        this.streamProcessor = mediaRecorder;
    }

    /**
     * Send audio data via WebSocket
     */
    sendAudioData(audioData) {
        if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
            return;
        }

        try {
            // Convert Float32Array to ArrayBuffer
            const buffer = audioData.buffer.slice(
                audioData.byteOffset,
                audioData.byteOffset + audioData.byteLength
            );
            
            this.websocket.send(buffer);

        } catch (error) {
            console.error('Error sending audio data:', error);
        }
    }

    /**
     * Send audio blob via WebSocket (fallback method)
     */
    async sendAudioBlob(blob) {
        if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
            return;
        }

        try {
            const arrayBuffer = await blob.arrayBuffer();
            this.websocket.send(arrayBuffer);

        } catch (error) {
            console.error('Error sending audio blob:', error);
        }
    }

    /**
     * Start monitoring audio levels for visual feedback
     */
    startLevelMonitoring() {
        if (!this.analyser) return;

        const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
        
        const updateLevel = () => {
            if (!this.isRecording && !this.isStreaming) return;
            
            this.analyser.getByteFrequencyData(dataArray);
            
            // Calculate average volume level
            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) {
                sum += dataArray[i];
            }
            const average = sum / dataArray.length;
            const level = average / 255; // Normalize to 0-1
            
            if (this.onLevelCallback) {
                this.onLevelCallback(level);
            }
            
            // Continue monitoring
            requestAnimationFrame(updateLevel);
        };
        
        updateLevel();
    }

    /**
     * Stop monitoring audio levels
     */
    stopLevelMonitoring() {
        // Level monitoring will stop automatically when isRecording/isStreaming becomes false
    }

    /**
     * Set callback for recording data
     */
    setDataCallback(callback) {
        this.onDataCallback = callback;
    }

    /**
     * Set callback for audio level updates
     */
    setLevelCallback(callback) {
        this.onLevelCallback = callback;
    }

    /**
     * Set callback for errors
     */
    setErrorCallback(callback) {
        this.onErrorCallback = callback;
    }

    /**
     * Set callback for streaming data
     */
    setStreamingCallback(callback) {
        this.onStreamingCallback = callback;
    }

    /**
     * Clean up resources
     */
    cleanup() {
        try {
            this.stopRecording();
            this.stopStreaming();
            
            if (this.streamProcessor) {
                this.streamProcessor.disconnect();
            }
            
            if (this.microphone) {
                this.microphone.disconnect();
            }
            
            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
            }
            
            if (this.audioContext && this.audioContext.state !== 'closed') {
                this.audioContext.close();
            }
            
            console.log('Audio recorder cleaned up');

        } catch (error) {
            console.error('Error during cleanup:', error);
        }
    }

    /**
     * Check if the recorder is ready
     */
    isReady() {
        return this.mediaRecorder && this.audioContext && this.stream;
    }

    /**
     * Get current recording state
     */
    getState() {
        return {
            isRecording: this.isRecording,
            isStreaming: this.isStreaming,
            isReady: this.isReady(),
            mediaRecorderState: this.mediaRecorder ? this.mediaRecorder.state : 'unavailable'
        };
    }
}

// Export for use in other modules
window.AudioRecorder = AudioRecorder;
