/**
 * Speech Emotion Recognition Web Application
 * Main application logic and UI handling
 */

class SERApp {
    constructor() {
        this.audioRecorder = null;
        this.emotionChart = null;
        this.realtimeChart = null;
        this.currentFile = null;
        this.isSystemReady = false;
        
        // Emotion icons mapping
        this.emotionIcons = {
            neutral: '😐',
            happy: '😊',
            sad: '😢',
            angry: '😠',
            fear: '😨',
            disgust: '🤢',
            surprise: '😲'
        };
        
        // Color scheme for emotions
        this.emotionColors = {
            neutral: '#6c757d',
            happy: '#ffc107',
            sad: '#007bff',
            angry: '#dc3545',
            fear: '#6f42c1',
            disgust: '#20c997',
            surprise: '#fd7e14'
        };
        
        this.realtimeData = [];
        this.maxRealtimePoints = 20;

        // Speech-to-text (browser Web Speech API)
        this.speechRecognition = null;
        this.isTranscribing = false;
        this.transcriptText = '';
        this.sttSupported = false;
    }

    /**
     * Initialize the application
     */
    async init() {
        try {
            // Initialize UI components
            this.initializeUI();
            
            // Initialize speech-to-text (if supported)
            this.initializeSpeechToText();
            
            // Check system health
            await this.checkSystemHealth();
            
            // Initialize audio recorder
            this.audioRecorder = new AudioRecorder();
            
            // Set up event handlers
            this.setupEventHandlers();
            
            // Initialize charts
            this.initializeCharts();
            
            console.log('SER App initialized successfully');

        } catch (error) {
            console.error('Failed to initialize app:', error);
            this.showError('Application initialization failed: ' + error.message);
        }
    }

    /**
     * Initialize UI components
     */
    initializeUI() {
        // Update loading status
        this.updateSystemStatus('Initializing...', 'loading');
        
        // Set up file drag and drop
        this.setupFileDragAndDrop();
        
        // Initialize feather icons
        if (typeof feather !== 'undefined') {
            feather.replace();
        }
    }

    /**
     * Initialize browser speech-to-text (Web Speech API)
     */
    initializeSpeechToText() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const statusEl = document.getElementById('transcript-status');

        if (!SpeechRecognition) {
            this.sttSupported = false;
            if (statusEl) {
                statusEl.classList.add('unsupported');
                statusEl.textContent = 'Speech-to-text is not supported in this browser. Please try Chrome or Edge.';
            }
            return;
        }

        this.sttSupported = true;
        this.speechRecognition = new SpeechRecognition();
        this.speechRecognition.continuous = true;
        this.speechRecognition.interimResults = true;
        this.speechRecognition.lang = 'en-US';

        this.speechRecognition.onstart = () => {
            this.isTranscribing = true;
            if (statusEl) {
                statusEl.classList.remove('unsupported');
                statusEl.classList.add('active');
                statusEl.textContent = 'Listening... recognized text will appear below while you speak.';
            }
        };

        this.speechRecognition.onend = () => {
            // If we still expect to be transcribing (e.g., recording still on), restart
            if (this.isTranscribing && this.sttSupported && this.speechRecognition) {
                try {
                    this.speechRecognition.start();
                } catch (e) {
                    // Ignore if already started or cannot restart
                }
            } else if (statusEl) {
                statusEl.classList.remove('active');
                statusEl.textContent = 'Start a live recording to see the recognized text of your speech.';
            }
        };

        this.speechRecognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
        };

        this.speechRecognition.onresult = (event) => {
            let newText = '';
            for (let i = event.resultIndex; i < event.results.length; i++) {
                newText += event.results[i][0].transcript + ' ';
            }
            if (newText.trim().length === 0) {
                return;
            }

            this.transcriptText = (this.transcriptText + ' ' + newText).trim();
            const textEl = document.getElementById('transcript-text');
            if (textEl) {
                textEl.textContent = this.transcriptText;
                textEl.classList.remove('transcript-text-placeholder');
            }
        };
    }

    /**
     * Check system health and model status
     */
    async checkSystemHealth() {
        try {
            const response = await fetch('/api/health');
            const health = await response.json();
            
            if (health.status === 'healthy') {
                this.isSystemReady = true;
                this.updateSystemStatus('Ready', 'healthy');
                this.updateModelStatus('Loaded');
            } else {
                this.updateSystemStatus('Error: ' + health.message, 'error');
                this.updateModelStatus('Failed to Load');
            }

        } catch (error) {
            console.error('Health check failed:', error);
            this.updateSystemStatus('Connection Error', 'error');
            this.updateModelStatus('Unknown');
        }
    }

    /**
     * Update system status display
     */
    updateSystemStatus(status, type = 'normal') {
        const statusElement = document.getElementById('system-status');
        if (statusElement) {
            statusElement.textContent = status;
            statusElement.className = `status-value ${type}`;
        }
    }

    /**
     * Update model status display
     */
    updateModelStatus(status) {
        const statusElement = document.getElementById('model-status');
        if (statusElement) {
            statusElement.textContent = status;
        }
    }

    /**
     * Set up event handlers
     */
    setupEventHandlers() {
        // Recording controls
        document.getElementById('start-recording')?.addEventListener('click', () => this.startRecording());
        document.getElementById('stop-recording')?.addEventListener('click', () => this.stopRecording());
        document.getElementById('clear-results')?.addEventListener('click', () => this.clearResults());
        
        // File upload
        document.getElementById('file-input')?.addEventListener('change', (e) => this.handleFileSelect(e));
        document.getElementById('analyze-file')?.addEventListener('click', () => this.analyzeFile());
        
        // Streaming controls
        document.getElementById('start-streaming')?.addEventListener('click', () => this.startStreaming());
        document.getElementById('stop-streaming')?.addEventListener('click', () => this.stopStreaming());
        
        // Modal controls
        document.getElementById('close-error-modal')?.addEventListener('click', () => this.hideError());
        document.getElementById('error-ok-btn')?.addEventListener('click', () => this.hideError());
        
        // Audio recorder callbacks
        if (this.audioRecorder) {
            this.audioRecorder.setDataCallback((blob) => this.handleRecordingData(blob));
            this.audioRecorder.setLevelCallback((level) => this.updateAudioLevel(level));
            this.audioRecorder.setErrorCallback((error) => this.showError(error));
            this.audioRecorder.setStreamingCallback((data) => this.handleStreamingData(data));
        }
    }

    /**
     * Set up file drag and drop functionality
     */
    setupFileDragAndDrop() {
        const uploadArea = document.getElementById('upload-area');
        if (!uploadArea) return;

        uploadArea.addEventListener('click', () => {
            document.getElementById('file-input')?.click();
        });

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelection(files[0]);
            }
        });
    }

    /**
     * Handle file selection
     */
    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.handleFileSelection(file);
        }
    }

    /**
     * Process selected file
     */
    handleFileSelection(file) {
        // Validate file type
        const allowedTypes = ['.wav', '.mp3', '.m4a', '.flac'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!allowedTypes.includes(fileExtension)) {
            this.showError('Unsupported file format. Please use WAV, MP3, M4A, or FLAC files.');
            return;
        }

        // Validate file size (10MB limit)
        if (file.size > 10 * 1024 * 1024) {
            this.showError('File size exceeds 10MB limit.');
            return;
        }

        this.currentFile = file;
        
        // Update UI
        document.getElementById('file-name').textContent = file.name;
        document.getElementById('file-size').textContent = this.formatFileSize(file.size);
        document.getElementById('file-info').style.display = 'flex';
        document.getElementById('upload-area').style.display = 'none';
        
        // Reset file input for re-selection
        document.getElementById('file-input').value = '';
    }

    /**
     * Format file size for display
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    /**
     * Start audio recording
     */
    async startRecording() {
        if (!this.isSystemReady) {
            this.showError('System not ready. Please wait for initialization to complete.');
            return;
        }

        try {
            // Initialize recorder if not done
            if (!this.audioRecorder.isReady()) {
                const initialized = await this.audioRecorder.initialize();
                if (!initialized) {
                    this.showError('Failed to initialize microphone. Please check permissions.');
                    return;
                }
            }

            // Start recording
            const started = this.audioRecorder.startRecording();
            if (started) {
                this.updateRecordingUI(true);
                this.startTranscription();
            }

        } catch (error) {
            console.error('Recording error:', error);
            this.showError('Failed to start recording: ' + error.message);
        }
    }

    /**
     * Stop audio recording
     */
    stopRecording() {
        if (this.audioRecorder) {
            this.audioRecorder.stopRecording();
            this.updateRecordingUI(false);
            this.stopTranscription();
        }
    }

    /**
     * Start speech-to-text transcription (linked to live recording)
     */
    startTranscription() {
        if (!this.sttSupported || !this.speechRecognition) {
            return;
        }
        if (this.isTranscribing) return;

        this.isTranscribing = true;
        const statusEl = document.getElementById('transcript-status');
        if (statusEl) {
            statusEl.classList.add('active');
            statusEl.textContent = 'Listening... recognized text will appear below while you speak.';
        }
        try {
            this.speechRecognition.start();
        } catch (e) {
            console.error('Failed to start speech recognition:', e);
        }
    }

    /**
     * Stop speech-to-text transcription
     */
    stopTranscription() {
        if (!this.sttSupported || !this.speechRecognition) {
            return;
        }
        this.isTranscribing = false;
        try {
            this.speechRecognition.stop();
        } catch (e) {
            console.error('Failed to stop speech recognition:', e);
        }
    }

    /**
     * Update recording UI state
     */
    updateRecordingUI(isRecording) {
        const startBtn = document.getElementById('start-recording');
        const stopBtn = document.getElementById('stop-recording');
        const indicator = document.getElementById('recording-indicator');
        const pulse = indicator?.querySelector('.pulse');
        const indicatorText = indicator?.querySelector('span');

        if (startBtn) startBtn.disabled = isRecording;
        if (stopBtn) stopBtn.disabled = !isRecording;
        
        if (pulse) {
            pulse.classList.toggle('recording', isRecording);
        }
        
        if (indicatorText) {
            indicatorText.textContent = isRecording ? 'Recording...' : 'Ready to record';
        }
    }

    /**
     * Handle recorded audio data
     */
    async handleRecordingData(audioBlob) {
        if (!audioBlob || audioBlob.size === 0) {
            this.showError('No audio data recorded');
            return;
        }

        this.showLoading('Processing recorded audio...');
        
        try {
            // Create FormData for file upload
            const formData = new FormData();
            formData.append('file', audioBlob, 'recording.webm');
            formData.append('return_probabilities', 'true');

            // Send to API
            const response = await fetch('/api/predict/file', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            this.displayPredictionResults(result);

        } catch (error) {
            console.error('Error processing recording:', error);
            this.showError('Failed to process recording: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    /**
     * Analyze uploaded file
     */
    async analyzeFile() {
        if (!this.currentFile) {
            this.showError('No file selected');
            return;
        }

        if (!this.isSystemReady) {
            this.showError('System not ready. Please wait for initialization to complete.');
            return;
        }

        this.showLoading('Analyzing audio file...');

        try {
            // Create FormData for file upload
            const formData = new FormData();
            formData.append('file', this.currentFile);
            formData.append('return_probabilities', 'true');

            // Send to API
            const response = await fetch('/api/predict/file', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }

            const result = await response.json();
            this.displayPredictionResults(result);

        } catch (error) {
            console.error('Error analyzing file:', error);
            this.showError('Failed to analyze file: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    /**
     * Start real-time streaming
     */
    async startStreaming() {
        if (!this.isSystemReady) {
            this.showError('System not ready. Please wait for initialization to complete.');
            return;
        }

        try {
            // Initialize recorder if not done
            if (!this.audioRecorder.isReady()) {
                const initialized = await this.audioRecorder.initialize();
                if (!initialized) {
                    this.showError('Failed to initialize microphone. Please check permissions.');
                    return;
                }
            }

            // Start streaming
            const started = await this.audioRecorder.startStreaming();
            if (started) {
                this.updateStreamingUI(true);
                this.showRealtimeChart();
            }

        } catch (error) {
            console.error('Streaming error:', error);
            this.showError('Failed to start streaming: ' + error.message);
        }
    }

    /**
     * Stop real-time streaming
     */
    stopStreaming() {
        if (this.audioRecorder) {
            this.audioRecorder.stopStreaming();
            this.updateStreamingUI(false);
            this.hideRealtimeChart();
        }
    }

    /**
     * Update streaming UI state
     */
    updateStreamingUI(isStreaming) {
        const startBtn = document.getElementById('start-streaming');
        const stopBtn = document.getElementById('stop-streaming');
        const status = document.getElementById('streaming-status');

        if (startBtn) startBtn.disabled = isStreaming;
        if (stopBtn) stopBtn.disabled = !isStreaming;
        
        if (status) {
            status.textContent = isStreaming ? 
                'Live monitoring active - emotions are being detected in real-time' : 
                'Click "Start Live Monitoring" to begin real-time emotion detection';
            status.className = isStreaming ? 'streaming-status active' : 'streaming-status';
        }
    }

    /**
     * Handle streaming prediction data
     */
    handleStreamingData(data) {
        if (data.error) {
            console.error('Streaming error:', data.error);
            this.updateStreamingStatus(data.message || 'Processing error', 'error');
            return;
        }

        // Update real-time chart
        this.updateRealtimeChart(data);
        
        // Update streaming status
        const confidence = Math.round(data.confidence * 100);
        this.updateStreamingStatus(
            `Detected: ${data.predicted_emotion} (${confidence}% confidence)`, 
            'active'
        );
    }

    /**
     * Update streaming status display
     */
    updateStreamingStatus(message, type) {
        const status = document.getElementById('streaming-status');
        if (status) {
            status.textContent = message;
            status.className = `streaming-status ${type}`;
        }
    }

    /**
     * Display prediction results
     */
    displayPredictionResults(result) {
        // Hide no-results message
        document.querySelector('.no-results')?.style.setProperty('display', 'none');
        
        // Show prediction display
        const predictionDisplay = document.getElementById('prediction-display');
        if (predictionDisplay) {
            predictionDisplay.style.display = 'block';
            predictionDisplay.classList.add('fade-in');
        }

        // Update main prediction
        this.updateMainPrediction(result);
        
        // Update probability chart
        this.updateEmotionChart(result.emotion_probabilities || result.emotions || {});
        
        // Update probability list
        this.updateProbabilitiesList(result.emotion_probabilities || result.emotions || {});
    }

    /**
     * Update main prediction display
     */
    updateMainPrediction(result) {
        const emotion = result.predicted_emotion || 'neutral';
        const confidence = result.confidence || 0;
        
        // Update emotion icon
        const iconElement = document.getElementById('emotion-icon');
        if (iconElement) {
            iconElement.textContent = this.emotionIcons[emotion] || '😐';
        }
        
        // Update emotion text
        const textElement = document.getElementById('predicted-emotion-text');
        if (textElement) {
            textElement.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
            textElement.style.color = this.emotionColors[emotion] || '#333';
        }
        
        // Update confidence
        const confidenceValue = document.getElementById('confidence-value');
        const confidenceFill = document.getElementById('confidence-fill');
        
        if (confidenceValue) {
            confidenceValue.textContent = Math.round(confidence * 100) + '%';
        }
        
        if (confidenceFill) {
            confidenceFill.style.width = (confidence * 100) + '%';
        }
    }

    /**
     * Update probabilities list
     */
    updateProbabilitiesList(emotions) {
        const container = document.getElementById('probabilities-list');
        if (!container) return;

        container.innerHTML = '';

        // Sort emotions by probability
        const sortedEmotions = Object.entries(emotions)
            .sort(([,a], [,b]) => b - a);

        sortedEmotions.forEach(([emotion, probability]) => {
            const item = document.createElement('div');
            item.className = 'probability-item';
            
            item.innerHTML = `
                <span class="emotion-name">${emotion.charAt(0).toUpperCase() + emotion.slice(1)}</span>
                <div class="probability-bar">
                    <div class="probability-fill" style="width: ${probability * 100}%; background: ${this.emotionColors[emotion] || '#667eea'}"></div>
                </div>
                <span class="probability-value">${Math.round(probability * 100)}%</span>
            `;
            
            container.appendChild(item);
        });
    }

    /**
     * Initialize charts
     */
    initializeCharts() {
        this.initializeEmotionChart();
        this.initializeRealtimeChart();
    }

    /**
     * Initialize emotion probability chart
     */
    initializeEmotionChart() {
        const canvas = document.getElementById('emotion-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        this.emotionChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = Math.round(context.parsed * 100);
                                return `${label}: ${value}%`;
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * Initialize real-time chart
     */
    initializeRealtimeChart() {
        const canvas = document.getElementById('realtime-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        this.realtimeChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Confidence'
                        },
                        min: 0,
                        max: 1
                    }
                }
            }
        });
    }

    /**
     * Update emotion chart with new data
     */
    updateEmotionChart(emotions) {
        if (!this.emotionChart || !emotions) return;

        const labels = Object.keys(emotions);
        const data = Object.values(emotions);
        const colors = labels.map(emotion => this.emotionColors[emotion] || '#667eea');

        this.emotionChart.data.labels = labels.map(label => 
            label.charAt(0).toUpperCase() + label.slice(1)
        );
        this.emotionChart.data.datasets[0].data = data;
        this.emotionChart.data.datasets[0].backgroundColor = colors;
        
        this.emotionChart.update();
    }

    /**
     * Update real-time chart
     */
    updateRealtimeChart(data) {
        if (!this.realtimeChart || !data.emotions) return;

        const timestamp = new Date().toLocaleTimeString();
        
        // Initialize datasets if empty
        if (this.realtimeChart.data.datasets.length === 0) {
            Object.keys(data.emotions).forEach(emotion => {
                this.realtimeChart.data.datasets.push({
                    label: emotion.charAt(0).toUpperCase() + emotion.slice(1),
                    data: [],
                    borderColor: this.emotionColors[emotion] || '#667eea',
                    backgroundColor: this.emotionColors[emotion] || '#667eea',
                    tension: 0.1,
                    fill: false
                });
            });
        }

        // Add new data point
        this.realtimeChart.data.labels.push(timestamp);
        
        this.realtimeChart.data.datasets.forEach((dataset, index) => {
            const emotionName = Object.keys(data.emotions)[index];
            if (emotionName) {
                dataset.data.push(data.emotions[emotionName]);
            }
        });

        // Limit data points
        if (this.realtimeChart.data.labels.length > this.maxRealtimePoints) {
            this.realtimeChart.data.labels.shift();
            this.realtimeChart.data.datasets.forEach(dataset => {
                dataset.data.shift();
            });
        }

        this.realtimeChart.update('none');
    }

    /**
     * Show real-time chart
     */
    showRealtimeChart() {
        const container = document.querySelector('.realtime-chart-container');
        if (container) {
            container.style.display = 'block';
            
            // Clear existing data
            if (this.realtimeChart) {
                this.realtimeChart.data.labels = [];
                this.realtimeChart.data.datasets.forEach(dataset => {
                    dataset.data = [];
                });
                this.realtimeChart.update();
            }
        }
    }

    /**
     * Hide real-time chart
     */
    hideRealtimeChart() {
        const container = document.querySelector('.realtime-chart-container');
        if (container) {
            container.style.display = 'none';
        }
    }

    /**
     * Update audio level indicator
     */
    updateAudioLevel(level) {
        const levelFill = document.getElementById('level-fill');
        if (levelFill) {
            levelFill.style.width = (level * 100) + '%';
        }
    }

    /**
     * Clear all results
     */
    clearResults() {
        // Hide prediction display
        const predictionDisplay = document.getElementById('prediction-display');
        if (predictionDisplay) {
            predictionDisplay.style.display = 'none';
        }

        // Show no-results message
        const noResults = document.querySelector('.no-results');
        if (noResults) {
            noResults.style.display = 'block';
        }

        // Clear file selection
        this.currentFile = null;
        document.getElementById('file-info').style.display = 'none';
        document.getElementById('upload-area').style.display = 'block';

        // Clear charts
        if (this.emotionChart) {
            this.emotionChart.data.labels = [];
            this.emotionChart.data.datasets[0].data = [];
            this.emotionChart.update();
        }

        // Reset audio level
        this.updateAudioLevel(0);

        // Reset transcript
        this.transcriptText = '';
        const textEl = document.getElementById('transcript-text');
        const statusEl = document.getElementById('transcript-status');
        if (textEl) {
            textEl.textContent = 'Your spoken words will appear here as text when speech-to-text is active.';
            textEl.classList.add('transcript-text-placeholder');
        }
        if (statusEl) {
            statusEl.classList.remove('active');
            if (!this.sttSupported) {
                statusEl.classList.add('unsupported');
                statusEl.textContent = 'Speech-to-text is not supported in this browser. Please try Chrome or Edge.';
            } else {
                statusEl.textContent = 'Start a live recording to see the recognized text of your speech.';
            }
        }
    }

    /**
     * Show loading overlay
     */
    showLoading(message = 'Processing...') {
        const overlay = document.getElementById('loading-overlay');
        const text = document.getElementById('loading-text');
        
        if (overlay) overlay.style.display = 'flex';
        if (text) text.textContent = message;
    }

    /**
     * Hide loading overlay
     */
    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) overlay.style.display = 'none';
    }

    /**
     * Show error modal
     */
    showError(message) {
        const modal = document.getElementById('error-modal');
        const messageElement = document.getElementById('error-message');
        
        if (messageElement) messageElement.textContent = message;
        if (modal) modal.style.display = 'flex';
        
        console.error('Error:', message);
    }

    /**
     * Hide error modal
     */
    hideError() {
        const modal = document.getElementById('error-modal');
        if (modal) modal.style.display = 'none';
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        if (this.audioRecorder) {
            this.audioRecorder.cleanup();
        }
        
        if (this.emotionChart) {
            this.emotionChart.destroy();
        }
        
        if (this.realtimeChart) {
            this.realtimeChart.destroy();
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new SERApp();
    app.init();
    
    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        app.cleanup();
    });
});
