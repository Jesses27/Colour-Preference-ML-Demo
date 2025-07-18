// Neural Network Color Preference Demo
class ColorPreferenceNN {
    constructor() {
        this.model = null;
        this.trainingData = [];
        this.trainingCount = 0;
        this.currentLoss = 0;
        this.currentAccuracy = 0;
        this.isTraining = false;
        this.minTrainingExamples = 10; // Increased minimum examples
        this.previousWeights = null;
        this.weightHistory = [];
        this.trainingHistory = [];
        this.validationData = []; // New: validation data
        this.batchSize = 8; // New: batch training
        this.maxTrainingData = 100; // New: limit training data size
        this.useAlternativeModel = false; // New: option to use alternative model
        this.alternativeModel = null; // New: alternative rule-based model
        
        // Explanation system
        this.explanations = null;
        this.explanationHistory = [];
        this.maxExplanations = 10;
        
        this.init();
    }

    async init() {
        try {
            // Wait for TensorFlow.js to be ready
            await tf.ready();
            
            // Load explanations
            await this.loadExplanations();
            
            // Create the neural network
            this.createModel();
            
            // Initialize alternative model
            this.initAlternativeModel();
            
            // Initialize UI
            this.initUI();
            
            // Generate initial colors
            this.generateNewColors();
            
            this.updateNetworkStatus('Ready to train!');
        } catch (error) {
            console.error('Error initializing:', error);
            this.updateNetworkStatus('Error initializing TensorFlow.js');
        }
    }

    async loadExplanations() {
        try {
            const response = await fetch('explanations.json');
            this.explanations = await response.json();
        } catch (error) {
            console.error('Error loading explanations:', error);
            this.explanations = {};
        }
    }

    createModel() {
        // Improved neural network with more capacity and regularization
        this.model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [3], // RGB values
                    units: 16, // Increased from 8
                    activation: 'relu',
                    kernelInitializer: 'glorotNormal',
                    kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }) // Add regularization
                }),
                tf.layers.dropout({ rate: 0.2 }), // Add dropout
                tf.layers.dense({
                    units: 12, // Increased from 4
                    activation: 'relu',
                    kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({
                    units: 8,
                    activation: 'relu',
                    kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
                }),
                tf.layers.dense({
                    units: 1,
                    activation: 'sigmoid' // Output preference probability
                })
            ]
        });

        // Compile the model with better optimizer settings
        this.model.compile({
            optimizer: tf.train.adam(0.005), // Reduced learning rate
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
    }

    initAlternativeModel() {
        // Simple rule-based model using HSV color space
        this.alternativeModel = {
            huePreferences: new Map(), // Track hue preferences
            saturationPreferences: new Map(), // Track saturation preferences
            valuePreferences: new Map(), // Track brightness preferences
            trainingCount: 0,
            
            // Convert RGB to HSV
            rgbToHsv: function(r, g, b) {
                r /= 255; g /= 255; b /= 255;
                const max = Math.max(r, g, b);
                const min = Math.min(r, g, b);
                const diff = max - min;
                
                let h = 0, s = 0, v = max;
                
                if (diff !== 0) {
                    s = diff / max;
                    switch (max) {
                        case r: h = (g - b) / diff + (g < b ? 6 : 0); break;
                        case g: h = (b - r) / diff + 2; break;
                        case b: h = (r - g) / diff + 4; break;
                    }
                    h /= 6;
                }
                
                return { h: h * 360, s: s * 100, v: v * 100 };
            },
            
            // Train on a color preference
            train: function(color, preferred) {
                const hsv = this.rgbToHsv(color.r, color.g, color.b);
                
                // Update hue preferences
                const hueBin = Math.floor(hsv.h / 30); // 12 hue bins
                if (!this.huePreferences.has(hueBin)) {
                    this.huePreferences.set(hueBin, { preferred: 0, total: 0 });
                }
                const hueStats = this.huePreferences.get(hueBin);
                hueStats.total++;
                if (preferred) hueStats.preferred++;
                
                // Update saturation preferences
                const satBin = Math.floor(hsv.s / 25); // 4 saturation bins
                if (!this.saturationPreferences.has(satBin)) {
                    this.saturationPreferences.set(satBin, { preferred: 0, total: 0 });
                }
                const satStats = this.saturationPreferences.get(satBin);
                satStats.total++;
                if (preferred) satStats.preferred++;
                
                // Update value preferences
                const valBin = Math.floor(hsv.v / 25); // 4 value bins
                if (!this.valuePreferences.has(valBin)) {
                    this.valuePreferences.set(valBin, { preferred: 0, total: 0 });
                }
                const valStats = this.valuePreferences.get(valBin);
                valStats.total++;
                if (preferred) valStats.preferred++;
                
                this.trainingCount++;
            },
            
            // Predict preference for a color
            predict: function(color) {
                const hsv = this.rgbToHsv(color.r, color.g, color.b);
                
                const hueBin = Math.floor(hsv.h / 30);
                const satBin = Math.floor(hsv.s / 25);
                const valBin = Math.floor(hsv.v / 25);
                
                let hueScore = 0.5, satScore = 0.5, valScore = 0.5;
                
                // Calculate scores based on training data
                if (this.huePreferences.has(hueBin)) {
                    const stats = this.huePreferences.get(hueBin);
                    hueScore = stats.preferred / stats.total;
                }
                
                if (this.saturationPreferences.has(satBin)) {
                    const stats = this.saturationPreferences.get(satBin);
                    satScore = stats.preferred / stats.total;
                }
                
                if (this.valuePreferences.has(valBin)) {
                    const stats = this.valuePreferences.get(valBin);
                    valScore = stats.preferred / stats.total;
                }
                
                // Weighted average of all factors
                return (hueScore * 0.4 + satScore * 0.3 + valScore * 0.3);
            }
        };
    }

    initUI() {
        // Training phase elements
        const color1Btn = document.querySelector('#color1 .prefer-btn');
        const color2Btn = document.querySelector('#color2 .prefer-btn');
        const startInferenceBtn = document.getElementById('start-inference');
        
        // Inference phase elements
        const correctBtn = document.getElementById('correct-btn');
        const incorrectBtn = document.getElementById('incorrect-btn');
        const backToTrainingBtn = document.getElementById('back-to-training');
        
        // Event listeners
        color1Btn.addEventListener('click', () => this.trainOnPreference(0));
        color2Btn.addEventListener('click', () => this.trainOnPreference(1));
        startInferenceBtn.addEventListener('click', () => this.startInferenceMode());
        correctBtn.addEventListener('click', () => this.handleFeedback(true));
        incorrectBtn.addEventListener('click', () => this.handleFeedback(false));
        backToTrainingBtn.addEventListener('click', () => this.backToTraining());
        
        // Initialize canvas for weight visualization
        this.initWeightVisualization();
        
        // Add model switcher if not already present
        this.addModelSwitcher();
        
        // Initialize explanation system
        this.initExplanationSystem();
    }

    addModelSwitcher() {
        // Check if switcher already exists
        if (document.getElementById('model-switcher')) return;
        
        const header = document.querySelector('header');
        const switcher = document.createElement('div');
        switcher.id = 'model-switcher';
        switcher.innerHTML = `
            <div class="model-switch">
                <label>
                    <input type="radio" name="model" value="neural" checked> Neural Network
                </label>
                <label>
                    <input type="radio" name="model" value="rule"> Rule-Based (HSV)
                </label>
            </div>
        `;
        
        header.appendChild(switcher);
        
        // Add event listeners
        const radios = switcher.querySelectorAll('input[type="radio"]');
        radios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.useAlternativeModel = e.target.value === 'rule';
                this.updateNetworkStatus(`Switched to ${this.useAlternativeModel ? 'Rule-Based' : 'Neural Network'} model`);
                
                // Add model switch explanation
                this.addExplanation('interactions', 'model_switched', 'interaction');
                
                // Reset training data when switching models
                this.trainingData = [];
                this.trainingCount = 0;
                this.trainingHistory = [];
                this.updateStats();
                this.updateTrainingInsights();
                
                // Disable inference until retrained
                document.getElementById('start-inference').disabled = true;
            });
        });
    }

    initExplanationSystem() {
        // Add click handlers for UI elements
        this.addClickableElements();
        
        // Add clear explanations button handler
        const clearBtn = document.querySelector('.clear-explanations');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearExplanations());
        }
    }

    addClickableElements() {
        // Add clickable class and handlers to UI elements
        const clickableSelectors = [
            '#training-phase',
            '#inference-phase', 
            '.color-display',
            '.prefer-btn',
            '.stats',
            '#start-inference',
            '.prediction-bar',
            '#correct-btn',
            '#incorrect-btn',
            '#weight-table',
            '#weight-canvas',
            '.training-insights',
            '#model-switcher'
        ];
        
        clickableSelectors.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            elements.forEach(element => {
                element.classList.add('clickable');
                element.addEventListener('click', (e) => this.handleElementClick(e, selector));
            });
        });
    }

    handleElementClick(event, selector) {
        event.preventDefault();
        event.stopPropagation();
        
        // Map selectors to explanation keys
        const selectorMap = {
            '#training-phase': 'training_phase',
            '#inference-phase': 'inference_phase',
            '.color-display': 'color_displays',
            '.prefer-btn': 'prefer_buttons',
            '.stats': 'training_stats',
            '#start-inference': 'start_inference',
            '.prediction-bar': 'prediction_bars',
            '#correct-btn, #incorrect-btn': 'feedback_buttons',
            '#weight-table': 'weight_table',
            '#weight-canvas': 'weight_canvas',
            '.training-insights': 'training_insights',
            '#model-switcher': 'model_switcher'
        };
        
        const key = selectorMap[selector];
        if (key && this.explanations?.ui_elements?.[key]) {
            this.addExplanation('ui_elements', key);
        }
    }

    addExplanation(category, key, type = 'ui') {
        if (!this.explanations?.[category]?.[key]) return;
        
        const explanation = this.explanations[category][key];
        const bubbleType = this.getBubbleType(type, category);
        
        const bubble = document.createElement('div');
        bubble.className = `explanation-bubble ${bubbleType}`;
        bubble.innerHTML = `
            <div class="bubble-header">
                <span class="bubble-icon">${this.getBubbleIcon(bubbleType)}</span>
                <span class="bubble-title">${explanation.title}</span>
            </div>
            <div class="bubble-text">${explanation.explanation}</div>
        `;
        
        const content = document.querySelector('.explanation-content');
        if (content) {
            content.appendChild(bubble);
            
            // Add to history
            this.explanationHistory.push({ category, key, type });
            
            // Limit number of explanations
            if (this.explanationHistory.length > this.maxExplanations) {
                this.removeOldestExplanation();
            }
            
            // Scroll to bottom
            content.scrollTop = content.scrollHeight;
        }
    }

    getBubbleType(type, category) {
        if (type === 'interaction') {
            switch (category) {
                case 'color_preference_clicked': return 'training';
                case 'prediction_made': return 'prediction';
                case 'feedback_given': return 'training';
                case 'model_switched': return 'concept';
                case 'training_complete': return 'training';
                case 'inference_enabled': return 'prediction';
                default: return 'concept';
            }
        }
        return 'concept';
    }

    getBubbleIcon(type) {
        const icons = {
            'welcome': '👋',
            'training': '🎯',
            'prediction': '🔮',
            'concept': '💡'
        };
        return icons[type] || '💡';
    }

    removeOldestExplanation() {
        const content = document.querySelector('.explanation-content');
        if (content && content.children.length > 1) { // Keep at least the welcome message
            content.removeChild(content.children[1]); // Remove first non-welcome bubble
            this.explanationHistory.shift();
        }
    }

    clearExplanations() {
        const content = document.querySelector('.explanation-content');
        if (content) {
            // Keep only the welcome message
            const welcomeBubble = content.querySelector('.explanation-bubble.welcome');
            content.innerHTML = '';
            if (welcomeBubble) {
                content.appendChild(welcomeBubble);
            }
            this.explanationHistory = [];
        }
    }

    generateNewColors() {
        const color1 = this.generateRandomColor();
        const color2 = this.generateRandomColor();
        
        // Ensure colors are different enough
        while (this.colorDistance(color1, color2) < 100) {
            color2.r = Math.floor(Math.random() * 256);
            color2.g = Math.floor(Math.random() * 256);
            color2.b = Math.floor(Math.random() * 256);
        }
        
        this.currentColors = [color1, color2];
        this.updateColorDisplays();
    }

    generateRandomColor() {
        return {
            r: Math.floor(Math.random() * 256),
            g: Math.floor(Math.random() * 256),
            b: Math.floor(Math.random() * 256)
        };
    }

    colorDistance(color1, color2) {
        return Math.sqrt(
            Math.pow(color1.r - color2.r, 2) +
            Math.pow(color1.g - color2.g, 2) +
            Math.pow(color1.b - color2.b, 2)
        );
    }

    updateColorDisplays() {
        // Update training phase colors
        const color1Display = document.querySelector('#color1 .color-display');
        const color2Display = document.querySelector('#color2 .color-display');
        
        color1Display.style.backgroundColor = `rgb(${this.currentColors[0].r}, ${this.currentColors[0].g}, ${this.currentColors[0].b})`;
        color2Display.style.backgroundColor = `rgb(${this.currentColors[1].r}, ${this.currentColors[1].g}, ${this.currentColors[1].b})`;
        
        // Update prediction phase colors
        const predColor1Display = document.querySelector('#pred-color1 .color-display');
        const predColor2Display = document.querySelector('#pred-color2 .color-display');
        
        predColor1Display.style.backgroundColor = `rgb(${this.currentColors[0].r}, ${this.currentColors[0].g}, ${this.currentColors[0].b})`;
        predColor2Display.style.backgroundColor = `rgb(${this.currentColors[1].r}, ${this.currentColors[1].g}, ${this.currentColors[1].b})`;
    }

    async trainOnPreference(preferredIndex) {
        if (this.isTraining) return;
        
        this.isTraining = true;
        
        // Add explanation for training interaction
        this.addExplanation('interactions', 'color_preference_clicked', 'interaction');
        
        // Store previous weights for analysis
        this.previousWeights = this.getWeights();
        
        // Get the preferred and non-preferred colors
        const preferredColor = this.currentColors[preferredIndex];
        const nonPreferredColor = this.currentColors[1 - preferredIndex];
        
        if (this.useAlternativeModel) {
            // Train alternative model
            this.alternativeModel.train(preferredColor, true);
            this.alternativeModel.train(nonPreferredColor, false);
            
            // Update stats for alternative model
            this.trainingCount += 2;
            this.currentLoss = 0; // Not applicable for rule-based model
            this.currentAccuracy = 0.5; // Placeholder
            
            this.updateStats();
            this.updateTrainingInsights();
            
            // Enable inference mode after minimum training examples
            if (this.trainingCount >= this.minTrainingExamples * 2) {
                document.getElementById('start-inference').disabled = false;
                this.addExplanation('interactions', 'inference_enabled', 'interaction');
            }
            
            // Generate new colors for next training
            this.generateNewColors();
        } else {
            // Add to training data for neural network
            this.addTrainingData(preferredColor, 1);
            this.addTrainingData(nonPreferredColor, 0);
            
            // Train the model with accumulated data
            try {
                const history = await this.trainOnBatch();
                
                // Update training stats
                this.trainingCount += 2;
                this.currentLoss = history.history.loss[0];
                this.currentAccuracy = history.history.acc[0];
                
                // Store training history
                this.trainingHistory.push({
                    loss: this.currentLoss,
                    accuracy: this.currentAccuracy,
                    step: this.trainingCount
                });
                
                this.updateStats();
                this.updateWeightTable();
                this.updateWeightVisualization();
                this.updateTrainingInsights();
                
                // Add training complete explanation
                this.addExplanation('interactions', 'training_complete', 'interaction');
                
                // Enable inference mode after minimum training examples
                if (this.trainingCount >= this.minTrainingExamples * 2) {
                    document.getElementById('start-inference').disabled = false;
                    this.addExplanation('interactions', 'inference_enabled', 'interaction');
                }
                
                // Generate new colors for next training
                this.generateNewColors();
                
            } catch (error) {
                console.error('Training error:', error);
            }
        }
        
        this.isTraining = false;
    }

    addTrainingData(color, preference) {
        // Add new training example
        this.trainingData.push({
            input: [color.r / 255, color.g / 255, color.b / 255],
            target: preference
        });
        
        // Limit training data size to prevent memory issues
        if (this.trainingData.length > this.maxTrainingData) {
            this.trainingData = this.trainingData.slice(-this.maxTrainingData);
        }
    }

    async trainOnBatch() {
        if (this.trainingData.length === 0) {
            throw new Error('No training data available');
        }
        
        // Prepare batch data
        const inputs = this.trainingData.map(example => example.input);
        const targets = this.trainingData.map(example => example.target);
        
        // Convert to tensors
        const inputTensor = tf.tensor2d(inputs);
        const targetTensor = tf.tensor2d(targets, [targets.length, 1]);
        
        // Train the model
        const history = await this.model.fit(inputTensor, targetTensor, {
            epochs: 3, // Train for multiple epochs
            batchSize: Math.min(this.batchSize, this.trainingData.length),
            verbose: 0,
            shuffle: true // Shuffle data for better training
        });
        
        // Clean up tensors
        inputTensor.dispose();
        targetTensor.dispose();
        
        return history;
    }

    updateStats() {
        document.getElementById('training-count').textContent = this.trainingCount;
        document.getElementById('current-loss').textContent = this.currentLoss.toFixed(4);
        document.getElementById('current-accuracy').textContent = (this.currentAccuracy * 100).toFixed(1) + '%';
    }

    getWeights() {
        if (!this.model) return null;
        
        const weights = this.model.layers[0].getWeights()[0];
        return weights.dataSync();
    }

    updateWeightTable() {
        const weights = this.getWeights();
        if (!weights) return;
        
        // Update weight table for first layer (16 neurons now)
        for (let i = 0; i < 8; i++) {
            const redWeight = weights[i];
            const greenWeight = weights[i + 16];
            const blueWeight = weights[i + 32];
            
            document.getElementById(`w-r-${i + 1}`).textContent = redWeight.toFixed(3);
            document.getElementById(`w-g-${i + 1}`).textContent = greenWeight.toFixed(3);
            document.getElementById(`w-b-${i + 1}`).textContent = blueWeight.toFixed(3);
            
            // Color code the weights
            this.colorCodeWeight(`w-r-${i + 1}`, redWeight);
            this.colorCodeWeight(`w-g-${i + 1}`, greenWeight);
            this.colorCodeWeight(`w-b-${i + 1}`, blueWeight);
        }
        
        // Store weight history
        this.weightHistory.push({
            weights: Array.from(weights),
            step: this.trainingCount
        });
    }

    colorCodeWeight(elementId, weight) {
        const element = document.getElementById(elementId);
        element.className = 'weight-value';
        
        if (weight > 0.1) {
            element.classList.add('weight-positive');
        } else if (weight < -0.1) {
            element.classList.add('weight-negative');
        } else {
            element.classList.add('weight-neutral');
        }
    }

    updateTrainingInsights() {
        this.updateColorSensitivity();
        this.updateWeightChanges();
        this.updateLearningPattern();
        this.updatePredictionConfidence();
        this.updateTrainingRecommendations();
    }

    updateTrainingRecommendations() {
        const statusElement = document.getElementById('network-status');
        let status = '';
        
        if (this.trainingCount < 10) {
            status = `Training... (${this.trainingCount}/20 recommended examples)`;
        } else if (this.trainingCount < 20) {
            status = `Good progress! (${this.trainingCount}/20 examples)`;
        } else if (this.currentAccuracy > 0.8) {
            status = 'Model performing well! Ready for predictions.';
        } else if (this.currentAccuracy > 0.6) {
            status = 'Model learning... Try more diverse color combinations.';
        } else {
            status = 'Model needs more training with varied examples.';
        }
        
        statusElement.textContent = status;
    }

    updateColorSensitivity() {
        const weights = this.getWeights();
        if (!weights) return;
        
        // Calculate sensitivity for 16 neurons (3 inputs each)
        const redSensitivity = Math.abs(weights.slice(0, 16).reduce((a, b) => a + Math.abs(b), 0) / 16);
        const greenSensitivity = Math.abs(weights.slice(16, 32).reduce((a, b) => a + Math.abs(b), 0) / 16);
        const blueSensitivity = Math.abs(weights.slice(32, 48).reduce((a, b) => a + Math.abs(b), 0) / 16);
        
        const maxSensitivity = Math.max(redSensitivity, greenSensitivity, blueSensitivity);
        const dominantColor = maxSensitivity === redSensitivity ? 'Red' : 
                            maxSensitivity === greenSensitivity ? 'Green' : 'Blue';
        
        document.getElementById('color-sensitivity').innerHTML = `
            <strong>Dominant:</strong> ${dominantColor}<br>
            <small>R: ${redSensitivity.toFixed(3)} | G: ${greenSensitivity.toFixed(3)} | B: ${blueSensitivity.toFixed(3)}</small>
        `;
    }

    updateWeightChanges() {
        if (!this.previousWeights || this.weightHistory.length < 2) {
            document.getElementById('weight-changes').textContent = 'No changes yet';
            return;
        }
        
        const currentWeights = this.getWeights();
        const changes = currentWeights.map((w, i) => Math.abs(w - this.previousWeights[i]));
        const avgChange = changes.reduce((a, b) => a + b, 0) / changes.length;
        
        const changeLevel = avgChange > 0.1 ? 'High' : avgChange > 0.05 ? 'Medium' : 'Low';
        
        document.getElementById('weight-changes').innerHTML = `
            <strong>${changeLevel} activity</strong><br>
            <small>Avg change: ${avgChange.toFixed(4)}</small>
        `;
    }

    updateLearningPattern() {
        if (this.trainingHistory.length < 3) {
            document.getElementById('learning-pattern').textContent = 'Waiting for data...';
            return;
        }
        
        const recentLosses = this.trainingHistory.slice(-3).map(h => h.loss);
        const lossTrend = recentLosses[2] < recentLosses[0] ? 'Decreasing' : 'Stable';
        
        const recentAccuracies = this.trainingHistory.slice(-3).map(h => h.accuracy);
        const accuracyTrend = recentAccuracies[2] > recentAccuracies[0] ? 'Improving' : 'Stable';
        
        document.getElementById('learning-pattern').innerHTML = `
            <strong>Loss:</strong> ${lossTrend}<br>
            <strong>Accuracy:</strong> ${accuracyTrend}<br>
            <small>Last 3 steps</small>
        `;
    }

    updatePredictionConfidence() {
        if (this.trainingCount < this.minTrainingExamples * 2) {
            document.getElementById('prediction-confidence').textContent = 'Need more training';
            return;
        }
        
        const recentAccuracy = this.trainingHistory[this.trainingHistory.length - 1]?.accuracy || 0;
        const confidence = recentAccuracy > 0.8 ? 'High' : 
                         recentAccuracy > 0.6 ? 'Medium' : 'Low';
        
        // Add training recommendations
        let recommendation = '';
        if (this.trainingCount < 20) {
            recommendation = '<br><small>💡 Train more for better accuracy</small>';
        } else if (recentAccuracy < 0.7) {
            recommendation = '<br><small>💡 Try different color combinations</small>';
        } else if (recentAccuracy > 0.85) {
            recommendation = '<br><small>✅ Model performing well!</small>';
        }
        
        document.getElementById('prediction-confidence').innerHTML = `
            <strong>${confidence}</strong><br>
            <small>Based on ${this.trainingCount} examples</small>
            ${recommendation}
        `;
    }

    async startInferenceMode() {
        // Switch to inference phase
        document.getElementById('training-phase').classList.remove('active');
        document.getElementById('inference-phase').classList.add('active');
        
        this.updateNetworkStatus('Making predictions...');
        await this.makePrediction();
    }

    async makePrediction() {
        try {
            // Generate new colors for prediction
            this.generateNewColors();
            
            let prob1, prob2;
            
            if (this.useAlternativeModel) {
                // Use alternative model for prediction
                prob1 = this.alternativeModel.predict(this.currentColors[0]);
                prob2 = this.alternativeModel.predict(this.currentColors[1]);
            } else {
                // Use neural network for prediction
                const input1 = tf.tensor2d([[this.currentColors[0].r / 255, this.currentColors[0].g / 255, this.currentColors[0].b / 255]]);
                const input2 = tf.tensor2d([[this.currentColors[1].r / 255, this.currentColors[1].g / 255, this.currentColors[1].b / 255]]);
                
                const pred1 = await this.model.predict(input1).data();
                const pred2 = await this.model.predict(input2).data();
                
                prob1 = pred1[0];
                prob2 = pred2[0];
            }
            
            // Update prediction bars
            const pred1Fill = document.getElementById('pred1-fill');
            const pred2Fill = document.getElementById('pred2-fill');
            const pred1Text = document.getElementById('pred1-text');
            const pred2Text = document.getElementById('pred2-text');
            
            pred1Fill.style.width = `${prob1 * 100}%`;
            pred2Fill.style.width = `${prob2 * 100}%`;
            pred1Text.textContent = `${(prob1 * 100).toFixed(1)}%`;
            pred2Text.textContent = `${(prob2 * 100).toFixed(1)}%`;
            
            // Store predictions for feedback
            this.currentPredictions = [prob1, prob2];
            
            // Add prediction explanation
            this.addExplanation('interactions', 'prediction_made', 'interaction');
            
            this.updateNetworkStatus('Prediction complete!');
            
        } catch (error) {
            console.error('Prediction error:', error);
            this.updateNetworkStatus('Prediction error');
        }
    }

    async handleFeedback(isCorrect) {
        // Add feedback explanation
        this.addExplanation('interactions', 'feedback_given', 'interaction');
        
        // If prediction was incorrect, add to training data
        if (!isCorrect) {
            // Determine which color was predicted as preferred
            const predictedPreferred = this.currentPredictions[0] > this.currentPredictions[1] ? 0 : 1;
            
            // Ask user which they actually preferred
            const actualPreferred = confirm(
                `The network predicted you'd prefer the ${predictedPreferred === 0 ? 'left' : 'right'} color.\n` +
                `Which color did you actually prefer?\n\n` +
                `Click OK for left, Cancel for right.`
            ) ? 0 : 1;
            
            // Train on the correct preference
            await this.trainOnPreference(actualPreferred);
        }
        
        // Make new prediction
        await this.makePrediction();
    }

    backToTraining() {
        document.getElementById('inference-phase').classList.remove('active');
        document.getElementById('training-phase').classList.add('active');
        this.updateNetworkStatus('Ready to train!');
    }

    updateNetworkStatus(status) {
        document.getElementById('network-status').textContent = status;
    }

    initWeightVisualization() {
        this.canvas = document.getElementById('weight-canvas');
        this.ctx = this.canvas.getContext('2d');
    }

    updateWeightVisualization() {
        if (!this.model || !this.canvas) return;
        
        const ctx = this.ctx;
        const canvas = this.canvas;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Get weights from the first layer
        const weights = this.model.layers[0].getWeights()[0].dataSync();
        
        // Draw weight visualization with smaller cells for more weights
        const cellSize = 15; // Smaller cells
        const padding = 10;
        const cols = Math.ceil(Math.sqrt(weights.length));
        const rows = Math.ceil(weights.length / cols);
        
        for (let i = 0; i < weights.length; i++) {
            const row = Math.floor(i / cols);
            const col = i % cols;
            const x = padding + col * cellSize;
            const y = padding + row * cellSize;
            
            // Normalize weight to 0-1 range for color intensity
            const weight = weights[i];
            const intensity = Math.max(0, Math.min(1, (weight + 1) / 2));
            
            ctx.fillStyle = `rgba(0, 123, 255, ${intensity})`;
            ctx.fillRect(x, y, cellSize - 1, cellSize - 1);
        }
        
        // Add labels
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`Layer 1 Weights (${weights.length} total)`, canvas.width / 2, canvas.height - 10);
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ColorPreferenceNN();
}); 