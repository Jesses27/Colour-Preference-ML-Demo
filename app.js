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
        
        // Initialize model switcher
        this.initModelSwitcher();
        
        // Initialize explanation system
        this.initExplanationSystem();
        
        console.log('UI initialized, useAlternativeModel:', this.useAlternativeModel);
        
        // Set initial visual state of model switcher
        this.updateModelSwitcherVisualState();
    }

    updateModelSwitcherVisualState() {
        const switcher = document.getElementById('model-switcher');
        if (!switcher) return;
        
        const labels = switcher.querySelectorAll('.model-label');
        labels.forEach((label, index) => {
            if (index === 0) { // Neural Network
                label.style.color = this.useAlternativeModel ? '#666' : '#007bff';
                label.style.fontWeight = this.useAlternativeModel ? 'normal' : 'bold';
            } else { // Rule-Based
                label.style.color = this.useAlternativeModel ? '#007bff' : '#666';
                label.style.fontWeight = this.useAlternativeModel ? 'bold' : 'normal';
            }
        });
    }

    initModelSwitcher() {
        const switcher = document.getElementById('model-switcher');
        if (!switcher) {
            console.error('Model switcher not found in DOM');
            return;
        }
        
        console.log('Model switcher found in DOM');
        
        // Add event listeners
        const radios = switcher.querySelectorAll('input[type="radio"]');
        console.log('Found radios:', radios.length);
        
        radios.forEach((radio, index) => {
            console.log(`Adding listener to radio ${index}:`, radio.value);
            radio.addEventListener('change', (e) => {
                console.log('Radio changed:', e.target.value);
                this.useAlternativeModel = e.target.value === 'rule';
                console.log('useAlternativeModel set to:', this.useAlternativeModel);
                
                this.updateNetworkStatus(`Switched to ${this.useAlternativeModel ? 'Rule-Based (HSV)' : 'Neural Network'} model`);
                
                // Update header to show current model
                const header = document.querySelector('header h1');
                if (header) {
                    header.innerHTML = this.useAlternativeModel ? 
                        '📊 Rule-Based (HSV) Colour Preference Demo' : 
                        '🧠 Neural Network Colour Preference Demo';
                }
                
                // Add model switch explanation
                this.addExplanation('interactions', 'model_switched', 'interaction');
                
                // Reset training data when switching models
                this.trainingData = [];
                this.trainingCount = 0;
                this.trainingHistory = [];
                this.weightHistory = [];
                this.previousWeights = null;
                
                // Reset UI elements
                this.updateStats();
                this.updateTrainingInsights();
                this.updateWeightTable();
                this.updateWeightVisualization();
                
                // Disable inference until retrained
                document.getElementById('start-inference').disabled = true;
                
                // Generate new colors
                this.generateNewColors();
                
                // Add visual feedback to the switcher
                const labels = switcher.querySelectorAll('.model-label');
                labels.forEach((label, index) => {
                    if (index === 0) { // Neural Network
                        label.style.color = this.useAlternativeModel ? '#666' : '#007bff';
                        label.style.fontWeight = this.useAlternativeModel ? 'normal' : 'bold';
                    } else { // Rule-Based
                        label.style.color = this.useAlternativeModel ? '#007bff' : '#666';
                        label.style.fontWeight = this.useAlternativeModel ? 'bold' : 'normal';
                    }
                });
                
                console.log('Model switch complete');
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
        
        // Add specific handlers for insight cards
        this.addInsightCardHandlers();
        
        // Add model comparison explanation on first model switch
        this.addModelComparisonExplanation();
    }

    addModelComparisonExplanation() {
        // Add explanation about model differences when user first switches
        const modelSwitcher = document.getElementById('model-switcher');
        if (modelSwitcher) {
            const radios = modelSwitcher.querySelectorAll('input[type="radio"]');
            radios.forEach(radio => {
                radio.addEventListener('change', () => {
                    // Add model comparison explanation after a short delay
                    setTimeout(() => {
                        this.addExplanation('concepts', 'model_comparison', 'concept');
                    }, 1000);
                });
            });
        }
    }

    addInsightCardHandlers() {
        // Add click handlers for individual insight cards
        const insightCards = document.querySelectorAll('.insight-card');
        insightCards.forEach(card => {
            card.classList.add('clickable');
            card.addEventListener('click', (e) => this.handleInsightCardClick(e, card));
        });
    }

    handleInsightCardClick(event, card) {
        event.preventDefault();
        event.stopPropagation();
        
        // Get the insight type from the card's content
        const cardTitle = card.querySelector('h5').textContent;
        let insightKey = '';
        
        if (cardTitle.includes('Color Sensitivity')) {
            insightKey = 'color_sensitivity';
        } else if (cardTitle.includes('Weight Changes')) {
            insightKey = 'weight_changes';
        } else if (cardTitle.includes('Learning Pattern')) {
            insightKey = 'learning_pattern';
        } else if (cardTitle.includes('Prediction Confidence')) {
            insightKey = 'prediction_confidence';
        }
        
        if (insightKey && this.explanations?.ui_elements?.[insightKey]) {
            this.addExplanation('ui_elements', insightKey);
        }
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
            '#model-switcher': 'model_switcher',
            '.insight-card': 'training_insights' // Fallback for insight cards
        };
        
        const key = selectorMap[selector];
        if (key && this.explanations?.ui_elements?.[key]) {
            this.addExplanation('ui_elements', key);
        }
    }

    addExplanation(category, key, type = 'ui') {
        if (!this.explanations?.[category]?.[key]) return;
        
        // Check if this explanation was already shown recently to prevent duplicates
        const recentExplanations = this.explanationHistory.slice(-3);
        const isDuplicate = recentExplanations.some(exp => exp.category === category && exp.key === key);
        
        // Skip basic UI explanations if they were already shown
        if (type === 'ui' && isDuplicate) return;
        
        const explanation = this.explanations[category][key];
        const bubbleType = this.getBubbleType(type, category);
        
        // Add detailed math explanation for training interactions
        let detailedExplanation = explanation.explanation;
        if (category === 'interactions' && key === 'training_complete' && this.previousWeights) {
            detailedExplanation = this.addWeightChangeDetails(explanation.explanation);
        }
        
        const bubble = document.createElement('div');
        bubble.className = `explanation-bubble ${bubbleType}`;
        bubble.innerHTML = `
            <div class="bubble-header">
                <span class="bubble-icon">${this.getBubbleIcon(bubbleType)}</span>
                <span class="bubble-title">${explanation.title}</span>
            </div>
            <div class="bubble-text">${detailedExplanation}</div>
        `;
        
        const content = document.querySelector('.explanation-content');
        if (content) {
            // Add to the end (which will appear at bottom in normal column direction)
            content.appendChild(bubble);
            
            // Add to history
            this.explanationHistory.push({ category, key, type });
            
            // Limit number of explanations
            if (this.explanationHistory.length > this.maxExplanations) {
                this.removeOldestExplanation();
            }
            
            // Scroll to show new explanation (bottom in normal flex)
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
            // Remove the first child (oldest in normal flex)
            content.removeChild(content.firstChild);
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
        if (this.useAlternativeModel) {
            // Update weight table for rule-based model
            this.updateRuleBasedWeightTable();
        } else {
            // Update weight table for neural network
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
    }

    updateRuleBasedWeightTable() {
        if (!this.alternativeModel) return;
        
        // Calculate HSV preference strengths
        const hueStrength = this.alternativeModel.huePreferences.size / 12; // Normalize to 0-1
        const satStrength = this.alternativeModel.saturationPreferences.size / 4;
        const valStrength = this.alternativeModel.valuePreferences.size / 4;
        
        // Update table with HSV information
        for (let i = 0; i < 8; i++) {
            let hueValue, satValue, valValue;
            
            if (i < 4) {
                // First 4 neurons show hue preferences
                hueValue = hueStrength * (1 - i * 0.2);
                satValue = satStrength * 0.5;
                valValue = valStrength * 0.5;
            } else {
                // Last 4 neurons show saturation/value preferences
                hueValue = hueStrength * 0.3;
                satValue = satStrength * (1 - (i - 4) * 0.2);
                valValue = valStrength * (1 - (i - 4) * 0.2);
            }
            
            document.getElementById(`w-r-${i + 1}`).textContent = hueValue.toFixed(3);
            document.getElementById(`w-g-${i + 1}`).textContent = satValue.toFixed(3);
            document.getElementById(`w-b-${i + 1}`).textContent = valValue.toFixed(3);
            
            // Color code the weights
            this.colorCodeWeight(`w-r-${i + 1}`, hueValue);
            this.colorCodeWeight(`w-g-${i + 1}`, satValue);
            this.colorCodeWeight(`w-b-${i + 1}`, valValue);
        }
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
        if (this.useAlternativeModel) {
            // Update insights for rule-based model
            this.updateRuleBasedInsights();
        } else {
            // Update insights for neural network model
            this.updateColorSensitivity();
            this.updateWeightChanges();
            this.updateLearningPattern();
            this.updatePredictionConfidence();
            this.updateTrainingRecommendations();
        }
        
        // Add explanations for significant changes in insights
        this.addInsightExplanations();
    }

    updateRuleBasedInsights() {
        // Update color sensitivity for rule-based model
        this.updateRuleBasedColorSensitivity();
        
        // Update learning pattern for rule-based model
        this.updateRuleBasedLearningPattern();
        
        // Update prediction confidence for rule-based model
        this.updateRuleBasedPredictionConfidence();
        
        // Update training recommendations for rule-based model
        this.updateRuleBasedTrainingRecommendations();
    }

    addInsightExplanations() {
        // Add explanations for insight changes if they're significant
        if (this.trainingCount > 2) { // Only after some training
            // Check if weight changes are significant
            if (this.previousWeights) {
                const currentWeights = this.getWeights();
                if (currentWeights) {
                    const avgChange = currentWeights.map((w, i) => Math.abs(w - this.previousWeights[i]))
                        .reduce((a, b) => a + b, 0) / currentWeights.length;
                    
                    if (avgChange > 0.05) { // Significant weight change
                        this.addExplanation('ui_elements', 'weight_changes');
                    }
                }
            }
            
            // Check if learning pattern shows improvement
            if (this.trainingHistory.length >= 3) {
                const recentLosses = this.trainingHistory.slice(-3).map(h => h.loss);
                const recentAccuracies = this.trainingHistory.slice(-3).map(h => h.accuracy);
                
                const lossImproving = recentLosses[2] < recentLosses[0];
                const accuracyImproving = recentAccuracies[2] > recentAccuracies[0];
                
                if (lossImproving && accuracyImproving) {
                    this.addExplanation('ui_elements', 'learning_pattern');
                }
            }
            
            // Check if prediction confidence is high enough
            if (this.trainingCount >= this.minTrainingExamples * 2) {
                const recentAccuracy = this.trainingHistory[this.trainingHistory.length - 1]?.accuracy || 0;
                if (recentAccuracy > 0.7) {
                    this.addExplanation('ui_elements', 'prediction_confidence');
                }
            }
        }
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

    updateRuleBasedColorSensitivity() {
        if (!this.alternativeModel || this.alternativeModel.trainingCount < 2) {
            document.getElementById('color-sensitivity').innerHTML = `
                <strong>Analyzing...</strong><br>
                <small>Need more training data</small>
            `;
            return;
        }
        
        // Calculate sensitivity based on HSV preferences
        const hueSensitivity = this.alternativeModel.huePreferences.size;
        const satSensitivity = this.alternativeModel.saturationPreferences.size;
        const valSensitivity = this.alternativeModel.valuePreferences.size;
        
        const sensitivities = [
            { name: 'Hue', value: hueSensitivity, icon: '🎨' },
            { name: 'Saturation', value: satSensitivity, icon: '🌈' },
            { name: 'Brightness', value: valSensitivity, icon: '💡' }
        ];
        
        // Sort by sensitivity
        sensitivities.sort((a, b) => b.value - a.value);
        
        const dominant = sensitivities[0];
        const secondary = sensitivities[1];
        
        document.getElementById('color-sensitivity').innerHTML = `
            <strong>${dominant.icon} ${dominant.name}</strong><br>
            <small>Most sensitive to ${dominant.name.toLowerCase()} (${dominant.value} bins)</small><br>
            <small>Also learning ${secondary.name.toLowerCase()} (${secondary.value} bins)</small>
        `;
    }

    updateRuleBasedLearningPattern() {
        if (!this.alternativeModel || this.alternativeModel.trainingCount < 4) {
            document.getElementById('learning-pattern').innerHTML = `
                <strong>Learning...</strong><br>
                <small>Building color rules</small>
            `;
            return;
        }
        
        const totalBins = this.alternativeModel.huePreferences.size + 
                         this.alternativeModel.saturationPreferences.size + 
                         this.alternativeModel.valuePreferences.size;
        
        let pattern = '';
        if (totalBins < 6) {
            pattern = '<strong>Building Rules</strong><br><small>Learning basic color patterns</small>';
        } else if (totalBins < 12) {
            pattern = '<strong>Refining Rules</strong><br><small>Improving color understanding</small>';
        } else {
            pattern = '<strong>Advanced Rules</strong><br><small>Complex color preference patterns</small>';
        }
        
        document.getElementById('learning-pattern').innerHTML = pattern;
    }

    updateRuleBasedPredictionConfidence() {
        if (!this.alternativeModel || this.alternativeModel.trainingCount < 2) {
            document.getElementById('prediction-confidence').innerHTML = `
                <strong>Low</strong><br>
                <small>Need more training data</small>
            `;
            return;
        }
        
        const totalBins = this.alternativeModel.huePreferences.size + 
                         this.alternativeModel.saturationPreferences.size + 
                         this.alternativeModel.valuePreferences.size;
        
        let confidence = 'Low';
        if (totalBins >= 8) confidence = 'Medium';
        if (totalBins >= 15) confidence = 'High';
        
        let recommendation = '';
        if (this.alternativeModel.trainingCount < 10) {
            recommendation = '<br><small>💡 Train more for better rules</small>';
        } else if (totalBins < 8) {
            recommendation = '<br><small>💡 Try more diverse colors</small>';
        } else {
            recommendation = '<br><small>✅ Rules performing well!</small>';
        }
        
        document.getElementById('prediction-confidence').innerHTML = `
            <strong>${confidence}</strong><br>
            <small>Based on ${this.alternativeModel.trainingCount} examples</small>
            ${recommendation}
        `;
    }

    updateRuleBasedTrainingRecommendations() {
        // Update weight changes for rule-based model
        const totalBins = this.alternativeModel ? 
            (this.alternativeModel.huePreferences.size + 
             this.alternativeModel.saturationPreferences.size + 
             this.alternativeModel.valuePreferences.size) : 0;
        
        let activity = 'Low';
        if (totalBins > 5) activity = 'Medium';
        if (totalBins > 10) activity = 'High';
        
        document.getElementById('weight-changes').innerHTML = `
            <strong>${activity}</strong><br>
            <small>${totalBins} color bins learned</small>
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
        if (!this.canvas) return;
        
        const ctx = this.ctx;
        const canvas = this.canvas;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (this.useAlternativeModel) {
            // Visualize rule-based model
            this.updateRuleBasedWeightVisualization(ctx, canvas);
        } else {
            // Visualize neural network weights
            if (!this.model) return;
            
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

    updateRuleBasedWeightVisualization(ctx, canvas) {
        if (!this.alternativeModel) return;
        
        const padding = 20;
        const cellSize = 25;
        const cols = 8;
        const rows = 3;
        
        // Draw HSV preference visualization
        const hueBins = this.alternativeModel.huePreferences.size;
        const satBins = this.alternativeModel.saturationPreferences.size;
        const valBins = this.alternativeModel.valuePreferences.size;
        
        // Draw hue preferences (top row)
        for (let i = 0; i < cols; i++) {
            const x = padding + i * cellSize;
            const y = padding;
            const intensity = i < hueBins ? 0.8 : 0.2;
            
            ctx.fillStyle = `rgba(255, 0, 0, ${intensity})`; // Red for hue
            ctx.fillRect(x, y, cellSize - 2, cellSize - 2);
        }
        
        // Draw saturation preferences (middle row)
        for (let i = 0; i < cols; i++) {
            const x = padding + i * cellSize;
            const y = padding + cellSize;
            const intensity = i < satBins ? 0.8 : 0.2;
            
            ctx.fillStyle = `rgba(0, 255, 0, ${intensity})`; // Green for saturation
            ctx.fillRect(x, y, cellSize - 2, cellSize - 2);
        }
        
        // Draw value preferences (bottom row)
        for (let i = 0; i < cols; i++) {
            const x = padding + i * cellSize;
            const y = padding + 2 * cellSize;
            const intensity = i < valBins ? 0.8 : 0.2;
            
            ctx.fillStyle = `rgba(0, 0, 255, ${intensity})`; // Blue for value
            ctx.fillRect(x, y, cellSize - 2, cellSize - 2);
        }
        
        // Add labels
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('HSV Rule Visualization', canvas.width / 2, canvas.height - 10);
        ctx.fillText('Red=Hue, Green=Saturation, Blue=Value', canvas.width / 2, canvas.height - 25);
    }

    addWeightChangeDetails(baseExplanation) {
        if (!this.previousWeights || !this.model) return baseExplanation;
        
        const currentWeights = this.getWeights();
        if (!currentWeights) return baseExplanation;
        
        // Calculate weight changes
        const weightChanges = currentWeights.map((w, i) => w - this.previousWeights[i]);
        const avgChange = weightChanges.reduce((a, b) => a + Math.abs(b), 0) / weightChanges.length;
        const maxChange = Math.max(...weightChanges.map(w => Math.abs(w)));
        
        // Find most changed weights
        const maxChangeIndex = weightChanges.findIndex(w => Math.abs(w) === maxChange);
        const layerSize = 16; // First layer has 16 neurons
        const inputIndex = Math.floor(maxChangeIndex / layerSize);
        const neuronIndex = maxChangeIndex % layerSize;
        const inputNames = ['Red', 'Green', 'Blue'];
        const inputName = inputNames[inputIndex];
        
        // Calculate gradient magnitude
        const gradientMagnitude = Math.sqrt(weightChanges.reduce((sum, w) => sum + w * w, 0));
        
        const mathDetails = `
            <br><br><strong>📊 Weight Analysis:</strong>
            <br>• Average weight change: ${avgChange.toFixed(4)}
            <br>• Largest change: ${maxChange.toFixed(4)} (${inputName} → Neuron ${neuronIndex + 1})
            <br>• Gradient magnitude: ${gradientMagnitude.toFixed(4)}
            <br><br><strong>🧮 Math:</strong>
            <br>Δw = w_new - w_old = ${maxChange.toFixed(4)}
            <br>Learning rate effect: Δw ∝ -α∇L
            <br>Where α = 0.005 (learning rate), ∇L = gradient of loss
        `;
        
        return baseExplanation + mathDetails;
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ColorPreferenceNN();
}); 