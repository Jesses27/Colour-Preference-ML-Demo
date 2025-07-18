// Neural Network Color Preference Demo
class ColorPreferenceNN {
    constructor() {
        this.model = null;
        this.trainingData = [];
        this.trainingCount = 0;
        this.currentLoss = 0;
        this.currentAccuracy = 0;
        this.isTraining = false;
        this.minTrainingExamples = 5;
        this.previousWeights = null;
        this.weightHistory = [];
        this.trainingHistory = [];
        
        this.init();
    }

    async init() {
        try {
            // Wait for TensorFlow.js to be ready
            await tf.ready();
            
            // Create the neural network
            this.createModel();
            
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

    createModel() {
        // Simple neural network: RGB input -> hidden layer -> preference output
        this.model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [3], // RGB values
                    units: 8,
                    activation: 'relu',
                    kernelInitializer: 'glorotNormal'
                }),
                tf.layers.dense({
                    units: 4,
                    activation: 'relu'
                }),
                tf.layers.dense({
                    units: 1,
                    activation: 'sigmoid' // Output preference probability
                })
            ]
        });

        // Compile the model
        this.model.compile({
            optimizer: tf.train.adam(0.01),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
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
        
        // Prepare training data
        const input1 = tf.tensor2d([[preferredColor.r / 255, preferredColor.g / 255, preferredColor.b / 255]]);
        const input2 = tf.tensor2d([[nonPreferredColor.r / 255, nonPreferredColor.g / 255, nonPreferredColor.b / 255]]);
        
        const target1 = tf.tensor2d([[1]]); // Preferred
        const target2 = tf.tensor2d([[0]]); // Non-preferred
        
        // Train the model
        try {
            const history = await this.model.fit(input1, target1, {
                epochs: 1,
                verbose: 0
            });
            
            const history2 = await this.model.fit(input2, target2, {
                epochs: 1,
                verbose: 0
            });
            
            // Update training stats
            this.trainingCount += 2;
            this.currentLoss = (history.history.loss[0] + history2.history.loss[0]) / 2;
            this.currentAccuracy = (history.history.acc[0] + history2.history.acc[0]) / 2;
            
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
            
            // Enable inference mode after minimum training examples
            if (this.trainingCount >= this.minTrainingExamples * 2) {
                document.getElementById('start-inference').disabled = false;
            }
            
            // Generate new colors for next training
            this.generateNewColors();
            
        } catch (error) {
            console.error('Training error:', error);
        } finally {
            this.isTraining = false;
        }
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
        
        // Update weight table
        for (let i = 0; i < 8; i++) {
            const redWeight = weights[i];
            const greenWeight = weights[i + 8];
            const blueWeight = weights[i + 16];
            
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
    }

    updateColorSensitivity() {
        const weights = this.getWeights();
        if (!weights) return;
        
        const redSensitivity = Math.abs(weights.slice(0, 8).reduce((a, b) => a + Math.abs(b), 0) / 8);
        const greenSensitivity = Math.abs(weights.slice(8, 16).reduce((a, b) => a + Math.abs(b), 0) / 8);
        const blueSensitivity = Math.abs(weights.slice(16, 24).reduce((a, b) => a + Math.abs(b), 0) / 8);
        
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
        
        document.getElementById('prediction-confidence').innerHTML = `
            <strong>${confidence}</strong><br>
            <small>Based on ${this.trainingCount} examples</small>
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
            
            // Prepare input data
            const input1 = tf.tensor2d([[this.currentColors[0].r / 255, this.currentColors[0].g / 255, this.currentColors[0].b / 255]]);
            const input2 = tf.tensor2d([[this.currentColors[1].r / 255, this.currentColors[1].g / 255, this.currentColors[1].b / 255]]);
            
            // Get predictions
            const pred1 = await this.model.predict(input1).data();
            const pred2 = await this.model.predict(input2).data();
            
            // Update prediction bars
            const pred1Fill = document.getElementById('pred1-fill');
            const pred2Fill = document.getElementById('pred2-fill');
            const pred1Text = document.getElementById('pred1-text');
            const pred2Text = document.getElementById('pred2-text');
            
            const prob1 = pred1[0];
            const prob2 = pred2[0];
            
            pred1Fill.style.width = `${prob1 * 100}%`;
            pred2Fill.style.width = `${prob2 * 100}%`;
            pred1Text.textContent = `${(prob1 * 100).toFixed(1)}%`;
            pred2Text.textContent = `${(prob2 * 100).toFixed(1)}%`;
            
            // Store predictions for feedback
            this.currentPredictions = [prob1, prob2];
            
            this.updateNetworkStatus('Prediction complete!');
            
        } catch (error) {
            console.error('Prediction error:', error);
            this.updateNetworkStatus('Prediction error');
        }
    }

    async handleFeedback(isCorrect) {
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
        
        // Draw weight visualization
        const cellSize = 20;
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
            ctx.fillRect(x, y, cellSize - 2, cellSize - 2);
        }
        
        // Add labels
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Layer 1 Weights', canvas.width / 2, canvas.height - 10);
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ColorPreferenceNN();
}); 