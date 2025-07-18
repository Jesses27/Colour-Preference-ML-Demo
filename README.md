# 🧠 Neural Network Color Preference Demo

An interactive web application that trains a neural network to learn your color preferences through a simple A/B testing interface.

## ✨ Features

- **Interactive Training**: Click on your preferred color to train the neural network
- **Real-time Visualization**: Watch the network weights update as you train
- **Prediction Mode**: See how well the model predicts your preferences
- **Two Model Types**: Choose between Neural Network and Rule-Based (HSV) models
- **Training Insights**: Get detailed analysis of the learning process
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Getting Started

1. **Open the Demo**: Visit the live demo at [https://jesses27.github.io/Colour-Preference-ML-Demo/](https://jesses27.github.io/Colour-Preference-ML-Demo/)

2. **Choose Your Model**: 
   - **Neural Network**: Deep learning model with multiple layers
   - **Rule-Based (HSV)**: Simpler model based on color space analysis

3. **Start Training**: Click on your preferred color in each pair

4. **Switch to Prediction**: Once you've trained enough, test the model's predictions

## 🧠 Model Architecture

### Neural Network Model
- **Input Layer**: 3 neurons (RGB values)
- **Hidden Layer 1**: 16 neurons with ReLU activation + Dropout
- **Hidden Layer 2**: 12 neurons with ReLU activation + Dropout  
- **Hidden Layer 3**: 8 neurons with ReLU activation
- **Output Layer**: 1 neuron with Sigmoid activation
- **Regularization**: L2 regularization to prevent overfitting

### Rule-Based Model (HSV)
- **Color Space**: HSV (Hue, Saturation, Value)
- **Binning Strategy**: 
  - 12 hue bins (30° each)
  - 4 saturation bins (25% each)
  - 4 value bins (25% each)
- **Prediction**: Weighted average of hue, saturation, and value preferences

## 📊 Training Recommendations

### How Many Training Examples?

**Minimum**: 10-20 examples (5-10 color pairs)
**Recommended**: 30-50 examples (15-25 color pairs)
**Optimal**: 50+ examples for complex preferences

### Training Tips

1. **Be Consistent**: Try to make consistent choices during training
2. **Diverse Colors**: Train on a variety of colors (bright, dark, saturated, muted)
3. **Take Your Time**: Don't rush - think about your preferences
4. **Retrain if Needed**: If predictions are poor, switch models or retrain

### Model Selection Guide

**Choose Neural Network if:**
- You have complex, nuanced color preferences
- You want to train on 20+ examples
- You prefer a "black box" approach

**Choose Rule-Based if:**
- You want interpretable results
- You have simple, consistent preferences
- You prefer faster training with fewer examples

## 🎯 Accuracy Expectations

### Neural Network Model
- **10 examples**: 60-70% accuracy
- **20 examples**: 70-80% accuracy  
- **30+ examples**: 80-90% accuracy

### Rule-Based Model
- **10 examples**: 65-75% accuracy
- **20 examples**: 70-80% accuracy
- **30+ examples**: 75-85% accuracy

## 🔧 Technical Details

### Built With
- **TensorFlow.js**: Neural network implementation
- **Vanilla JavaScript**: Core application logic
- **CSS3**: Modern, responsive styling
- **HTML5**: Semantic markup

### Browser Compatibility
- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## 📈 Understanding the Visualizations

### Weight Table
Shows the connection strengths between RGB inputs and the first hidden layer neurons. Colors indicate:
- **Blue**: Positive weights (strengthening connection)
- **Red**: Negative weights (weakening connection)  
- **Gray**: Neutral weights

### Training Insights
- **Color Sensitivity**: Which color channel the model focuses on
- **Weight Changes**: How much the model is learning
- **Learning Pattern**: Whether loss is decreasing and accuracy improving
- **Prediction Confidence**: How reliable the current predictions are

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow.js team for the excellent ML library
- The open source community for inspiration and tools
- Everyone who tests and provides feedback on the demo

---

**Happy Training! 🎨🧠** 