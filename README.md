# Neural Network Colour Preference Demo

An interactive web app to demonstrate how a neural network learns user preferences.

🖱️ **Users pick their preferred colour from two options.**
📈 The app trains a small neural network in real-time using TensorFlow.js.
🔮 Once trained, the app predicts future colour preferences.

---

## 🌐 Live Demo
Hosted on **GitHub Pages**: [link to come]

---

## 🚀 Features
- **Training Phase**: User picks preferred colours; network updates weights live.
- **Inference Phase**: Network predicts which colour the user prefers.
- **Visualization**: Shows weight changes and predictions.
- **100% Client-side**: No backend required.

---

## 🛠 Tech Stack

| Technology        | Purpose                               |
|------------------|---------------------------------------|
| HTML/CSS          | Layout and styling                    |
| JavaScript        | App logic                             |
| [TensorFlow.js](https://www.tensorflow.org/js) | Neural network training/inference |
| GitHub Pages / Netlify | Free static site hosting           |

---

## 📂 Project Structure

```
colour-preference-demo/
├── index.html          # Main HTML file
├── style.css           # Styling for layout and UI
├── app.js              # Main JavaScript logic (uses TensorFlow.js)
├── assets/             # Images, icons, etc.
├── README.md           # This file
└── .gitignore          # Ignore node_modules, etc.
```

---

## 📋 Requirements

- Modern web browser (Chrome, Firefox, Edge)
- No backend or server – 100% static site.
- For local development, a simple HTTP server like VSCode Live Server or Python's `http.server`.

---

## 🚧 Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/colour-preference-demo.git
   cd colour-preference-demo
   ```

2. **Open `index.html` locally**
   Or run a local server:
   ```bash
   python3 -m http.server
   ```

3. **Start coding!**
   - Modify `app.js` to tweak network size, training logic, or UI.

---

## 📡 Hosting

### GitHub Pages
1. Push to `main` branch.
2. Go to **Settings > Pages**.
3. Set source to `main` and `/ (root)`.
4. Visit `https://your-username.github.io/colour-preference-demo/`



---

## 💡 Future Enhancements
- Add 2D visualization of decision boundary.
- Support more than two colour options.


---

## 📖 License
MIT License © 2025 