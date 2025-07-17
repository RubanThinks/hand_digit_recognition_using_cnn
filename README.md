# 🖐️ Handwritten Digit Recognition using CNN

Welcome to the **Hand Digit Recognition Using CNN** project!  
This repository provides a complete pipeline to recognize handwritten digits (0-9) using a Convolutional Neural Network (CNN) in Python. You can train the model, visualize its performance, and deploy a fun interactive web app where you can draw your own digit and get instant predictions!

---

## 🚀 Features

- 🎯 **Accurate CNN Model:** Built with TensorFlow/Keras, trained on MNIST dataset.
- 🖼️ **Interactive Web App:** Draw a digit in your browser and let the model predict it (Streamlit powered).
- 📊 **Visualization:** Confusion matrices and prediction heatmaps to understand model performance.
- 💾 **End-to-End Workflow:** From training to deployment, all code and resources included.
- 📦 **Easy Setup:** All dependencies listed in `requirements.txt`.

---

## 🗂️ Project Structure

```
hand_digit_recognition_using_cnn/
├── app.py             # Streamlit web app for drawing and prediction
├── main.py            # Model training and evaluation
├── final.h5           # Trained model weights (TensorFlow/Keras HDF5 format)
├── requirements.txt   # List of Python dependencies
└── README.md          # Project documentation
```

---

## ⚡ Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/RubanThinks/hand_digit_recognition_using_cnn.git
cd hand_digit_recognition_using_cnn
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional)

If you want to retrain the model, simply run:
```bash
python main.py
```
This will train the CNN on the MNIST dataset and save the model as `final.h5`.

### 4. Run the Web App

```bash
streamlit run app.py
```

Draw a digit (0–9) in the box and click **Predict** to see the model's prediction!

---

## 🧠 Model Architecture

- **Input:** 28x28 grayscale images
- **Layers:**
  - Conv2D (32 filters, 3x3, ReLU)
  - MaxPooling2D
  - Conv2D (32 filters, 3x3, ReLU)
  - MaxPooling2D
  - Flatten
  - Dense (128, sigmoid)
  - Output Dense (10, softmax)

---

## 🎨 Web App Demo

The interactive app (in `app.py`) lets you draw a digit and see the model's guess in real-time.

![Draw a Digit Streamlit App Example](https://user-images.githubusercontent.com/your-demo-image-path.png)

---

## 📈 Model Performance

- Achieves **>99% accuracy** on MNIST test set.
- Confusion matrix and visualizations included in `main.py`.

---

## 🛠️ Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies:
  - tensorflow-cpu
  - numpy
  - pillow
  - matplotlib
  - seaborn
  - streamlit
  - streamlit-drawable-canvas

---

## 🤝 Contributing

Pull requests and feedback are welcome! Please open an issue for suggestions or improvements.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow](https://www.tensorflow.org/)
- [Streamlit](https://streamlit.io/)
- [Streamlit Drawable Canvas](https://github.com/andfanilo/streamlit-drawable-canvas)

---

> Made with ❤️ by [RubanThinks](https://github.com/RubanThinks)
