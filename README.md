# DECAM : Detect Malaria using infected blood samples
# 🩸 MalScan - Automated Malaria Detection Using Blood Cell Images

MalScan is an AI-powered web application that detects **malaria-infected blood cells** from microscopic cell images using **Deep Learning (CNN)**.

Users can upload or drag-and-drop a blood cell image, and the system predicts whether the sample is:

✅ **Infected**  
✅ **Uninfected**

---

# Features

- Modern Frontend UI (HTML, CSS, JavaScript)
- Drag & Drop Image Upload
- CNN-Based Malaria Detection
- Real-Time Prediction
- Confidence Score Display
- Flask Backend API
- Fast and Lightweight Deployment
- Medical Awareness Section
- India Helpline / Nearby Test Center Links

---

# Model Used

This project uses a **Convolutional Neural Network (CNN)** trained on malaria cell images.

### Model Architecture:

- Conv2D + MaxPooling
- Conv2D + MaxPooling
- Conv2D + MaxPooling
- Flatten
- Dense Layer
- Dropout
- Sigmoid Output

---

# 📂 Dataset Used

Malaria Cell Images Dataset by NIH

Source:  
https://ceb.nlm.nih.gov/repositories/malaria-datasets/

Also available on Kaggle:

https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria


---

# 🛠️ Tech Stack

## Frontend

- HTML5
- CSS3
- JavaScript

## Backend

- Python
- Flask
- Flask-CORS

## Machine Learning / Deep Learning

- TensorFlow
- Keras
- OpenCV
- NumPy

---

# 📁 Project Structure

```bash
MalScan/
│── app.py
│── index.html
│── style.css
│── script.js
│── cnn_malaria_model.keras
│── requirements.txt
│── README.md
