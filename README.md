# DECAM: Detect Malaria Using Infected Blood Samples

# MalScan - Automated Malaria Detection Using Blood Cell Images

MalScan is an AI-powered web application that detects malaria-infected blood cells from microscopic cell images using a deep learning CNN model.

Users can upload or drag-and-drop a blood cell image, and the system predicts whether the sample is:

- Infected
- Uninfected

<img width="1902" height="778" alt="image" src="https://github.com/user-attachments/assets/1c46d534-e651-426b-97fc-e619e6863db5" />

## Features

- Modern frontend UI with HTML, CSS, and JavaScript
- Drag-and-drop image upload
- CNN-based malaria detection
- Real-time prediction
- Confidence score display
- Flask backend API
- Lightweight local deployment
- Medical awareness section
- India helpline and nearby test center links

## Model Used

This project uses a Convolutional Neural Network (CNN) trained on malaria cell images.

### Model Architecture

- Conv2D + MaxPooling
- Conv2D + MaxPooling
- Conv2D + MaxPooling
- Flatten
- Dense Layer
- Dropout
- Sigmoid Output

## Dataset Used

Malaria Cell Images Dataset by NIH

Source:
https://ceb.nlm.nih.gov/repositories/malaria-datasets/

Also available on Kaggle:
https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

## Tech Stack

### Frontend

- HTML5
- CSS3
- JavaScript

### Backend

- Python
- Flask
- Flask-CORS

### Machine Learning / Deep Learning

- TensorFlow
- Keras
- OpenCV
- NumPy

## Backend Configuration

The Flask backend can be configured with environment variables so local development and deployment do not require source-code edits.

| Variable | Default | Purpose |
|---|---|---|
| `MALSCAN_MODEL_PATH` | `cnn_malaria_model.keras` | Path to the trained Keras model file. |
| `MALSCAN_CORS_ORIGINS` | `http://127.0.0.1:5000,http://localhost:5000,http://127.0.0.1:5500,http://localhost:5500` | Comma-separated frontend origins allowed to call the API. |
| `MALSCAN_HOST` | `127.0.0.1` | Host used when running `python app.py`. |
| `MALSCAN_PORT` | `5000` | Port used when running `python app.py`. |
| `MALSCAN_DEBUG` | `false` | Set to `true` only for local debugging. |

Example local run:

```bash
python app.py
```

Example with explicit settings:

```bash
MALSCAN_MODEL_PATH=cnn_malaria_model.keras MALSCAN_DEBUG=true python app.py
```

Health check:

```bash
curl http://127.0.0.1:5000/health
```

## Project Structure

```bash
MalScan/
|-- app.py
|-- index.html
|-- cnn_malaria_model.keras
|-- requirements.txt
|-- README.md
```
