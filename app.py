import os
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.models import load_model


DEFAULT_CORS_ORIGINS = (
    "http://127.0.0.1:5000,"
    "http://localhost:5000,"
    "http://127.0.0.1:5500,"
    "http://localhost:5500"
)


def get_bool_env(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_int_env(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


app = Flask(__name__)

MODEL_PATH = Path(os.getenv("MALSCAN_MODEL_PATH", "cnn_malaria_model.keras"))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("MALSCAN_CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model file not found at '{MODEL_PATH}'. Set MALSCAN_MODEL_PATH to the model location."
    )

CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}})

model = load_model(MODEL_PATH, compile=False)


def preprocess(image):
    image = cv2.resize(image, (64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)


@app.route("/")
def home():
    return "MalScan Backend Running"


@app.route("/health")
def health():
    return jsonify({
        "success": True,
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded"
            })

        file = request.files["file"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No selected file"
            })

        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({
                "success": False,
                "error": "Invalid image file"
            })

        processed = preprocess(image)
        prediction = model.predict(processed, verbose=0)[0][0]

        if prediction >= 0.5:
            result = "Infected"
            confidence = prediction
        else:
            result = "Uninfected"
            confidence = 1 - prediction

        return jsonify({
            "success": True,
            "prediction": result,
            "confidence": round(float(confidence) * 100, 2),
            "model": "CNN"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(
        host=os.getenv("MALSCAN_HOST", "127.0.0.1"),
        port=get_int_env("MALSCAN_PORT", 5000),
        debug=get_bool_env("MALSCAN_DEBUG", False),
    )
