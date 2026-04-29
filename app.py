from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load trained CNN model
model = load_model("cnn_malaria_model.h5")

# -----------------------------------
# Preprocess uploaded image
# -----------------------------------
def preprocess(image):
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------------
# Home route
# -----------------------------------
@app.route("/")
def home():
    return "MalScan Backend Running 🚀"

# -----------------------------------
# Prediction route
# -----------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]

        # Convert image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image"})

        # Preprocess
        processed = preprocess(image)

        # Prediction
        prob = model.predict(processed)[0][0]

        if prob > 0.5:
            result = "Infected"
            confidence = prob
        else:
            result = "Uninfected"
            confidence = 1 - prob

        return jsonify({
            "prediction": result,
            "confidence": round(float(confidence) * 100, 2),
            "model": "CNN"
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# -----------------------------------
# Run server
# -----------------------------------
if __name__ == "__main__":
    app.run(debug=True)