from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import joblib

app = Flask(__name__)
CORS(app)  # allow frontend access

# 🔥 Load your BEST trained model
model = joblib.load("malaria_model.pkl")

MODEL_NAME = "Best Model"  # update if you want (e.g., "Random Forest")

# 🧠 Preprocessing
def preprocess(image):
    image = cv2.resize(image, (64, 64))
    image = image.flatten() / 255.0
    return image.reshape(1, -1)

# ✅ Home route
@app.route("/")
def home():
    return "MalScan Backend Running 🚀"

# 🔍 Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check file
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty file"}), 400

        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image"}), 400

        # Preprocess
        processed = preprocess(image)

        # Prediction
        pred = model.predict(processed)[0]
        result = "Infected" if pred == 1 else "Uninfected"

        # 🔥 Confidence (if model supports it)
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = float(np.max(model.predict_proba(processed)))

        return jsonify({
            "prediction": result,
            "confidence": confidence,
            "model": MODEL_NAME
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ▶️ Run server
if __name__ == "__main__":
    app.run(debug=True)