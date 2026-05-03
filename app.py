from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from tensorflow.keras.models import load_model


app = Flask(__name__)
CORS(app)



model = load_model("cnn_malaria_model.keras", compile=False)


def preprocess(image):
    # Resize to same size used in training
    image = cv2.resize(image, (64, 64))

    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    image = image.astype("float32") / 255.0

    
    image = np.expand_dims(image, axis=0)

    return image



@app.route("/")
def home():
    return "MalScan Backend Running 🚀"





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
    app.run(host="0.0.0.0", port=5000, debug=True)

