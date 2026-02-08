
import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# --- Class Names ---
class_names = [
      'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)',
    'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)',
    'Speed limit (120km/h)', 'No passing', 'No passing for vehicles over 3.5 metric tons',
    'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
    'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left',
    'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
    'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
    'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead',
    'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
    'Keep left', 'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons']





# --- Confidence Threshold ---
CONFIDENCE_THRESHOLD = 0.7  # You can adjust this value as needed

# --- Load Keras Model ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.h5")
model = load_model(MODEL_PATH)

# --- Image Preprocessing for Keras Model ---
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((30, 30))
    img_array = np.array(img) / 255.0
    if img_array.shape != (30, 30, 3):
        raise ValueError("Image must be 30x30 RGB.")
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Prediction Function ---
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    pred_prob = model.predict(img_array)
    pred_class = int(np.argmax(pred_prob))
    confidence = float(np.max(pred_prob))
    predicted_name = class_names[pred_class]
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "prediction": "Low confidence",
            "message": "The model is not confident in its prediction.",
            "confidence": round(confidence, 3)
        }
    return {
        "prediction": predicted_name,
        "class_id": pred_class,
        "confidence": round(confidence, 3)
    }

# --- Flask App ---
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

@app.route("/")
def home():
    return render_template("Home.html")

@app.route("/about")
def about_page():
    return render_template("About.html")

@app.route("/contact")
def contact_page():
    return render_template("Contact.html")

@app.route("/methodology")
def methodology_page():
    return render_template("Methodology.html")

@app.route("/datasets")
def datasets_page():
    return render_template("Datasets.html")

@app.route("/flowchart.jpg")
def flowchart():
    return send_from_directory(".", "flowchart.jpg")

@app.route("/upload")
def upload_page():
    return render_template("Upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    temp_path = os.path.join("temp.jpg")
    file.save(temp_path)


    try:
        result = predict_image(temp_path)
        os.remove(temp_path)
        return jsonify(result)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Server running at: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
