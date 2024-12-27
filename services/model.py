import os
import time
import torch
from flask import Flask, request, jsonify
from PIL import Image
import tempfile
import torchvision.transforms as T

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from io import BytesIO
from utils.predict import predict_single_image
from model_classes.classifier import CatClassifier

val_transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app = Flask(__name__)

with open("version.txt", "r") as f:
    version_str = f.read().strip()
model_version = int(version_str) - 1

checkpoint_path = f"./models/cat_classifier_{model_version}.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_classes = 3
model = CatClassifier(num_classes=num_classes).to(device)

try:
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded model weights from {checkpoint_path}")
except FileNotFoundError:
    print(f"Model checkpoint {checkpoint_path} not found")
    raise SystemExit(1)

model.eval()

class_idx_to_name = {0: "coco", 1: "none", 2: "rico"}

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image file provided in 'file' field"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Read the uploaded file into memory
    file_bytes = file.read()

    # Load the image from memory
    image = Image.open(BytesIO(file_bytes))

    start_time = time.time()

    # Run inference
    pred_class_idx, probabilities = predict_single_image(
        model, image, device, val_transform
    )

    pred_class_name = class_idx_to_name[pred_class_idx]
    confidence = probabilities[pred_class_idx]

    response = {
        "pred_class": pred_class_name,
        "confidence": float(confidence),
        "probabilities": {
            class_idx_to_name[i]: float(prob) for i, prob in enumerate(probabilities)
        },
        "inference_time": round(time.time() - start_time, 4)
    }
    
    return jsonify(response)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Model service running"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
