from utils.transformers import train_dataset, val_transform
from classes.classifier import CatClassifier
import torch
import time

from utils.predict import predict_single_image

START_TIME = time.time()


MOST_RECENT_MODEL_VERSION = open("version.txt").read().strip()
CHECKPOINT_PATH = f"cat_classifier_{int(MOST_RECENT_MODEL_VERSION) - 1}.pth"
IMAGE_PATH = "coco.jpeg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_classes = 2
model = CatClassifier(num_classes=num_classes).to(device)

try:
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    print(f"'{CHECKPOINT_PATH}' model weights loaded")
except FileNotFoundError:
    print(f"err '{CHECKPOINT_PATH}' 404")
    exit(1)

model.eval()


class_idx_to_name = {v: k for k, v in train_dataset.class_to_idx.items()}

pred_class_idx, probabilities = predict_single_image(
    model, IMAGE_PATH, device, val_transform
)

pred_class_name = class_idx_to_name[pred_class_idx]
confidence = probabilities[pred_class_idx]

print(f"Predicted class: {pred_class_name}")
print(f"Confidence: {confidence:.4f}")

print("Class probabilities:")
for idx, prob in enumerate(probabilities):
    class_name = class_idx_to_name[idx]
    print(f"  {class_name}: {prob:.4f}")

print(f"Time: {time.time() - START_TIME:.2f}s")
