import torch
from PIL import Image
import torch.nn.functional as F

def predict_single_image(model, image_path, device, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)

        probs = F.softmax(outputs, dim=1)

        _, predicted_class = torch.max(probs, dim=1)

    return predicted_class.item(), probs.squeeze().cpu().numpy()