import torch
import torch.nn as nn
from classes.classifier import CatClassifier
import torch.optim as optim

from utils.epoch import train_one_epoch, validate_one_epoch
from utils.transformers import train_loader, val_loader

MODEL_VERSION = open("version.txt").read().strip()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = 2
    model = CatClassifier(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    num_epochs = 20

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device)

        print(f"""
          Epoch [{epoch+1}/{num_epochs}]
          loss train: {train_loss:.4f} accuracy train: {train_acc*100:.2f}%
          loss val: {val_loss:.4f} accuracy val: {val_acc*100:.2f}%")
        """)

    save_path = f"./models/cat_classifier_{MODEL_VERSION}.pth"
    save_and_bump_model_version()

    torch.save(model.state_dict(), save_path)

    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train()


def save_and_bump_model_version():
    with open("version.txt", "w") as f:
        f.write(str(int(MODEL_VERSION) + 1))
