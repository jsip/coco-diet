import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
import torch

train_transform = T.Compose([
    T.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = 'images'

train_dataset = ImageFolder(root=data_dir, transform=train_transform)
val_dataset = ImageFolder(root=data_dir, transform=val_transform)

labels = [label for _, label in train_dataset.samples]
class_counts = Counter(labels)

target_count = 165
weights = []

for label in labels:
    weights.append(target_count / class_counts[label])

weights = torch.DoubleTensor(weights)

sampler = WeightedRandomSampler(
    weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=32,
                          sampler=sampler, num_workers=2)
val_loader = DataLoader(val_dataset,   batch_size=32,
                        shuffle=False, num_workers=2)
