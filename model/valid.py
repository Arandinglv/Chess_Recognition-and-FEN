import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import torch.nn as nn
from model import MyResNet50
from dataloader import ChassDataset


batch_size = 32
data_dir = "../dataset/valid_ImageDataset"
model_path = "./chess_piece_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 13
channels = 1


if channels == 1:
    transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=channels),  # 三通道灰度图
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
else:  # 三通道
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



valid_dataset = ChassDataset(data_dir, transform=transforms)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

model = MyResNet50(num_classes=num_classes, channel=channels).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

criterion = nn.CrossEntropyLoss()
total_correct = 0
valid_loss = 0

with torch.no_grad():
    for images, labels in valid_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        valid_loss += loss.item() * images.size(0)

        # Accuracy
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()


average_valid_loss = valid_loss / len(valid_dataset)
average_valid_acc = 100 * total_correct / len(valid_dataset)

print(f"Validation loss: {average_valid_loss: .4f}")
print(f"Validation Acc: {average_valid_acc:.2f}%")
