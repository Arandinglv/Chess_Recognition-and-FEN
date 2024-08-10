import torch
import os
from torchvision import transforms
from dataloader import ChassDataset
from torch.utils.data import DataLoader
from model import MyResNet50
import torch.nn as nn
import torch.optim as optim
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

num_epochs = 10
batch_size = 32
learning_rate = 0.0001

data_dir = "../dataset/ImageDataset"
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

train_dataset = ChassDataset(data_dir, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

model = MyResNet50(num_classes=num_classes, channel=channels).to(device)  # model -> gpu
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_dataloader)

for epoch in range(num_epochs):
    total_correct = 0
    total_samples = 0

    for i, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)  # data -> gpu
        labels = labels.to(device)  # data -> gpu

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Calculate accuracy
        _, predicted = torch.max(outputs.detach(), 1)  # outputs是[batch_size, classes], 返回最大分数的索引
        # 不推荐用output.data, 此外, 1代表维度, 在这里是类别维度
        total_samples += labels.size(0)  # 是获取当前批次中的样本数量
        total_correct += (predicted == labels).sum().item()  # sum()显示的是标量张量, item可以返回Python数值类型

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            accuracy = 100 * total_correct / total_samples
            print(
                f"Epoch [{epoch + 1} / {num_epochs}], Step [{i + 1} / {total_step}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")


torch.save(model.state_dict(), "chess_piece_model1.pth")

