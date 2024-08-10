from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import os

class ChassDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  #
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = os.listdir(root_dir)
        self.label_class = []  # 类别和文件夹的对应关系
        # self.label_image_table = []  # 用来逐个记录 其实没啥用

        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(root_dir, class_name)
            self.label_class.append([label, class_name])
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)
                # self.label_image_table.append([label, class_name])

        with open("label_class.txt", "w") as file:
            for label, class_name in self.label_class:
                file.write(f"{label}: {class_name} \n")



    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]


        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        # 来确定它应该从数据集中读取多少个样本
        return len(self.image_paths)

