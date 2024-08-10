import torch
import torch.nn as nn

class Conv_group(nn.Module):
    def __init__(self, in_channel):
        super(Conv_group, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        return x


