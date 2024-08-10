import torchvision.models as models
from modules import Conv_group
import torch.nn as nn
import torch


class MyResNet50(nn.Module):
    def __init__(self, num_classes=13, channel=1):
        super(MyResNet50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.stage = nn.Sequential(*list(resnet50.children())[:-2])
        # 单通道受输入: self.stage[0]
        if channel == 1:
            self.stage[0] = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.new_conv1 = Conv_group(2048)
        self.new_conv2 = Conv_group(256)
        self.new_conv3 = Conv_group(256)

        self.last_conv = nn.Conv2d(2048 + 256 + 256 + 256, out_channels=256, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        stage_out = self.stage(x)
        conv1_out = self.new_conv1(stage_out)
        conv2_out = self.new_conv2(conv1_out)
        conv3_out = self.new_conv3(conv2_out)

        group_x = torch.cat((stage_out, conv1_out, conv2_out, conv3_out), dim=1)
        group_out = self.last_conv(group_x)
        x = self.avgpool(group_out)
        out = torch.flatten(x, 1)  # start_dim (int): the first dim to flatten
        out = self.fc(out)

        return out

# # 测试模型
# if __name__ == "__main__":
#     model = MyResNet50(num_classes=1000)
#     x = torch.randn(1, 3, 224, 224)
#     output = model(x)
#     print(output.shape)
