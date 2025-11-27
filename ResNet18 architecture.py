# ResNet18

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- Basic Residual Block (для ResNet-18/34) ---
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # Первый 3×3 conv
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Второй 3×3 conv
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # downsample (identity или 1×1 conv, stride=2)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=200):
        super(ResNet18, self).__init__()

        # conv1: 7×7, 64, stride 2
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # maxpool 3×3, stride 2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual stages (как в Table 1 ResNet-18):
        # conv2_x : 64 → 64, 2 blocks
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)

        # conv3_x : 64 → 128, 2 blocks
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)

        # conv4_x : 128 → 256, 2 blocks
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)

        # conv5_x : 256 → 512, 2 blocks
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # average pool → linear
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # выход заменён на 200 классов
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    # создаёт блоки внутри conv2_x … conv5_x
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None

        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))

        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 224×224 → 112×112
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)  # 56×56
        x = self.layer2(x)  # 28×28
        x = self.layer3(x)  # 14×14
        x = self.layer4(x)  # 7×7

        x = self.avgpool(x)  # 1×1
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

model = ResNet18(num_classes=200)
model.eval()

image = torch.zeros((1, 3, 224, 224))
output = model(image)

print("Output shape:", output.shape)
print("Trainable params:", count_trainable_parameters(model))
