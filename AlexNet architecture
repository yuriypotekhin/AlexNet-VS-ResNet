# AlexNet

import torch.nn as nn
import torch.nn.functional as F
import torch

def count_trainable_parameters(model):
    """
    Counts the total number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # conv layer (224 * 224 * 3 -> 55 * 55 * 48)
        self.fc1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2, dilation=1)

        # conv layer + max_pooling 2*2 : (55 * 55 * 48 -> 27 * 27 * 128)
        self.fc2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2, dilation=1)

        # cross conv layers + max_pooling 2*2 : (27 * 27 * 128 -> 13 * 13 * 192)
        self.fc3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1, dilation=1)

        # conv layers (13 * 13 * 192 -> 13 * 13 * 192)
        self.fc4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1)

        # conv layers (13 * 13 * 192 -> 13 * 13 * 128)
        self.fc5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1)

        # Lin
        self.fc6 = nn.Linear(6 * 6 * 256, 4096)

        # Lin
        self.fc7 = nn.Linear(4096, 4096)

        #Lin
        self.fc8 = nn.Linear(4096, 200)

        self.dropout = nn.Dropout(p=0.5)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        # flatten image input
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = self.maxpool(x)
        x = F.relu(self.fc2(x))
        x = self.maxpool(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.maxpool(x)

        x = x.view(-1, 6 * 6 * 256)
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.dropout(F.relu(self.fc7(x)))
        x = F.relu(self.fc8(x))
        return x

model = AlexNet()
model.eval()

print("Output shape:", output.shape)
print("Trainable params:", count_trainable_parameters(model))
