import torch.nn as nn
from torch import sigmoid
from torch.nn.functional import relu


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc_encode = nn.Linear(8192, 4096)

        self.deconv1 = nn.ConvTranspose2d(512, 128, 2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.fc_decode = nn.Linear(64, 4096)

    def forward(self, x):
        # encode
        print(x.shape)
        out = relu(self.conv1(x))
        print(out.shape)
        out = self.pool(out)
        print(out.shape)
        out = relu(self.conv2(out))
        print(out.shape)
        out = self.pool(out)
        print(out.shape)
        out = out.view(out.size(0), -1)
        print(f"out shape: {out.shape}")
        out = sigmoid(self.fc_encode(out))

        # need to reshape into 2d first.
        # decode
        out = relu(self.deconv1(out))
        out = relu(self.deconv2(out))
        #out = out.view(out.size(0), -1)
        out = sigmoid(self.fc_decode(out))
        return out
