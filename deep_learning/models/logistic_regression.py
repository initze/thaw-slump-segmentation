import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.regress = nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, x):
        return self.regress(x)
