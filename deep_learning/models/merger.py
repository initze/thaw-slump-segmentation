import torch
import torch.nn as nn
import torch.nn.functional as F

class Merger(nn.Module):
    def __init__(self, n_left, n_middle, n_right):
        super().__init__()
        self.init_left = nn.Linear(n_left, 4096)
        self.init_middle = nn.Linear(n_middle, 4096, bias=False)
        self.init_right = nn.Linear(n_right, 4096, bias=False)

        self.together1 = nn.Linear(4096, 1024)
        self.together2 = nn.Linear(1024, 128)
        self.together3 = nn.Linear(128, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, left, middle, right):
        left = self.init_left(left)
        middle = self.init_middle(middle)
        right = self.init_right(right)

        left = left
        right1 = middle + right
        right2 = torch.roll(right1, 1, dims=0)

        ret = []
        for right in [right1, right2]:
            x = self.relu(left + right)
            x = self.relu(self.together1(x))
            x = self.relu(self.together2(x))
            x = self.relu(self.together3(x))
            ret.append(x)
        return ret
