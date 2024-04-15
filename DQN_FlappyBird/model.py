



import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.actions  = 2 # UP and DO NOTHING

        self.conv1    = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2)
        self.bn1      = nn.BatchNorm2d(32)
        self.conv2    = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2      = nn.BatchNorm2d(64)
        self.map_size = (64, 16, 9)
        self.fc1      = nn.Linear(self.map_size[0]*self.map_size[1]*self.map_size[2], 256)
        self.fc2      = nn.Linear(256, self.actions)

    def forward(self, x):
        # if not isinstance(x, torch.Tensor):
        #     x = torch.tensor(x, dtype=torch.float)
        # x = x.permute(0, 3, 1, 2) ?
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = x.view(x.size()[0], -1)
        x = F.relu_(self.fc1(x))
        x = self.fc2(x)

        return x