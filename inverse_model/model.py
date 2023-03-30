import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import utils, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np


class inverse_model(nn.Module):

    def __init__(self, input_size=478*3,label_size=13):
        super(inverse_model, self).__init__()

        self.inputsize = input_size
        self.outputsize = label_size

        self.fc1 = nn.Linear(self.inputsize, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3= nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, self.outputsize)

        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.bn3 = nn.BatchNorm1d(4096)
        # (2) no batch norm
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x2 = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.relu(self.fc3(torch.add(x,x2)))
        x = self.bn3(x)
        x = torch.sigmoid(self.fc4(x))
        return x

    def loss(self, pred, target):
        return torch.mean((pred - target) ** 2)

if __name__ == "__main__":
    import time


    model = inverse_model(input_size=478* 3,label_size=13)
    model.eval()
    with torch.no_grad():
        input_data = torch.ones((1, 478* 3))
        run_times = 1000
        t0 = time.time()
        for i in range(run_times):
            output_data = model.forward(input_data)
            # print(output_data.shape)
        t1 = time.time()
        print(1/((t1-t0)/run_times))
        print(((t1-t0)/run_times))
