import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import utils, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np


class inverse_model(nn.Module):
    def __init__(self, input_size=478*3,
                 label_size=13,
                 num_layer= 4,
                 d_hidden = 128,
                 use_bn=True,
                 skip_layer=1,
                 final_sigmoid = True
                 ):
        super(inverse_model, self).__init__()

        self.layers = []
        self.bns = []
        self.use_bn = use_bn
        self.final_sigmoid =final_sigmoid

        hidden_layers = [d_hidden]*num_layer
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_layers[0]))

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_layers[i+1]))

        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], label_size))

        # Convert Python lists to PyTorch modules
        self.layers = nn.ModuleList(self.layers)
        self.bns = nn.ModuleList(self.bns)

        self.skip_layer = skip_layer

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
            if self.use_bn:
                x = self.bns[i](x)
            if i == self.skip_layer:
                skip_value = x
            if i == self.skip_layer + 1:
                x = torch.add(x, skip_value)
        if self.final_sigmoid:
            x = torch.sigmoid(self.layers[-1](x))
        else:
            x = self.layers[-1](x)
        return x

    def loss(self, pred, target):
        return torch.mean((pred - target) ** 2)

    # def __init__(self, input_size=478*3,label_size=13, hid_size = 1024):
    #     super(inverse_model, self).__init__()
    #
    #     self.hid_size = hid_size
    #     self.inputsize = input_size
    #     self.outputsize = label_size
    #
    #     self.fc1 = nn.Linear(self.inputsize,  self.hid_size)
    #     self.fc2 = nn.Linear( self.hid_size,  self.hid_size)
    #     self.fc3= nn.Linear( self.hid_size,  self.hid_size)
    #     self.fc4 = nn.Linear( self.hid_size, self.outputsize)
    #
    #     self.bn1 = nn.BatchNorm1d( self.hid_size)
    #     self.bn2 = nn.BatchNorm1d( self.hid_size)
    #     self.bn3 = nn.BatchNorm1d( self.hid_size)
    #     # (2) no batch norm
    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = self.bn1(x)
    #     x2 = F.relu(self.fc2(x))
    #     x = self.bn2(x)
    #     x = F.relu(self.fc3(torch.add(x,x2)))
    #     x = self.bn3(x)
    #     x = torch.sigmoid(self.fc4(x))
    #     return x


if __name__ == "__main__":
    import time

    d_input = 60 * 3 * 3 + 6 * 2
    d_output = 6
    model = inverse_model(input_size=d_input,label_size=d_output,d_hidden=2048)
    model.eval()
    with torch.no_grad():
        input_data = torch.ones((1, d_input))
        run_times = 1000
        t0 = time.time()
        for i in range(run_times):
            output_data = model.forward(input_data)
            # print(output_data.shape)
        t1 = time.time()
        print(1/((t1-t0)/run_times))
        print(((t1-t0)/run_times))
