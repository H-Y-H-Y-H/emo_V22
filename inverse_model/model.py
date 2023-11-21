import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import utils, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np


class inverse_model(nn.Module):
    def __init__(self,
                 input_size=478*3,
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


# class ImprovedInverseModel(nn.
import torch
import torch.nn as nn
from torch.nn import Transformer

class TransformerInverse(nn.Module):
    def __init__(self,
                encoder_input_size = 6,
                decoder_input_size = 60,
                output_size = 6,
                nhead = 2     ,
                num_encoder_layers = 2  ,
                num_decoder_layers = 2  ,
                dim_feedforward = 512
                 ):
        super(TransformerInverse, self).__init__()

        self.encoder_embedding = nn.Linear(encoder_input_size, dim_feedforward)
        self.decoder_embedding = nn.Linear(decoder_input_size, dim_feedforward)
        self.transformer = Transformer(d_model=dim_feedforward, nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       batch_first=True)  # Set batch_first to True
        self.output_layer = nn.Linear(dim_feedforward, output_size)

    def forward(self, encoder_input, decoder_input):
        encoder_input = self.encoder_embedding(encoder_input)
        decoder_input = self.decoder_embedding(decoder_input)

        output = self.transformer(encoder_input, decoder_input)
        output = self.output_layer(output[:, -1, :])  # Adjusted for batch_first
        return output


if __name__ == "__main__":
    import time

    # Example Usage
    encoder_input_size = 6
    decoder_input_size = 60
    output_size = 6
    nhead = 2
    num_encoder_layers = 2
    num_decoder_layers = 2
    dim_feedforward = 512

    model = TransformerInverse(encoder_input_size, decoder_input_size, output_size, nhead, num_encoder_layers,
                              num_decoder_layers, dim_feedforward)

    # Adjust inputs for batch_first
    # For batch size = 1
    encoder_input = torch.rand(1, 2, 6)  # Example encoder input
    decoder_input = torch.rand(1, 3, 60)  # Example decoder input

    output = model(encoder_input, decoder_input)
    print(output)


    # d_input = 60 * 3 * 1 + 6 * 2
    # d_output = 6
    # model = inverse_model(input_size=d_input,label_size=d_output,d_hidden=2048)
    # model.eval()
    # with torch.no_grad():
    #     input_data = torch.ones((1, d_input))
    #     run_times = 1000
    #     t0 = time.time()
    #     for i in range(run_times):
    #         output_data = model.forward(input_data)
    #         # print(output_data.shape)
    #     t1 = time.time()
    #     print(1/((t1-t0)/run_times))
    #     print(((t1-t0)/run_times))
