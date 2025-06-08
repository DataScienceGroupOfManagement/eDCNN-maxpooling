
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from loguru import logger

class ContractDeepConvolutionalNeuralNetworkSingleChannelsInitialBias(nn.Module):
    def __init__(self, input_dim, num_layers=5, in_channels=1, out_channels=1, kernel_size=3, output_dim=1,  bias_value=0.01):
        super(ContractDeepConvolutionalNeuralNetworkSingleChannelsInitialBias, self).__init__()

        input_dim = int(input_dim)
        num_layers = int(num_layers)
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        output_dim = int(output_dim)
        kernel_size = int(kernel_size)

        self.first_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0, bias=False)
        self.first_bias = Variable(torch.tensor(bias_value), requires_grad=True)

        self.conv1d_layers = torch.nn.ModuleList()
        for layer_ids in range(num_layers - 1):
            self.conv1d_layers.append(
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0, bias=False))

        self.bias_layers = []
        for layer_ids in range(num_layers - 1):
            self.bias_layers.append(Variable(torch.tensor(bias_value), requires_grad=True))

        self.Flatten = nn.Flatten()
        feature_length = out_channels * (input_dim - (kernel_size - 1) * num_layers)
        self.fc = nn.Linear(in_features=feature_length, out_features=output_dim, bias=True)

    def forward(self, x):
        x = self.first_layer(x) + self.first_bias
        x = F.relu(x)
        for layer_id, layer in enumerate(self.conv1d_layers):
            x = F.relu(layer(x) + self.bias_layers[layer_id])
        x = self.Flatten(x)
        x = self.fc(x)
        return x

# cDCNN-fc, single channel
class ContractDeepConvolutionalNeuralNetworkFullyConnectedSingleChannelsInitialBias(nn.Module):
    def __init__(self, input_dim, num_layers=5, num_fc_layer=1, in_channels=1, out_channels=1, kernel_size=3, output_dim=1,
                 bias_value=0.01):
        super(ContractDeepConvolutionalNeuralNetworkFullyConnectedSingleChannelsInitialBias, self).__init__()

        input_dim = int(input_dim)
        num_layers = int(num_layers)
        num_fc_layer = int(num_fc_layer)
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        output_dim = int(output_dim)
        kernel_size = int(kernel_size)

        self.first_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     padding=0, bias=False)
        self.first_bias = Variable(torch.tensor(bias_value), requires_grad=True)

        self.conv1d_layers = torch.nn.ModuleList()
        for layer_ids in range(num_layers - 1):
            self.conv1d_layers.append(
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0,
                          bias=False))

        self.bias_layers = []
        for layer_ids in range(num_layers - 1):
            self.bias_layers.append(Variable(torch.tensor(bias_value), requires_grad=True))

        self.Flatten = nn.Flatten()
        feature_length = out_channels * (input_dim - (kernel_size - 1) * num_layers)

        # fc layer after flatten
        self.final_fc_layers = torch.nn.ModuleList()
        for layer_ids in range(num_fc_layer):
            self.final_fc_layers.append(nn.Linear(in_features=feature_length, out_features=feature_length))

        self.fc = nn.Linear(in_features=feature_length, out_features=output_dim, bias=True)

    def forward(self, x):
        x = self.first_layer(x) + self.first_bias
        x = F.relu(x)
        for layer_id, layer in enumerate(self.conv1d_layers):
            x = F.relu(layer(x) + self.bias_layers[layer_id])
        x = self.Flatten(x)
        for layer in self.final_fc_layers:
            x = F.relu(layer(x))
        x = self.fc(x)
        return x
