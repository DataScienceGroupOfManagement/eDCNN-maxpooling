import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from loguru import logger

class eDCNN_maxpooling_network(nn.Module):
    def __init__(self, input_dim=4,
                 num_layers=8,
                 in_channels=1,
                 out_channels=1,
                 kernel_size=3,
                 pooling_size=2,
                 bias_value=0.001,
                 output_dim=1):

        super(eDCNN_maxpooling_network, self).__init__()
        logger.info('eDCNN_maxpooling_network!')

        input_dim = int(input_dim)
        num_layers = int(num_layers)
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        pooling_size = int(pooling_size)
        output_dim = int(output_dim)
        kernel_size = int(kernel_size)
        padding_size = int(kernel_size - 1)  # zero padding

        self.layers_per_component = int(num_layers / 2)

        # the first component
        self.conv1d_first_list = torch.nn.ModuleList()
        self.conv1d_first_list.append( nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding_size, bias=False))
        self.bias_first_list = []
        self.bias_first_list.append(Variable(torch.tensor(bias_value), requires_grad=True))
        for layer_ids in range(self.layers_per_component - 1):
            self.conv1d_first_list.append(nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding_size, bias=False))
            self.bias_first_list.append(Variable(torch.tensor(bias_value), requires_grad=True))
        self.first_max_pooling = nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size, ceil_mode=True)

        feature_length =  input_dim + (kernel_size -1)  *  self.layers_per_component
        feature_length =  np.ceil(feature_length / pooling_size)

        # the second component
        self.conv1d_second_list = torch.nn.ModuleList()
        self.conv1d_second_list.append( nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding_size, bias=False))
        self.bias_second_list = []
        self.bias_second_list.append(Variable(torch.tensor(bias_value), requires_grad=True))

        for layer_ids in range(self.layers_per_component - 1):
            self.conv1d_second_list.append( nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding_size, bias=False))
            self.bias_second_list.append(Variable(torch.tensor(bias_value), requires_grad=True))

        self.second_max_pooling = nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size, ceil_mode=True)

        feature_length = feature_length + (kernel_size -1)  * self.layers_per_component
        feature_length = np.ceil(feature_length / pooling_size)

        self.Flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=int(feature_length), out_features=output_dim, bias=True)

    def forward(self, x):
        # first component
        for layer_id, layer in enumerate(self.conv1d_first_list):
            x = F.relu(layer(x) + self.bias_first_list[layer_id])
        x = self.first_max_pooling(x)
        # second component
        for layer_id, layer in enumerate(self.conv1d_second_list):
            x = F.relu(layer(x) + self.bias_second_list[layer_id])
        x = self.second_max_pooling(x)

        x = self.Flatten(x)
        x = self.fc(x)
        return x


class eDCNN_zeropadding_network(nn.Module):
    def __init__(self, input_dim=4,
                 num_layers=8,
                 in_channels=1,
                 out_channels=1,
                 kernel_size=3,
                 bias_value=0.001,
                 output_dim=1):

        super(eDCNN_zeropadding_network, self).__init__()
        logger.info('eDCNN_zeropadding_network!')

        input_dim = int(input_dim)
        num_layers = int(num_layers)
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        output_dim = int(output_dim)
        kernel_size = int(kernel_size)
        padding_size = int(kernel_size - 1)  # zero padding

        self.layers_per_component = int(num_layers / 2)

        # the first component
        self.conv1d_first_list = torch.nn.ModuleList()
        self.conv1d_first_list.append(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding_size,  bias=False))
        self.bias_first_list = []
        self.bias_first_list.append(Variable(torch.tensor(bias_value), requires_grad=True))

        for layer_ids in range(self.layers_per_component - 1):
            self.conv1d_first_list.append(
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,  padding=padding_size, bias=False))
            self.bias_first_list.append(Variable(torch.tensor(bias_value), requires_grad=True))

        feature_length = input_dim + (kernel_size - 1) * self.layers_per_component

        # the second component
        self.conv1d_second_list = torch.nn.ModuleList()
        self.conv1d_second_list.append(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,  padding=padding_size, bias=False))
        self.bias_second_list = []
        self.bias_second_list.append(Variable(torch.tensor(bias_value), requires_grad=True))

        for layer_ids in range(self.layers_per_component - 1):
            self.conv1d_second_list.append(
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,  padding=padding_size, bias=False))
            self.bias_second_list.append(Variable(torch.tensor(bias_value), requires_grad=True))

        feature_length = feature_length + (kernel_size - 1) * self.layers_per_component

        self.Flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=int(feature_length), out_features=output_dim, bias=True)

    def forward(self, x):
        # first component
        for layer_id, layer in enumerate(self.conv1d_first_list):
            x = F.relu(layer(x) + self.bias_first_list[layer_id])
        # second component
        for layer_id, layer in enumerate(self.conv1d_second_list):
            x = F.relu(layer(x) + self.bias_second_list[layer_id])

        x = self.Flatten(x)
        x = self.fc(x)
        return x

