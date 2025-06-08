import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepFullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, output_dim=1):
        super(DeepFullyConnectedNetwork, self).__init__()
        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        num_layers = int(num_layers)
        output_dim = int(output_dim)

        self.first_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.layers = torch.nn.ModuleList()
        for layer_ids in range(num_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.first_layer(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.fc(x)
        return x
