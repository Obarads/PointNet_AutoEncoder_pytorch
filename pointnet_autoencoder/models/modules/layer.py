import torch
from torch import nn

class Layers(nn.Module):
    def __init__(self, layer, norm, act):
        super(Layers, self).__init__()
        module_list = []
        module_list.append(layer)
        module_list.append(norm)
        if act is not None:
            module_list.append(act)

        self.mlp = nn.Sequential(
            *module_list
        )

    def forward(self, x):
        return self.mlp(x)

class MLP1D(Layers):
    def __init__(self, in_channels, out_channels, act=nn.ReLU(inplace=True)):
        conv = nn.Conv1d(in_channels, out_channels, 1)
        norm = nn.BatchNorm1d(out_channels)
        super(MLP1D, self).__init__(conv, norm, act)

class MLP2D(Layers):
    def __init__(self, in_channels, out_channels, act=nn.ReLU(inplace=True)):
        conv = nn.Conv2d(in_channels, out_channels, (1,1))
        norm = nn.BatchNorm2d(out_channels)
        super(MLP2D, self).__init__(conv, norm, act)

class Linear(Layers):
    def __init__(self, in_features, out_features, act=nn.ReLU(inplace=True)):
        layer = nn.Linear(in_features, out_features)
        norm = nn.BatchNorm1d(out_features)
        super(Linear, self).__init__(layer, norm, act)
