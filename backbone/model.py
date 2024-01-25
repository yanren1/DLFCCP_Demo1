import torch
import torch.nn as nn
from typing import Callable, List, Optional
import random

class simpleMLP(nn.Module):
    def __init__(self,
                in_channels: int,
                hidden_channels: List[int],
                norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                inplace: Optional[bool] = None,
                bias: bool = True,
                dropout: float = 0.0,
                ):
        super(simpleMLP, self).__init__()
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))
        #sigmoid
        layers.append(torch.nn.Linear(hidden_channels[-1], hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)


    def forward(self, x):
        x = self.mlp(x)
        return x

class DummyModel(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DummyModel, self).__init__()

        #
        self.w1 = nn.Parameter(torch.randn(in_channels, 16))
        self.b1 = nn.Parameter(torch.randn(16))

        self.w2 = nn.Parameter(torch.randn(16, 32))
        self.b2 = nn.Parameter(torch.randn(32))

        self.w3 = nn.Parameter(torch.randn(32, 64))
        self.b3 = nn.Parameter(torch.randn(64))

        self.w4 = nn.Parameter(torch.randn(64, 8))
        self.b4 = nn.Parameter(torch.randn(8))

        self.w5 = nn.Parameter(torch.randn(in_channels, out_channels))
        self.b5 = nn.Parameter(torch.randn(out_channels))


        # Dummy operations
        self.operations = random.choices(
            [torch.sin,torch.cos,torch.tan,torch.square,torch.relu,torch.tanh,lambda x: x ** 2,lambda x: x ** 3],
            k=5)

    def forward(self, x):
        for op in self.operations:
            x = op(x)
        x = torch.matmul(x,self.w1) + self.b1

        x = torch.matmul(x, self.w2) + self.b2
        x = torch.matmul(x, self.w3) + self.b3
        x = torch.matmul(x, self.w4) + self.b4
        x = torch.matmul(x, self.w5) + self.b5

        x = torch.sigmoid(x)
        return x
