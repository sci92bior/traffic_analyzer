import torch
from torch import nn, Tensor


class LinearLayer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            bias: bool = True,
            w_init_gain: str = "linear",
    ):
        super(LinearLayer, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_layer(x)
        return x

class Dummy(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x
