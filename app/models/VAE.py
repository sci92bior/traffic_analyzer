import torch.nn as nn
from torch import Tensor
import torch
from app.models.Layers import LinearLayer, Dummy


class Encoder(nn.Module):
    def __init__(self, start_dim, hidden_dim, step_size, num_of_layers, latent_dim):
        super(Encoder, self).__init__()

        self.lin_layers = nn.ModuleList(
            [
                nn.Sequential(
                    LinearLayer(
                        start_dim if i == 0 else hidden_dim if i == 1 else int(hidden_dim / pow(step_size, i - 1)),
                        hidden_dim if i == 0 else int(hidden_dim / pow(step_size, i)),
                        w_init_gain="leaky_relu"
                    ),
                    Dummy() if i ==0 else nn.BatchNorm1d(hidden_dim if i == 0 else int(hidden_dim / pow(step_size, i))),
                    nn.LeakyReLU(),
                )
                for i in range(num_of_layers)
            ]
        )
        self.out = LinearLayer(int(hidden_dim / pow(step_size, num_of_layers - 1)),
                               int(hidden_dim / pow(step_size, num_of_layers)))
        self.fc_mu = LinearLayer(int(hidden_dim / pow(step_size, num_of_layers)), latent_dim)
        self.fc_var = LinearLayer(int(hidden_dim / pow(step_size, num_of_layers)), latent_dim)

    def reparametrize(self,mu,logvar,training = True):
        """
           Reparameterization trick to sample from N(mu, var) from
           N(0,1).
           :param mu: (Tensor) Mean of the latent Gaussian [B x D]
           :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
           :return: (Tensor) [B x D]
           """
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def forward(self, x: Tensor) -> Tensor:
        for idx, lin in enumerate(self.lin_layers):
            x = lin(x)
        x = self.out(x)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        z = self.reparametrize(mu,logvar,self.training)
        return mu, logvar, z


class Decoder(nn.Module):
    def __init__(self, start_dim, hidden_dim, step_size, num_of_layers,latent_dim):
        super(Decoder, self).__init__()
        self.fc_input = LinearLayer(latent_dim,int(hidden_dim / pow(step_size, num_of_layers)))
        self.lin_layers = nn.ModuleList(
            [
                nn.Sequential(
                    LinearLayer(
                        int(hidden_dim / pow(step_size, num_of_layers)) if i == 0 else int(
                            hidden_dim / pow(step_size, (num_of_layers - i))),
                        int(hidden_dim / pow(step_size, (num_of_layers - i) - 1)),
                        w_init_gain="leaky_relu"
                    ),
                    nn.LeakyReLU(),
                )
                for i in range(num_of_layers)
            ]
        )
        self.out = LinearLayer(int(hidden_dim),
                               start_dim)
    def sample(self,num_samples, device):
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(device)

        samples = self.decode(z)
        return samples

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc_input(x)
        for idx, lin in enumerate(self.lin_layers):
            x = lin(x)
        x = self.out(x)
        return x
