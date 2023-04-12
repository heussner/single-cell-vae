import torch
from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from typing import List
from math import sqrt


class BetaVAE(BaseVAE):

    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        img_size: int = 128,
        hidden_dims: List = None,
        n_downsample: int = -1,
        inter_dim: int = 512,
        beta_range: List = [0.2, 4],
        gamma: float = 1000.0,
        max_capacity: int = 25,
        Capacity_max_iter: int = 1e5,
        loss_type: str = "B",
        likelihood_dist: str = "gauss",
        **kwargs
    ) -> None:
        super(BetaVAE, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.beta_max = beta_range[1]
        self.beta = beta_range[0]
        self.gamma = gamma
        self.loss_type = loss_type
        self.likelihood_dist = likelihood_dist
        self.C_max = torch.torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.last_hidden = hidden_dims[-1]

        if n_downsample == -1:
            n_downsample = len(hidden_dims)
        else:
            assert n_downsample <= len(hidden_dims) and n_downsample >= 0

        strides = [2 for i in range(n_downsample)] + [
            1 for i in range(len(hidden_dims) - n_downsample)
        ]

        # Build Encoder
        for st, h_dim in zip(strides, hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=st,
                        padding=1,
                    ),
                    # nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                )
            )
            in_channels = h_dim

        modules.append(nn.Flatten(start_dim=1),)

        self.encoder = nn.Sequential(*modules)

        self.dsample = self.img_size // (2 ** n_downsample)
        self.dsample **= 2
        self.inter = nn.Linear(hidden_dims[-1] * self.dsample, inter_dim)

        self.fc_mu = nn.Linear(inter_dim, latent_dim)
        self.fc_var = nn.Linear(inter_dim, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.dsample)

        hidden_dims.reverse()

        add_layers = True
        if n_downsample == len(hidden_dims):
            n_downsample -= 1
            add_layers = False

        for i in range(n_downsample):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    # nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU(),
                )
            )

        if add_layers:
            for i in range(n_downsample, len(hidden_dims) - 1):
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(
                            hidden_dims[i],
                            hidden_dims[i + 1],
                            kernel_size=3,
                            padding=1,
                        ),
                        # nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.ReLU(),
                    )
                )

        if add_layers:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        hidden_dims[-1],
                        out_channels=hidden_dims[-1],
                        kernel_size=3,
                        padding=1,
                    ),
                    # nn.BatchNorm2d(hidden_dims[-1]),
                    nn.ReLU(),
                )
            )

        else:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[-1],
                        out_channels=hidden_dims[-1],
                        stride=2,
                        kernel_size=3,
                        padding=1,
                        output_padding=1,
                    ),
                    # nn.BatchNorm2d(hidden_dims[-1]),
                    nn.ReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.out_layer = nn.Sequential(
            nn.Conv2d(
                hidden_dims[-1], out_channels=self.in_channels, kernel_size=3, padding=1
            ),
            nn.Sigmoid(),
        )

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (torch.Tensor) Input torch.Tensor to encoder [N x C x H x W]
        :return: (torch.Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = self.inter(result)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        shape = int(sqrt(self.dsample))
        result = result.view(-1, self.last_hidden, shape, shape)
        result = self.decoder(result)
        result = self.out_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (torch.Tensor) Mean of the latent Gaussian
        :param logvar: (torch.Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
        kld_weight = 1

        if self.likelihood_dist == "gauss":
            recons_loss = F.mse_loss(recons, input)
        elif self.likelihood_dist == "bern":
            assert recons.isfinite().all()
            assert (recons <= 1).all()
            assert (recons >= 0).all()
            assert input.isfinite().all()
            assert (input <= 1).all()
            assert (input >= 0).all()

            recons_loss = F.binary_cross_entropy(recons, input)
            recons_loss *= (self.img_size ** 2) / 2
        else:
            raise ValueError("Undefined likelihood distribution.")

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
        )

        if self.loss_type == "H":  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == "B":  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(
                self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0]
            )
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError("Undefined loss type.")

        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss,
            "KLD": kld_loss,
            "KLD_Scaled": self.beta * kld_weight * kld_loss,
        }

    def sample(self, num_samples: int, current_device: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (torch.Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.Tensor) [B x C x H x W]
        :return: (torch.Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
