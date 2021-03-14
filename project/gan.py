import torch
import torch.nn as nn
from typing import Callable

from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.

        self.conv = self.make_discriminator(in_size[0])
        self.eval()

    @staticmethod
    def make_discriminator(in_size):
        modules = []
        for channel_in, channel_out in [(in_size, 64), (64, 128), (128, 256), (256, 512)]:
            modules.append(nn.Conv2d(in_channels=channel_in, out_channels=channel_out,
                                     kernel_size=4, padding=1, stride=2, bias=False))
            if channel_in != in_size:
                modules.append(nn.BatchNorm2d(num_features=channel_out))
            modules.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        modules.append(nn.Conv2d(in_channels=512, out_channels=1,
                                 kernel_size=4, padding=0, stride=1, bias=False))
        modules.append(nn.Sigmoid())

        return nn.Sequential(*modules)

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        y = self.conv(x)
        y = torch.squeeze(y, 3)
        y = torch.squeeze(y, 2)
        return y

class SNDiscriminator(Discriminator):

    @staticmethod
    def make_discriminator(in_size):
        modules = []
        for channel_in, channel_out in [(in_size, 64), (64, 128), (128, 256), (256, 512)]:
            modules.append(spectral_norm(nn.Conv2d(in_channels=channel_in, out_channels=channel_out,
                                     kernel_size=4, padding=1, stride=2, bias=False)))
            if channel_in != in_size:
                modules.append(nn.BatchNorm2d(num_features=channel_out))
            modules.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        modules.append(spectral_norm(nn.Conv2d(in_channels=512, out_channels=1,
                                 kernel_size=4, padding=0, stride=1, bias=False)))
        modules.append(nn.Sigmoid())

        return nn.Sequential(*modules)

class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim
        h_size = 64

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(z_dim, h_size * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(h_size * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(h_size * 8, h_size * 4, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(h_size * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(h_size * 4, h_size * 2, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(h_size * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(h_size * 2, h_size, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(h_size),
            nn.ReLU(),
            nn.ConvTranspose2d(h_size, out_channels, featuremap_size, 2, 1, bias=False),
            nn.Tanh()
        )

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        samples = torch.randn(n, self.z_dim, device=device)
        if with_grad:
            samples = self.forward(samples)
        else:
            with torch.no_grad():
                samples = self.forward(samples)
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        z = z.unsqueeze(2)
        z = z.unsqueeze(3)
        x = self.decoder(z)
        return x


