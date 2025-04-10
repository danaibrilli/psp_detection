from abc import abstractmethod

import torch
from torch import nn
from torch.nn import functional as F
from typing import TypeVar
from torch import tensor as Tensor
import torch
from typing import List
from vae_base import VAEBackbone

from torch.utils.data import DataLoader
from utils.Dataloader import fMRIDataset
from torchmetrics import StructuralSimilarityIndexMeasure


class VAE3D(VAEBackbone):

    def __init__(self,
                 in_channels: int = 1,
                 latent_dim: int = 1024,
                 hidden_dims: List = None,
                 beta=1,
                 input_shape = (155,240,240),
                 example_input_shape=None,
                 **kwargs):  # -> None
        super(VAE3D, self).__init__()

        self.latent_dim = latent_dim
        self.beta = float(beta)  # added
        modules = []
        if hidden_dims is None:
            hidden_dims = [4, 16, 32, 64, 128, 256]
        # don't modify hidden_dims
        self.hidden_dims = hidden_dims.copy()
        hidden_dims_variable = hidden_dims.copy()

        # Build Encoder
        # formula: o = floor( (i + p - k) / s) + 1
        for h_dim in hidden_dims_variable:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels=in_channels,
                              out_channels=h_dim,  # need 3d conv layers
                              kernel_size=3,
                              stride=2,  # will reduce the lengths of each dim to half: size = 1/8 size
                              padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
            )
            
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        #self.linear_postEnc = nn.Linear(40960, hidden_dims_variable[-1]*8)
        # input dim should be the dim after flatten
        self.fc_mu = nn.Linear(hidden_dims_variable[-1]*8, latent_dim) #was *8
        self.fc_var = nn.Linear(hidden_dims_variable[-1]*8, latent_dim)  # 8 = 2 ** 3

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(
            latent_dim, hidden_dims[-1]*8) #was *8 # should be times 2**3

        hidden_dims_variable.reverse()
        for i in range(len(hidden_dims_variable) -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(hidden_dims_variable[i],  # 3d
                                    hidden_dims_variable[i + 1],
                                    kernel_size=3,
                                    stride=2,
                                    padding=0, # was 0 -> 1 for patch
                                    output_padding=1),
                    nn.BatchNorm3d(hidden_dims_variable[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential( 
            nn.Conv3d(hidden_dims_variable[-1], out_channels=1,  # output_channels should be the same as input channel= -> 1
                      kernel_size=3, padding=2), #changed padding from 2 to 1 for patch
            nn.Sigmoid()
        )

        if example_input_shape:
            self.example_input_array = torch.rand(1, *example_input_shape)
        
        pass

    def encode(self, input: Tensor):  # -> List[torch.Tensor@encode]
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x L x W x H]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
    
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor):  # -> Tensor
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C=1 x L x W x H]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], 2, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor):  # -> Tensor
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs):  # -> List[Tensor]
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      recons,
                      inputs,
                      mu,
                      log_var,
                      M_N,
                      *args,
                      **kwargs):  # HACK: bad practice
        r"""
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        # Account for the minibatch samples from the dataset
        #kld_weight = 1#M_N * float(self.beta)
        mse = F.mse_loss(recons, inputs)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(inputs.device)
        ssim_loss = 1 - ssim(recons, inputs)
        recons_loss = 0.8*mse + 0.2*ssim_loss
        #recons_loss = mse
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1) / self.latent_dim)


        loss = recons_loss + self.beta * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

    def sample(self,
               num_samples: int,
               current_device: int, 
               **kwargs):  # -> Tensor
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim, device=current_device)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs):  # -> Tensor
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
    


if __name__ == "__main__":

    features = torch.rand ((8, 1 ,64, 64, 64))
    vae = VAE3D()
    print('model defined')

    encode = vae.encode(features)
    print('encode')
    reparameter = vae.reparameterize(encode[0], encode[1])
    print('reparameter')
    decode = vae.decode(reparameter)
    print('decode')
    print(vae.forward(features))
    print('forward')