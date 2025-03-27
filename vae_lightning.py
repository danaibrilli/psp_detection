import pytorch_lightning as pl
import torch.nn.functional as F
from vae3d import VAE3D
import torch.nn as nn
import torch

class VAE(pl.LightningModule):
    def __init__(self, VAE3D):
        super().__init__()
        self.model =  VAE3D
        self.train_losses = []
        self.train_recon_losses = []
        self.train_KLD_losses = []
        self.results = []

    def training_step(self, batch, batch_idx):
        y, inp, mu, log_var = self.model.forward(batch)
        #M_N = batch_size/train_samples for minibatch something
        M_N = 1
        loss = self.model.loss_function(y, inp, mu, log_var, M_N)
        self.train_losses.append(loss['loss'].detach().cpu().numpy())
        self.train_recon_losses.append(loss['Reconstruction_Loss'].detach().cpu().numpy())
        self.train_KLD_losses.append(loss['KLD'].detach().cpu().numpy())

        self.log("train_loss",loss['loss'].item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("recon_loss", loss['Reconstruction_Loss'].item(), on_step=True, on_epoch=True, prog_bar = True, logger=True)
        self.log("KLD_loss", loss['KLD'].item(),on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss['loss']

    def validation_step(self, batch, batch_idx):
        y, inp, mu, log_var = self.model.forward(batch)
        M_N = 1
        loss = self.model.loss_function(y, inp, mu, log_var, M_N)
        self.log('val_loss',loss['loss'],on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer