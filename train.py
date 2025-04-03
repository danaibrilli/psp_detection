from vae3d import VAE3D
from utils.Dataloader import fMRIDataset, DataLoader
from pytorch_lightning import Trainer
import wandb
import argparse
import yaml
from vae_lightning import VAE
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger 
from pytorch_lightning.callbacks import ModelCheckpoint
import shutil


parser = argparse.ArgumentParser(description='Train a VAE model on MRI data')
parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
args = parser.parse_args()
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

wandb.init(project="PsP_TP_VAE", 
           config=config
        )   

wandb_logger = WandbLogger(project="PsP_TP_VAE")

checkpoint_callback = ModelCheckpoint(
    dirpath=config['checkpoint_dir'], 
    filename="model-{epoch:03d}",  
    save_top_k=-1,  
    every_n_epochs=config['save_every_n_epochs'], 
    save_last=config['save_last']
)

#TODO: add automatic patch vs full MRI image selection
mridataset = fMRIDataset(config["data_dir"], percentage=config["dataset_percentage"],  master_file_dir = None, 
                         master_file_save_dir=config["master_file_save_dir"], dummy_data = False, MRI_TYPE=config["MRI_Type"])
train_set = mridataset

print('train set length', len(train_set))
print('train_sample shape', train_set[0].shape)

train_loader = DataLoader(train_set, batch_size = config['batch_size'], shuffle = True)

trainer = Trainer(accelerator = 'gpu', devices = [config["device"]], callbacks=[checkpoint_callback], max_epochs = config["epochs"], logger = wandb_logger)
model = VAE(VAE3D=VAE3D( beta=config["beta"]))
trainer.fit(model = model, train_dataloaders = train_loader)


shutil.copy(args.config, config['checkpoint_dir'])
wandb.save(args.config)
wandb.finish()

loss, recon_loss, kld_loss = model.train_losses, model.train_recon_losses, model.train_KLD_losses
names = ['loss', 'recon_loss', 'kld_loss']
lists = [loss, recon_loss, kld_loss]


fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i, ax in enumerate(axes):
    ax.plot(lists[i], marker='o')
    ax.set_title(names[i])
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')

plt.tight_layout()
plt.savefig('subplots_figure.png')
plt.show()

img1 = mridataset[0]
rec = model.model.generate(img1.unsqueeze(0))

selected_slices = [35, 50, 64, 80, 95]
for sli in selected_slices:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(img1[0][:, :, sli], cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(rec[0][:, :, sli].detach().numpy().squeeze(), cmap='gray')
    axes[1].set_title('Reconstructed')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'slice_{sli}.png')