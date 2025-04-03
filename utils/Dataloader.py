import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import nibabel as ni
from nibabel.processing import resample_to_output
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")

SUSAN_PATH = '/Users/danai/cubic-share/projects/brain_tumor/Brain_Tumor_2020/Protocols/8_Susan'

class fMRIDataset(Dataset):
    """Dataset for fMRI Images
    This dataset class performs dataloading on already-preprocessed data, which has been #TODO: fill in exactly the preprocessing steps
    MRI_TYPE are ["t1","t1ce","t2", "flair"] #TODO:maybe add diffusion"""
    def __init__(self, root_dir, percentage = 1, master_file_dir = None, master_file_save_dir ='master_list.pkl', 
                 transform=None, fold_id=None, MRI_TYPE = 't1ce', dummy_data=False):
        """
        Arguments:
            root_dir (string): Path to the directory of cases, inside which are folders of imagings
            percentage (float, optional): Percentage of the dataset to use for experiment
            transform (callable, optional): Optional transform to be applied
            fold_id (int, optional): index in kfold
            #TODO: add labels path & handling
        """
        self.dummy_data = dummy_data
        self.root_dir = root_dir
        self.percentage = percentage
        self.transform = transform
        self.fold_id = fold_id
        self.modality = MRI_TYPE
        if master_file_dir is not None: 
            with open (master_file_dir, "rb") as master_file:
                self.master_list = pickle.load(master_file)
        else: 
            self.master_list = self.create_master_file_list()
            with open (master_file_save_dir,"wb") as pkl_file:
                pickle.dump(self.master_list, pkl_file)

        if self.percentage != 1:
            self.master_list = self.master_list[:int(self.percentage * len(self.master_list))]

    def create_master_file_list(self):
        """
        Creates dataframe containing all cases and their paths
        """
        CASES = [f for f in os.listdir(self.root_dir) if not f.startswith('.')]
        master_list = []
        for case in CASES:
            case_path = os.path.join(self.root_dir,case)
            MODALITIES =[f for f in os.listdir(case_path) if not f.startswith('.')]
            #for scan in SCANS:
            #    scan_path = os.path.join(case_path, scan)
            #    MODALITIES = os.listdir(scan_path)
            for modality in MODALITIES: 
                if modality.endswith('.gz') and self.modality in modality:
                    #print(os.path.join(scan_path,modality))
                    master_list.append(os.path.join(case_path,modality))
            
        return master_list

        

    def __len__(self):
        return len(self.master_list)
    

    def __getitem__(self, idx):
        from torch.nn import functional as F

        if isinstance(idx, slice):
            return [self[ii] for ii in range(*idx.indices(len(self)))]

        if self.dummy_data :
            idx_mri =  torch.rand(1, 1, 155, 240, 240)
        else: 
            idx_path = self.master_list[idx]
            idx_mri = ni.load(idx_path)
            if self.modality != 'patch':
                new_voxel_size = (
                    idx_mri.header.get_zooms()[0] * idx_mri.shape[0] / 128,
                    idx_mri.header.get_zooms()[1] * idx_mri.shape[1] / 128,
                    idx_mri.header.get_zooms()[2] * idx_mri.shape[2] / 128
                )
                resampled = resample_to_output(idx_mri, new_voxel_size).get_fdata()[:128,:128,:128]
            else:
                resampled = idx_mri.get_fdata()
            normalized = (resampled - resampled.min()) / (resampled.max() - resampled.min())
            idx_mri = torch.tensor(normalized).unsqueeze(0).float()

        return idx_mri


if __name__ == "__main__":
    mridataset = fMRIDataset('/Users/danai/cubic-share/projects/brain_tumor/Brain_Tumor_2020/Protocols/8_Susan', percentage= 1,  master_file_dir = '/home/brilli/vae_exp_code/master_list.pkl', dummy_data = False)
    print(len(mridataset))
    print(mridataset[0].shape)
    plt.imshow(mridataset[0][0][:,:,64], cmap='gray')
    plt.savefig('sample_image5.png')
    
    train_dataloader = DataLoader(mridataset, batch_size = 8, shuffle = True)
    import time
    start = time.time()
    train_features = next(iter(train_dataloader))
    end = time.time()
    print(f"Feature batch shape: {train_features.size()} and it took {end-start} seconds to load")
