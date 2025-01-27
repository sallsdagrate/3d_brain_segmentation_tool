import tarfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from matplotlib import colors
import nibabel as nib
from monai.networks.nets import UNet
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.transforms import (
    Compose,
    ToTensord,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    DivisiblePadd
)

source_path = 'Task01_BrainTumour'

with open(f'{source_path}/dataset.json') as f:
    dataset_json = json.load(f)

class BrainTumorDataset(Dataset):
    def __init__(self, img_paths, lbl_paths, transforms=None):
        self.img_paths = img_paths
        self.lbl_paths = lbl_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load multi-modal image
        image_nii = nib.load(self.img_paths[idx])
        image = image_nii.get_fdata().transpose(3, 0, 1, 2)  # shape: (4, H, W, D)
        
        label_nii = nib.load(self.lbl_paths[idx])
        label = label_nii.get_fdata()[np.newaxis, ...]  # shape: (1, H, W, D)

        if self.transforms:
            transformed = self.transforms({
                'image': image,
                'label': label
                })

        image_tensor = transformed['image'].float()
        label_tensor = transformed['label'][0].long()  # for segmentation

        return image_tensor, label_tensor

# 80/20 train/validation split
n = dataset_json['numTraining']
n_train = int(n * 0.8)
n_val = n - n_train

all_list = dataset_json['training']
all_ims = [source_path + sample['image'] for sample in all_list]
all_lbs = [source_path + sample['label'] for sample in all_list]

train_ims, val_ims, train_lbs, val_lbs = train_test_split(all_ims, all_lbs, test_size=0.2, random_state=21)

transform_train = Compose([
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    RandScaleIntensityd(keys="image", factors=0.1, prob=0.7),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.7),
    DivisiblePadd(k=16, keys=["image", "label"]),
    ToTensord(keys=["image", "label"])
])

transform_val = Compose([
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    DivisiblePadd(k=16, keys=["image", "label"]),
    ToTensord(keys=["image", "label"])
])

IN_MODALITIES=len(dataset_json['modality'])
NUM_CLASSES=len(dataset_json['labels'])

def dice_loss(pred_softmax, target, epsilon=1e-6):
    # pred_softmax: (N, C, D, H, W)
    # target: (N, D, H, W), integer labels
    num_classes = pred_softmax.shape[1]
    target_onehot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()

    intersection = torch.sum(pred_softmax * target_onehot, dim=(2,3,4))
    denom = torch.sum(pred_softmax, dim=(2,3,4)) + torch.sum(target_onehot, dim=(2,3,4))
    dice = (2.0 * intersection + epsilon) / (denom + epsilon)
    return 1 - dice.mean()  # average across classes

def main():
    train_dataset = BrainTumorDataset(train_ims, train_lbs, transform_train)
    val_dataset = BrainTumorDataset(val_ims, val_lbs, transform_val)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=8)

    model = UNet(
        spatial_dims=3, 
        in_channels=IN_MODALITIES,   # 4 MRI modalities
        out_channels=NUM_CLASSES,  # 4 output classes
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        dropout=0.2,
    )
    
    num_epochs = 20
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    torch.cuda.empty_cache()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
    
        progress_bar = tqdm(train_loader)
        for batch_data in progress_bar:
            images, labels = batch_data
            images, labels = images.to(device), labels.to(device)  # move to GPU
    
            optimizer.zero_grad()
            logits = model(images)  # (N, out_channels, D, H, W)
            pred_softmax = torch.softmax(logits, dim=1)
            
            loss = dice_loss(pred_softmax, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix(loss=loss.item())
    
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, LR: {scheduler.get_lr()}")
    
        # Validation step...
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in tqdm(val_loader):
                images, labels = batch_data
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                pred_softmax = torch.softmax(logits, dim=1)
                loss = dice_loss(pred_softmax, labels)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")

        torch.save(model.state_dict(), f"models/unet_3d_model_{epoch}.pth")
        scheduler.step()

if __name__ == "__main__":
    main()