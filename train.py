import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.BrainTumourDataset import BrainTumorDataset
from utils.config import (
    transform_train,
    transform_val,
    dice_loss,
    source_path,
    batch_size,
    num_workers,
    testVsTrainSplit,
    get_unet3d,
    get_segresnet
    )


with open(f'{source_path}/dataset.json') as f:
    dataset_json = json.load(f)

# 80/20 train/validation split
all_list = dataset_json['training']
all_ims = [source_path + sample['image'] for sample in all_list]
all_lbs = [source_path + sample['label'] for sample in all_list]
train_ims, val_ims, train_lbs, val_lbs = train_test_split(all_ims, all_lbs, test_size=testVsTrainSplit, random_state=21)

IN_MODALITIES=len(dataset_json['modality']) # 4 modalities
NUM_CLASSES=len(dataset_json['labels']) # 4 output classes

def main():
    parser = argparse.ArgumentParser(
        description="Train a 3D segmentation model (SegResNet or UNet3D)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["segresnet", "unet3d"],
        help="Choose which model to train: segresnet or unet3d."
    )
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    if device == 'cpu':
        print("WARNING: Training on CPU. This will be slow.")
    else:
        print("Using CUDA.")
        torch.cuda.empty_cache()

    # Create the model
    if args.model == "segresnet":
        model, training_params = get_unet3d(IN_MODALITIES, NUM_CLASSES)
        print("Using SegResNet")
    else:  # 'unet3d'
        model, training_params = get_segresnet(IN_MODALITIES, NUM_CLASSES)
        print("Using UNet3D")
    model.to(device)

    train_dataset = BrainTumorDataset(train_ims, train_lbs, transform_train)
    val_dataset = BrainTumorDataset(val_ims, val_lbs, transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_epochs = training_params['num_epochs']
    optimizer = optim.Adam(model.parameters(), lr=training_params['lr'], weight_decay=training_params['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    for epoch in range(num_epochs):
        
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader)
        for it, batch_data in enumerate(progress_bar):
            images, labels = batch_data
            images, labels = images.to(device), labels.to(device)  # move to GPU
    
            optimizer.zero_grad()
            logits = model(images)  # (N, out_channels, D, H, W)
            pred_softmax = torch.softmax(logits, dim=1)
            
            loss = dice_loss(pred_softmax, labels)
            loss.backward()

            optimizer.step()
    
            running_loss += loss.item()
            progress_bar.set_postfix(Ep=epoch+1, It=it, Loss=loss.item(), LR=scheduler.get_last_lr())
    
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()}")
    
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
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        torch.save(model.state_dict(), f"models/unet_3d_model_{epoch}.pth")
        scheduler.step()

if __name__ == "__main__":
    main()