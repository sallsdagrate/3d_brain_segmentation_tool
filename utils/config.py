import torch
import torch.nn.functional as F
from monai.networks.nets import UNet, SegResNet
from pathlib import Path

from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    Rand3DElasticd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    NormalizeIntensityd,
    DivisiblePadd,
    ToTensord
)

''' 
CONFIG VARIABLES
'''

source_path = 'Task01_BrainTumour'
num_workers = 8
testVsTrainSplit = 0.2
batch_size = 2

DATA_ROOT = Path(source_path)
DATASET_LINK_AWS = 'https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar'

def debug_print_transform(data):
    image = data["image"]
    # label = data["label"]
    print(f"[DEBUG] image shape: {image.shape}") #, label shape: {label.shape}")
    return data

'''
TRANSFORMS
'''
transform_train = Compose([
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    Rand3DElasticd(keys=["image", "label"], sigma_range=(4, 6), magnitude_range=(40, 100), prob=0.3),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    RandScaleIntensityd(keys="image", factors=0.1, prob=0.7),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.7),
    RandGaussianNoised(keys="image", std=0.01, prob=0.2),
    DivisiblePadd(k=16, keys=["image", "label"]),
    ToTensord(keys=["image", "label"])
])

transform_val = Compose([
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    DivisiblePadd(k=16, keys=["image", "label"]),
    ToTensord(keys=["image", "label"])
])

transform_test = Compose([
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    DivisiblePadd(k=16, keys="image"),
    ToTensord(keys=["image"])
])

''' 
DICE LOSS FUNCTION
'''
def dice_loss(pred_softmax, target, epsilon=1e-6):
    # pred_softmax: (N, C, D, H, W)
    # target: (N, D, H, W), integer labels
    num_classes = pred_softmax.shape[1]
    target_onehot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()

    intersection = torch.sum(pred_softmax * target_onehot, dim=(2,3,4))
    denom = torch.sum(pred_softmax, dim=(2,3,4)) + torch.sum(target_onehot, dim=(2,3,4))
    dice = (2.0 * intersection + epsilon) / (denom + epsilon)
    return 1 - dice.mean()  # average across classes


'''
MODELS
'''
def get_unet3d(in_channels, num_classes) -> tuple[torch.nn.Module, dict]:
    return UNet(
            spatial_dims=3, 
            in_channels=in_channels,
            out_channels=num_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            dropout=0.2,
        ), {
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'num_epochs': 20
    }

def get_segresnet(in_channels, out_channels) -> tuple[torch.nn.Module, dict]:
    """
    Construct a SegResNet model for 3D medical image segmentation.
    Adjust the parameters as needed.
    """
    return SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=8,
        blocks_down=[1, 2, 2],
        blocks_up=[1, 1],
        dropout_prob=0.2
    ),{
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'num_epochs': 20
    }
