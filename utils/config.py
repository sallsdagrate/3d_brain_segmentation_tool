import torch
import torch.nn.functional as F
from matplotlib import colors
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    ToTensord,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    NormalizeIntensityd,
    DivisiblePadd
)

def debug_print_transform(data):
    image = data["image"]
    # label = data["label"]
    print(f"[DEBUG] image shape: {image.shape}") #, label shape: {label.shape}")
    return data

'''
COLOUR MAPS
'''
cs = ['green', 'blue', 'red']
label_to_color = {i+1: c for i, c in enumerate(cs)}
cmap_black = colors.ListedColormap(['black', cs[0], cs[1], cs[2]])
cmap_empty = colors.ListedColormap(['none', cs[0], cs[1], cs[2]])
transparent_cmap = colors.ListedColormap([
    (0, 0, 0, 0),   # 0 => transparent
    (0, 1, 0, 1),   # 1 => green
    (0, 0, 1, 1),   # 2 => blue
    (1, 0, 0, 1)    # 3 => red
])

'''
TRANSFORMS
'''
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
3D-UNET MODEL
'''
def get_model(in_channels, num_classes):
    return UNet(
            spatial_dims=3, 
            in_channels=in_channels,
            out_channels=num_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            dropout=0.2,
        )

''' 
CONFIG VARIABLES
''' 
source_path = 'Task01_BrainTumour'
num_workers = 8
testVsTrainSplit = 0.2

batch_size = 2
num_epochs = 20
lr = 1e-3