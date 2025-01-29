import numpy as np
from torch.utils.data import Dataset
import nibabel as nib

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