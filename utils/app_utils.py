import platform
from typing import List, Dict, Tuple, Callable

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import nibabel as nib
import numpy as np
import torch
from matplotlib import colors
from utils.config import transform_test, get_segresnet, get_unet3d

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

MODEL_PATHS = {
    "segresnet": {
        "ensemble": [
            "models/segresnet/segresnet_model_6.pth",
            "models/segresnet/segresnet_model_7.pth",
            "models/segresnet/segresnet_model_8.pth"
        ],
        "mc_dropout": "models/segresnet/segresnet_model_8.pth",
        "func": get_segresnet
    },
    "unet3d": {
        "ensemble": [
            "models/unet3d/unet_3d_model_17.pth",
            "models/unet3d/unet_3d_model_18.pth",
            "models/unet3d/unet_3d_model_19.pth"
        ],
        "mc_dropout": "models/unet3d/unet_3d_model_19.pth",
        "func": get_unet3d
    }
}
# get_model_dict

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

def get_hardware_info() -> Dict[str, str]:
    """Get system and PyTorch hardware information"""
    return {
        "Python Platform": platform.platform(),
        "Processor": platform.processor(),
        "PyTorch Version": torch.__version__,
        "CUDA Available": str(torch.cuda.is_available()),
        **({"GPU Name": torch.cuda.get_device_name(0)} if torch.cuda.is_available() else {})
    }


def load_model(
        checkpoint_path: str,
        in_channels: int,
        num_classes: int,
        load_func: Callable[[int, int], tuple[torch.nn.Module, dict]],
        eval_mode: bool = True
) -> torch.nn.Module:
    """Load 3D U-Net model from checkpoint"""
    model = load_func(in_channels, num_classes)[0]
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    if eval_mode:
        model.eval()

    return model.to(DEVICE)


def run_inference(
        volume: torch.Tensor,
        models: List[torch.nn.Module],
        method: str = "ensemble",
        num_samples: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference with uncertainty estimation
    Returns: (pred_label, mean_probs, std_probs)
    """
    with torch.no_grad():
        vol_4d = volume.unsqueeze(0).to(DEVICE)
        preds = []

        if method == "ensemble":
            for model in models:
                logits = model(vol_4d)
                preds.append(torch.softmax(logits, dim=1).cpu().numpy())
        else:  # MC Dropout
            models[0].train()  # Only first model used for MC Dropout
            for _ in range(num_samples):
                logits = models[0](vol_4d)
                preds.append(torch.softmax(logits, dim=1).cpu().numpy())
            models[0].eval()

    all_preds = np.stack(preds, axis=0)
    mean_probs = all_preds.mean(axis=0)
    std_probs = all_preds.std(axis=0)
    return mean_probs.argmax(axis=1).squeeze(0), mean_probs.squeeze(0), std_probs.squeeze(0)


def create_static_color_key(class_dict: Dict[int, str]) -> plt.Figure:
    """Compact color key matching website theme"""
    int_class_dict = {int(k): v for k, v in class_dict.items()}
    sorted_classes = sorted(int_class_dict.items())

    fig, ax = plt.subplots(figsize=(6, 1 + 0.3 * len(sorted_classes)), facecolor='none')

    handles = [
        patches.Patch(
            color=cmap_black(i / (len(sorted_classes) - 1)),
            label=f"{idx}: {name}"
        )
        for i, (idx, name) in enumerate(sorted_classes)
        if idx != 0
    ]

    leg = ax.legend(
        handles=handles,
        title="Tissue Classes",
        loc="center",
        frameon=False,
        title_fontsize=25,
        fontsize=20,
    )
    plt.setp(leg.get_texts(), color='w')
    plt.setp(leg.get_title(), color='w')
    ax.axis("off")
    plt.tight_layout()
    return fig


# Remove class masking from visualization functions
def generate_slice_visualization(
        volume: torch.Tensor,
        segmentation: np.ndarray,  # Now uses full prediction
        dataset_meta: Dict,
        slice_idx: int,
        overlay_alpha: float = 0.5
) -> plt.Figure:
    """Generate comparative visualization of segmentation results"""
    modalities = dataset_meta["modality"]
    n_mods = len(modalities)

    fig, axs = plt.subplots(2, n_mods + 1, figsize=(4 * (n_mods + 1), 8))

    # Segmentation plot
    axs[0, 0].imshow(segmentation[..., slice_idx], cmap=cmap_black)
    axs[0, 0].set_title(f"Segmentation (slice={slice_idx})")
    axs[0, 0].axis("off")

    # Modality views
    for i in range(n_mods):
        base = volume[i, :, :, slice_idx].cpu().numpy()
        axs[0, i + 1].imshow(base, cmap="gray")
        axs[0, i + 1].set_title(modalities[str(i)])
        axs[0, i + 1].axis("off")

    # Overlay views
    for i in range(n_mods):
        base = volume[i, :, :, slice_idx].cpu().numpy()
        axs[1, i + 1].imshow(base, cmap="gray")
        axs[1, i + 1].imshow(segmentation[..., slice_idx], cmap=transparent_cmap, alpha=overlay_alpha)
        axs[1, i + 1].set_title(f"{modalities[str(i)]} Overlay")
        axs[1, i + 1].axis("off")

    fig.tight_layout()
    return fig


def load_medical_volume(image_path: str) -> torch.Tensor:
    """Load and preprocess medical image volume"""
    data = nib.load(image_path).get_fdata().transpose(3, 0, 1, 2)
    return transform_test({"image": data})["image"].float().to(DEVICE)


def compute_entropy_map(mean_probs: np.ndarray) -> np.ndarray:
    """Calculate entropy from probability distribution"""
    eps = 1e-10  # Prevent log(0)
    return -np.sum(mean_probs * np.log(mean_probs + eps), axis=0)


def gather_predicted_std(std_probs: np.ndarray, pred_label: np.ndarray) -> np.ndarray:
    """Extract standard deviation for predicted classes"""
    return np.take_along_axis(std_probs, np.expand_dims(pred_label, axis=0), axis=0).squeeze(0)


def generate_uncertainty_visualization(
        volume: torch.Tensor,
        uncertainty_map: np.ndarray,
        dataset_meta: Dict,
        slice_idx: int,
        title: str = "Uncertainty",
        cmap: str = "viridis"
) -> plt.Figure:
    """Generate visualization for uncertainty/entropy maps"""
    modalities = dataset_meta["modality"]
    n_mods = len(modalities)

    fig, axs = plt.subplots(1, n_mods + 1, figsize=(4 * (n_mods + 1), 5))

    # Main uncertainty map
    im = axs[0].imshow(uncertainty_map[..., slice_idx], cmap=cmap)
    axs[0].set_title(f"{title} (slice={slice_idx})")
    axs[0].axis("off")
    fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

    # Modality overlays
    for i in range(n_mods):
        base = volume[i, :, :, slice_idx].cpu().numpy()
        axs[i + 1].imshow(base, cmap="gray")
        axs[i + 1].imshow(uncertainty_map[..., slice_idx], cmap=cmap, alpha=0.6)
        axs[i + 1].set_title(f"{modalities[str(i)]} + {title}")
        axs[i + 1].axis("off")

    fig.tight_layout()
    return fig


# Helper functions for visualization components
def _plot_segmentation(ax: plt.Axes, seg_slice: np.ndarray, slice_idx: int):
    ax.imshow(seg_slice, cmap=cmap_black)
    ax.set_title(f"Segmentation (slice={slice_idx})")
    ax.axis("off")


def _plot_modality(ax: plt.Axes, base: np.ndarray, modality: str):
    ax.imshow(base, cmap="gray")
    ax.set_title(modality)
    ax.axis("off")


def _plot_overlay(ax: plt.Axes, base: np.ndarray, seg_slice: np.ndarray, alpha: float, modality: str):
    ax.imshow(base, cmap="gray")
    ax.imshow(seg_slice, cmap=transparent_cmap, alpha=alpha)
    ax.set_title(f"{modality} Overlay (α={alpha:.2f})")
    ax.axis("off")
