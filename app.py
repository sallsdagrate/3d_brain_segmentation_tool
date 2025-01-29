import streamlit as st
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json

from utils.config import (
    source_path,
    get_model,
    transform_test,
    cmap_black,       # black background: 0=black,1=green,2=blue,3=red
    transparent_cmap  # same idea, 0=transparent,1=green,2=blue,3=red
)

@st.cache_data
def run_inference(_model, image_path):
    data_nii = nib.load(image_path).get_fdata().transpose(3,0,1,2)  
    out = transform_test({"image": data_nii})
    vol = out["image"].float()  # shape (4,H,W,D)
    with torch.no_grad():
        logits = _model(vol.unsqueeze(0))  # (1,4,H,W,D)
        pred = torch.argmax(logits, dim=1) # (1,H,W,D)
    return vol, pred.squeeze(0).cpu().numpy()  # (4,H,W,D), (H,W,D)

def display_figure(volume, labels, ds_json, slice_idx, alpha):
    """
    2-row figure:
      Row 0, col=0 => black seg
      Row 0, col=[1..N] => grayscale channels
      Row 1, col=0 => blank
      Row 1, col=[1..N] => overlay
    """
    mods = ds_json["modality"]  # e.g., {"0":"FLAIR","1":"T1","2":"T1ce","3":"T2"}
    n_mod = len(mods)
    seg_slice = labels[..., slice_idx]

    fig, axs = plt.subplots(2, n_mod + 1, figsize=(4*(n_mod+1), 8))

    # Row0 col0 => black seg map
    axs[0,0].imshow(seg_slice, cmap=cmap_black)
    axs[0,0].set_title(f"Seg (Slice={slice_idx})")
    axs[0,0].axis("off")

    # Row0 col1..N => each channel grayscale
    for i in range(n_mod):
        base = volume[i,:,:,slice_idx].numpy()
        axs[0,i+1].imshow(base, cmap="gray")
        axs[0,i+1].set_title(mods[str(i)])
        axs[0,i+1].axis("off")

    # Row1 col0 => blank
    axs[1,0].axis("off")

    # Row1 col1..N => overlay
    for i in range(n_mod):
        base = volume[i,:,:,slice_idx].numpy()
        axs[1,i+1].imshow(base, cmap="gray")
        axs[1,i+1].imshow(seg_slice, cmap=transparent_cmap, alpha=alpha)
        axs[1,i+1].set_title(f"Overlay {mods[str(i)]}, α={alpha:.2f}")
        axs[1,i+1].axis("off")

    fig.tight_layout()
    return fig


model_path = "models/unet_3d_model_19.pth"
with open(f"{source_path}/dataset.json") as f:
    ds_json = json.load(f)

m = get_model(len(ds_json["modality"]), len(ds_json["labels"]))
m.load_state_dict(torch.load(model_path, map_location="cpu"))
m.eval()

test_files = [source_path + s for s in ds_json["test"]]

st.set_page_config(layout="wide")
st.title("3D MONAI UNet Inference on Brain Tumour Dataset")

sel_img = st.selectbox("Select an image:", test_files)
vol4d, lbl3d = run_inference(m, sel_img)  # (4,H,W,D), (H,W,D)

# Multi-class selection
label_map = ds_json["labels"]         # e.g. {"0":"background","1":"edema","2":"non-enhancing","3":"enhancing"}
int_to_cls = {int(k):v for k,v in label_map.items()}
all_classes = [int_to_cls[i] for i in range(1, len(int_to_cls))]  # skip background=0 in UI

st.subheader("Select classes to show:")
picked = st.multiselect("Classes:", all_classes, default=all_classes)
sel_idxs = []
for c in picked:
    for idx,name in int_to_cls.items():
        if name==c: sel_idxs.append(idx)

# Mask out everything not in sel_idxs
masked_label = np.zeros_like(lbl3d)
for idx in sel_idxs:
    masked_label[lbl3d==idx] = idx

max_slice = vol4d.shape[-1]-1
s_idx = st.slider("Slice index:", 0, max_slice, max_slice//2)
alpha = st.slider("Overlay alpha:", 0.0, 1.0, 0.5, 0.01)

fig = display_figure(vol4d, masked_label, ds_json, s_idx, alpha)
st.pyplot(fig)
