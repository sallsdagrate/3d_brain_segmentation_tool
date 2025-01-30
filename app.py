import os.path
import urllib

import streamlit as st
import json
import tarfile

from app_utils import (
    get_hardware_info,
    load_model,
    run_inference,
    create_static_color_key,
    load_medical_volume,
    generate_slice_visualization,
    generate_uncertainty_visualization,
    gather_predicted_std,
    compute_entropy_map,
    DATA_ROOT, DATASET_LINK_AWS, MODEL_PATHS
)

import urllib.request
import tarfile
import os

def download_dataset():
    with st.spinner("Downloading and extracting dataset..."):
        try:
            dataset_path = f"{DATA_ROOT}.tar"
            if not os.path.exists(dataset_path):
                with st.spinner("Downloading dataset"):
                    urllib.request.urlretrieve(DATASET_LINK_AWS, dataset_path)

            with st.spinner("Extracting files..."):
                with tarfile.open(dataset_path) as datafile:
                    datafile.extractall()

            st.success("Dataset downloaded and extracted successfully!")
            print(os.listdir())
        except Exception as e:
            st.error(f"Error downloading dataset: {str(e)}")
            raise


@st.cache_resource
def initialize_app():
    """Load dataset metadata and models once"""
    with (DATA_ROOT / "dataset.json").open() as f:
        dataset_meta = json.load(f)

    test_files = [str(DATA_ROOT / p) for p in dataset_meta["test"]]

    in_channels = len(dataset_meta["modality"])
    out_channels = len(dataset_meta["labels"])
    models = {
        "ensemble": [load_model(p, in_channels, out_channels) for p in MODEL_PATHS["ensemble"]],
        "mc_dropout": [load_model(MODEL_PATHS["mc_dropout"][0], in_channels, out_channels, eval_mode=False)]
    }

    return dataset_meta, test_files, models


# Cache segmentation results
@st.cache_data(max_entries=3, show_spinner=False)
def run_inference_cached(_models, method_name, volume_path, num_samples=10):
    vol = load_medical_volume(volume_path)
    return run_inference(vol, _models[method_name], method=method_name, num_samples=num_samples)


def main():
    st.set_page_config(layout="wide")
    st.title("3D Brain Tumor Segmentation")

    # Load initial data and models
    if not os.path.exists(DATA_ROOT):
        download_dataset()
    dataset_meta, test_files, models = initialize_app()

    # Hardware information
    with st.expander("System Information"):
        for k, v in get_hardware_info().items():
            st.write(f"**{k}:** {v}")

    # File selection
    selected_file = st.selectbox("Select medical scan:", test_files)
    volume = load_medical_volume(selected_file)

    # Visualization controls
    st.subheader("Visualization Settings")
    col1, col2, col3 = st.columns([2, 4, 2])  # Adjust column widths

    with col1:
        method = st.radio("Uncertainty Method:", ["ensemble", "mc_dropout"])

    with col2:
        slice_idx = st.slider("Slice Index:", 0, volume.shape[-1] - 1, volume.shape[-1] // 2)
        overlay_alpha = st.slider("Segmentation Opacity:", 0.0, 1.0, 0.5, 0.01)

    with col3:
        st.pyplot(create_static_color_key(dataset_meta["labels"]), use_container_width=False)

    if method == "mc_dropout":
        mc_passes = st.slider("Monte Carlo Samples:", 1, 20, 10)

    if st.button("Perform Segmentation"):
        with st.spinner("Analyzing scan..."):
            # Get cached results
            pred_label, mean_probs, std_probs = run_inference_cached(
                models,
                method,
                selected_file,
                mc_passes if method == "mc_dropout" else 10
            )

            # Calculate uncertainty metrics
            pred_std = gather_predicted_std(std_probs, pred_label)
            entropy_map = compute_entropy_map(mean_probs)

            # Display results
            st.subheader("Segmentation Results")

            # Primary segmentation
            st.markdown("#### Tissue Segmentation")
            st.pyplot(generate_slice_visualization(
                volume,
                pred_label,
                dataset_meta,
                slice_idx,
                overlay_alpha
            ))

            st.markdown("#### Standard Deviation Map")
            st.pyplot(generate_uncertainty_visualization(
                volume,
                pred_std,
                dataset_meta,
                slice_idx,
                title="Uncertainty",
                cmap="magma"
            ))

            st.markdown("#### Entropy Analysis")
            st.pyplot(generate_uncertainty_visualization(
                volume,
                entropy_map,
                dataset_meta,
                slice_idx,
                title="Entropy",
                cmap="viridis"
            ))


if __name__ == "__main__":
    main()
