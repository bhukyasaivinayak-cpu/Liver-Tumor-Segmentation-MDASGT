
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch

from model import MDASGT
from preprocess import adaptive_pfwf


# Application configuration
st.set_page_config(
    page_title="Liver MRI Tumor Segmentation",
    page_icon="🧬",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = APP_DIR / "checkpoints" / "best.pt"


# Automatically load the trained model
@st.cache_resource
def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint not found: {CHECKPOINT_PATH}"
        )

    # CPU works locally and on Streamlit Community Cloud
    device = torch.device("cpu")

    ckpt = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model = MDASGT(base=int(ckpt["base"]))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    size = int(ckpt["size"])
    return model, size


st.title("Liver MRI Tumor Segmentation")
st.caption(
    "PFWF → CRes2Net → MDA-SGT | Deep Learning Research Prototype"
)

st.warning(
    "Research demonstration trained on synthetic images. "
    "Not validated for clinical diagnosis or treatment."
)

try:
    model, size = load_model()
    st.success("Trained segmentation model loaded successfully!")
except Exception as e:
    st.error(f"Unable to load trained model: {e}")
    st.stop()


uploaded = st.file_uploader(
    "Upload a grayscale MRI image",
    type=["png", "jpg", "jpeg"]
)

threshold = st.slider(
    "Tumor segmentation threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05
)


if uploaded is not None:
    try:
        raw = np.asarray(
            Image.open(uploaded).convert("L")
        )

        # Adaptive PFWF preprocessing
        filtered = adaptive_pfwf(raw)

        # Resize to model input dimensions
        resized = cv2.resize(
            filtered.astype(np.float32),
            (size, size),
            interpolation=cv2.INTER_AREA
        )

        input_tensor = torch.from_numpy(
            resized[None, None]
        ).float()

        # Model inference
        with torch.inference_mode():
            logits = model(input_tensor)
            prob = torch.sigmoid(logits)[0, 0].numpy()

        # Restore original image resolution
        prob = cv2.resize(
            prob,
            (raw.shape[1], raw.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

        mask = prob >= threshold

        # Create tumor overlay
        overlay = np.stack([raw] * 3, axis=-1)

        tumor_color = np.array([255, 40, 80])

        overlay[mask] = (
            0.35 * overlay[mask]
            + 0.65 * tumor_color
        ).astype(np.uint8)

        # Display results
        st.subheader("Segmentation Results")

        col1, col2, col3 = st.columns(3)

        col1.image(
            raw,
            caption="Original MRI",
            use_container_width=True
        )

        col2.image(
            filtered,
            caption="PFWF Enhanced",
            clamp=True,
            use_container_width=True
        )

        col3.image(
            overlay,
            caption="Predicted Tumor Overlay",
            use_container_width=True
        )

        # Download binary segmentation mask
        encoded = BytesIO()

        Image.fromarray(
            mask.astype(np.uint8) * 255
        ).save(encoded, format="PNG")

        st.download_button(
            label="Download Tumor Segmentation Mask",
            data=encoded.getvalue(),
            file_name="tumor_mask.png",
            mime="image/png"
        )

    except Exception as e:
        st.error(f"Image processing failed: {e}")

