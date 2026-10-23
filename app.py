"""Streamlit web app for detecting resistor values in images.

Run with: streamlit run app.py
"""

from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = (
    PROJECT_DIR
    / "runs"
    / "detect"
    / "models"
    / "resistor_detector_stage2"
    / "weights"
    / "best.pt"
)

st.set_page_config(
    page_title="Resistor AI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; }
    .block-container { max-width: 1180px; padding-top: 2.5rem; }
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 20px;
        background: linear-gradient(120deg, #102a43, #174b70);
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 { color: white; margin: 0; font-size: 2.5rem; }
    .hero p { color: #d7e8f5; margin: .55rem 0 0; font-size: 1.05rem; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e3e9f1;
        padding: 1rem; border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the resistor detection model...")
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


st.markdown(
    """
    <div class="hero">
      <h1>Resistor AI</h1>
      <p>Upload a photo to detect resistor values with your trained computer vision model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider(
        "Minimum confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Detections below this confidence are hidden.",
    )
    model_path_text = st.text_input(
        "Model checkpoint",
        value=str(DEFAULT_MODEL),
        help="Path to a trained Ultralytics .pt checkpoint.",
    )
    st.caption("Tip: use the best.pt file from a completed training run.")

uploaded_file = st.file_uploader(
    "Choose a resistor image",
    type=("jpg", "jpeg", "png", "webp"),
    help="For best results, use a clear, well-lit photo.",
)

if uploaded_file is None:
    st.info("Upload an image to get started.")
    st.stop()

try:
    model_path = Path(model_path_text).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_DIR / model_path
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. Train a model or select a valid best.pt file."
        )

    image = Image.open(uploaded_file).convert("RGB")
    model = load_model(str(model_path))
    predictions = model.predict(source=image, conf=confidence, verbose=False)
except (OSError, ValueError, RuntimeError) as error:
    st.error(f"Could not run detection: {error}")
    st.stop()

result = predictions[0]
detections = []
if result.boxes is not None:
    for box in result.boxes:
        class_id = int(box.cls.item())
        detections.append(
            {
                "Resistor": model.names[class_id],
                "Confidence": float(box.conf.item()),
            }
        )
detections.sort(key=lambda detection: detection["Confidence"], reverse=True)

annotated_image = Image.fromarray(result.plot()[..., ::-1])
image_column, result_column = st.columns((1.15, 0.85), gap="large")

with image_column:
    st.subheader("Detection preview")
    st.image(annotated_image, use_container_width=True)

with result_column:
    st.subheader("Results")
    st.metric("Resistors detected", len(detections))
    if detections:
        st.dataframe(
            [
                {
                    "Resistor": detection["Resistor"],
                    "Confidence": f"{detection['Confidence']:.1%}",
                }
                for detection in detections
            ],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No resistors detected. Try a clearer image or lower the confidence threshold.")

    st.caption(f"Model: {model_path.name} · Confidence threshold: {confidence:.0%}")
