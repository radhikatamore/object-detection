import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

st.set_page_config(page_title="Object Detection", layout="wide")

st.title("YOLO Object Detection")
st.write("Upload an image and run object detection using a pretrained YOLO model.")

CUSTOM_MODEL_PATH = Path("model/object_detection.pt")
DEFAULT_MODEL_NAME = "yolov8n.pt"

MODEL_NAME = str(CUSTOM_MODEL_PATH) if CUSTOM_MODEL_PATH.exists() else DEFAULT_MODEL_NAME

confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.25,
    step=0.05,
)


@st.cache_resource
def load_model() -> YOLO:
    return YOLO(MODEL_NAME)


try:
    model = load_model()
    st.sidebar.success("Model loaded successfully")
    st.sidebar.caption(f"Using model: {MODEL_NAME}")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()


def detections_to_df(result) -> pd.DataFrame:
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return pd.DataFrame(columns=["class", "confidence", "x1", "y1", "x2", "y2"])

    rows = []
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        rows.append(
            {
                "class": result.names.get(cls_id, str(cls_id)),
                "confidence": round(conf, 4),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            }
        )
    return pd.DataFrame(rows)


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Detecting..."):
        img_array = np.array(image.convert("RGB"))
        results = model.predict(img_array, conf=confidence)

        result = results[0]
        plotted = result.plot()
        st.image(plotted, caption="Detected Objects", use_container_width=True)

        detections_df = detections_to_df(result)
        st.subheader("Detection Details")
        st.dataframe(detections_df, use_container_width=True)

        if detections_df.empty:
            st.info("No objects detected for the selected confidence threshold.")
else:
    st.info("Please upload an image file.")
