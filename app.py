# Install YOLOv8 and Pillow (for creating dummy images)
!pip install ultralytics pillow

# Import library
from ultralytics import YOLO
import os
from PIL import Image
import numpy as np

# Create dataset YAML file
data_yaml = """
path: dataset
train: images/train
val: images/val

names:
  0: person
  1: car
  2: bike
"""

# Save YAML file
os.makedirs("dataset", exist_ok=True)
with open("dataset/data.yaml", "w") as f:
    f.write(data_yaml)

# Create image and label directories
img_train_path = "dataset/images/train"
label_train_path = "dataset/labels/train"
img_val_path = "dataset/images/val"
label_val_path = "dataset/labels/val"

os.makedirs(img_train_path, exist_ok=True)
os.makedirs(label_train_path, exist_ok=True)
os.makedirs(img_val_path, exist_ok=True)
os.makedirs(label_val_path, exist_ok=True)

# Create dummy images and labels for demonstration
# In a real scenario, you would replace these with your actual dataset
for i in range(5): # Create 5 dummy training images/labels
    # Dummy image file (create a blank 640x640 RGB image)
    img = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
    img.save(f"{img_train_path}/train_img_{i}.jpg")
    # Dummy label file (example: class 0, center x, center y, width, height)
    with open(f"{label_train_path}/train_img_{i}.txt", "w") as f:
        f.write("0 0.5 0.5 0.5 0.5")

for i in range(2): # Create 2 dummy validation images/labels
    # Dummy image file (create a blank 640x640 RGB image)
    img = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
    img.save(f"{img_val_path}/val_img_{i}.jpg")
    # Dummy label file
    with open(f"{label_val_path}/val_img_{i}.txt", "w") as f:
        f.write("1 0.3 0.3 0.4 0.4")

print("Dummy dataset created.")

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")  # nano version (fast)

# Train model
model.train(
    data="dataset/data.yaml",
    epochs=2,  # Reduced epochs for quicker demonstration
    imgsz=640,
    batch=2,   # Reduced batch size for quicker demonstration
    name="custom_yolo_model"
)

# Test prediction (optional)
results = model.predict(source="https://ultralytics.com/images/bus.jpg", show=True)

# Print results
print("Training + Testing Completed ✅")

%%writefile app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Object Detection", layout="wide")

st.title("🚀 Object Detection")
st.write("Upload an image to run inference using your trained model.")

# Path to your trained model
# Note: model.train creates a 'runs/detect/custom_yolo_model/weights/best.pt'
MODEL_PATH = 'runs/detect/custom_yolo_model/weights/best.pt'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    else:
        # Fallback to base model if custom isn't trained yet
        return YOLO('yolov8n.pt')

try:
    model = load_model()
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not null:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button('Run Detection'):
        with st.spinner('Detecting...'):
            # Convert PIL to numpy
            img_array = np.array(image)
            # Run inference
            results = model(img_array)

            # Visualize results
            res_plotted = results[0].plot()
            st.image(res_plotted, caption='Detected Objects', use_container_width=True)

            # Show raw data
            with st.expander("See Detection Details"):
                st.write(results[0].tojson())
else:
    st.info("Please upload an image file.")

import os
import time
import re

# 1. Download cloudflared only if not present
if not os.path.exists("cloudflared-linux-amd64"):
    !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    !chmod +x cloudflared-linux-amd64
    print("Downloaded cloudflared")
else:
    print("cloudflared already exists")

# 2. Kill old processes (important in Colab reruns)
!pkill -f streamlit
!pkill -f cloudflared

# 3. Start Streamlit
!streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &

# 4. Start tunnel
!nohup ./cloudflared-linux-amd64 tunnel --url http://localhost:8501 > tunnel.log 2>&1 &

# 5. Wait for tunnel to initialize
time.sleep(5)

# 6. Extract and print the URL
url = None
with open("tunnel.log", "r") as f:
    for line in f:
        match = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line)
        if match:
            url = match.group(0)
            break

if url:
    print(f"\n🚀 Your app is live at:\n{url}\n")
else:
    print("❌ Tunnel URL not found yet. Try increasing sleep time.")
