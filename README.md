# Object Detection Streamlit App

A simple Streamlit app for object detection on uploaded images.

## Features

- Upload image (`jpg`, `jpeg`, `png`, `webp`)
- Automatic detection right after upload
- Confidence threshold slider
- Annotated output image
- Detection table with class, confidence, and bounding boxes
- Model fallback:
  - Uses `model/object_detection.pt` if present

## Project Files

- `app.py` - Streamlit application
- `requirements.txt` - Python dependencies
- `packages.txt` - Linux system dependencies for Streamlit Cloud

## Local Run

From the project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

Notes:

- If `streamlit run app.py` gives exit code `127`, use:

```bash
python -m streamlit run app.py
```

- This ensures Streamlit runs from the active virtual environment.

## Optional Custom Model

If you want to use your own trained model, place it at:

```text
model/object_detection.pt
```

The app will automatically use it.

## Streamlit Cloud Deployment

Ensure these files are in the repo root before deploying:

- `app.py`
- `requirements.txt`
- `packages.txt`

Current cloud config:

- `packages.txt`:
  - `libgl1`
  - `libglib2.0-0t64`
  - `libsm6`
  - `libxext6`
  - `libxrender1`

After pushing changes:

1. Reboot app (full rebuild)
2. Clear cache if needed

## Common Issues

### `ImportError: libGL.so.1` or `libgthread-2.0.so.0`

Cause: missing Linux shared libraries in cloud runtime.

Fix:

- Keep `packages.txt` in repo root with the listed packages
- Reboot app for a full dependency rebuild

### YOLO prediction logs in terminal

The app already disables verbose model logs with `verbose=False`.

## License

Use this project as a starter template for learning and demos.
