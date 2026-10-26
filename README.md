# Handwriting Feature Measurement Tool

A small Streamlit web app for approximate image-based handwriting feature measurements.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy as a website using Streamlit Community Cloud

1. Create/sign in to a GitHub account.
2. Create a new public repository (for example, `handwriting-feature-tool`).
3. Upload `app.py` and `requirements.txt` from this folder to the repository root.
4. Open https://share.streamlit.io/ and connect GitHub if prompted.
5. Choose the repository, branch `main`, and main file path `app.py`, then deploy.
6. After deployment, use the generated website URL.

## Limitations

This is not a validated forensic tool. Connected components are not necessarily letters; cursive strokes may merge and letters may fragment. Values are in pixels unless scale is calibrated. Image-based analysis cannot recover actual pen pressure or writing speed and does not determine whether handwriting is human- or AI-generated. Avoid uploading identifiable or confidential samples to a public cloud app unless you have appropriate permission and understand the hosting/privacy settings.
