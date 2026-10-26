import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Handwriting Feature Measurement", page_icon="✍️", layout="wide")
st.title("Handwriting Feature Measurement Tool")
st.write("Upload a clear scan/photo to estimate selected image-based handwriting features. This is a research aid, not an authorship detector or forensic conclusion.")
st.info("For a fair comparison, use images captured at the same resolution and scale, with similar lighting, page orientation, and contrast. Keep original files and document preprocessing.")

uploaded = st.file_uploader("Upload a handwriting image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    rgb = np.array(pil)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded image")
        st.image(pil, use_container_width=True)
    # Otsu threshold; assumes writing is darker than paper
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Remove tiny isolated noise, preserve larger strokes
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    page_h, page_w = gray.shape
    min_area = max(8, int(page_h * page_w * 0.000005))
    components = []
    clean = np.zeros_like(binary)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area and w >= 2 and h >= 2:
            components.append((x, y, w, h, area))
            clean[labels == i] = 255
    if not components:
        st.error("No usable foreground components were detected. Try a sharper, higher-contrast image.")
    else:
        heights = np.array([c[3] for c in components], dtype=float)
        widths = np.array([c[2] for c in components], dtype=float)
        areas = np.array([c[4] for c in components], dtype=float)
        ink_coverage = float(np.count_nonzero(clean)) / float(page_h * page_w) * 100
        # Approximate baseline from component bottoms, using a linear fit to the lower edges
        bottoms = np.array([c[1] + c[3] for c in components], dtype=float)
        xs = np.array([c[0] + c[2] / 2 for c in components], dtype=float)
        if len(components) >= 3 and np.std(xs) > 0:
            slope, intercept = np.polyfit(xs, bottoms, 1)
            baseline_angle = float(np.degrees(np.arctan(slope)))
        else:
            baseline_angle = float('nan')
        # Approximate stroke thickness via distance transform on the cleaned binary mask
        dist = cv2.distanceTransform((clean > 0).astype(np.uint8), cv2.DIST_L2, 3)
        skeleton = cv2.ximgproc.thinning(clean) if hasattr(cv2, 'ximgproc') else None
        if skeleton is not None and np.count_nonzero(skeleton):
            stroke_width = float(np.median(dist[skeleton > 0]) * 2)
        else:
            # Fallback: distance transform values on foreground pixels; less reliable
            vals = dist[clean > 0]
            stroke_width = float(np.median(vals) * 2) if vals.size else float('nan')
        metrics = [
            ("Image width", page_w, "pixels"),
            ("Image height", page_h, "pixels"),
            ("Detected connected components", len(components), "count; not equivalent to letters"),
            ("Median component height", round(float(np.median(heights)), 2), "pixels"),
            ("Median component width", round(float(np.median(widths)), 2), "pixels"),
            ("Median component height/width ratio", round(float(np.median(heights / np.maximum(widths, 1))), 3), "ratio"),
            ("Foreground ink coverage", round(ink_coverage, 3), "% of image"),
            ("Approximate baseline angle", round(baseline_angle, 3) if np.isfinite(baseline_angle) else "Not enough components", "degrees; rough estimate"),
            ("Approximate visible stroke width", round(stroke_width, 2) if np.isfinite(stroke_width) else "Unavailable", "pixels; rough estimate"),
        ]
        df = pd.DataFrame(metrics, columns=["Feature", "Value", "Unit / interpretation"])
        with col2:
            st.subheader("Estimated measurements")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download measurements (CSV)", df.to_csv(index=False).encode("utf-8-sig"), file_name="handwriting_measurements.csv", mime="text/csv")
        st.subheader("Preprocessed foreground mask")
        st.image(clean, caption="Thresholded foreground after small-component filtering", clamp=True, use_container_width=True)
        st.caption("Interpretation cautions: connected components may split or join letters, particularly in cursive writing. Pixel values are not physical millimetres unless image scale is calibrated. The baseline and stroke-width values are approximate. The app cannot determine pen pressure, writing speed, or whether a sample is AI-generated.")
else:
    st.markdown("### What this tool estimates\n- Component height and width (not true letter segmentation)\n- Median height-to-width ratio\n- Approximate baseline angle\n- Foreground ink coverage\n- Approximate visible stroke width")
    st.markdown("### Suggested workflow for your study\n1. Use anonymous sample IDs (e.g., H01–H10 and A01–A10).\n2. Scan all samples at the same resolution and keep the page scale consistent.\n3. Upload one image at a time and save each CSV with the matching sample ID.\n4. Combine the CSVs in Excel and compare groups descriptively.\n5. Report this as exploratory analysis; 10 samples per group is not enough to validate a reliable forensic classification method.")
