import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Handwriting Feature Analyzer", layout="wide")
st.title("Handwriting Feature Analyzer")
st.write("Measure selected handwriting features for research. Outputs are image-based estimates and require manual verification.")

uploaded = st.file_uploader("Upload handwriting image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])

def make_mask(gray):
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 11)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 5:
            clean[labels == i] = 255
    return clean

def get_components(mask, min_area=8):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    rows = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            rows.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                         "area": int(area), "cx": float(cents[i][0]), "cy": float(cents[i][1])})
    return pd.DataFrame(rows)

def baseline_angle(mask):
    df = get_components(mask, 10)
    if len(df) < 3:
        return None
    med_h = df["h"].median()
    df = df[df["h"] >= max(3, med_h * 0.45)]
    if len(df) < 3 or df["cx"].nunique() < 2:
        return None
    slope, _ = np.polyfit(df["cx"].to_numpy(), (df["y"] + df["h"]).to_numpy(), 1)
    return float(np.degrees(np.arctan(slope)))

def slant_angle(mask):
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 12,
                            minLineLength=max(7, mask.shape[0] // 30), maxLineGap=4)
    angles = []
    if lines is not None:
        for l in lines[:, 0]:
            x1, y1, x2, y2 = map(int, l)
            dx, dy = x2-x1, y2-y1
            if np.hypot(dx, dy) >= 7 and abs(dy) > abs(dx) * 0.8:
                a = float(np.degrees(np.arctan2(dx, -dy)))
                if -60 <= a <= 60:
                    angles.append(a)
    return float(np.median(angles)) if angles else None

def gap_data(mask):
    df = get_components(mask, 8).sort_values("x").reset_index(drop=True)
    gaps = []
    for i in range(len(df)-1):
        gap = int(df.loc[i+1, "x"] - (df.loc[i, "x"] + df.loc[i, "w"]))
        if gap >= 0:
            gaps.append(gap)
    return df, gaps

def show_num(x, unit=""):
    return "Not estimated" if x is None or not np.isfinite(x) else f"{x:.2f}{unit}"

if uploaded:
    original = np.array(Image.open(uploaded).convert("RGB"))
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    full_mask = make_mask(gray)
    H, W = full_mask.shape

    st.subheader("1. Select the exact line, word, or letter")
    st.caption("Adjust the crop around the feature you want to examine. For one letter, crop tightly around that letter.")
    c1, c2 = st.columns(2)
    with c1:
        x1 = st.slider("Crop left (pixels)", 0, max(0, W-1), 0)
        x2 = st.slider("Crop right (pixels)", 1, W, W)
    with c2:
        y1 = st.slider("Crop top (pixels)", 0, max(0, H-1), 0)
        y2 = st.slider("Crop bottom (pixels)", 1, H, H)

    if x2 <= x1 or y2 <= y1:
        st.error("The right/bottom crop edge must be greater than the left/top edge.")
        st.stop()

    rgb = original[y1:y2, x1:x2]
    mask = full_mask[y1:y2, x1:x2]
    st.subheader("2. Select the feature to analyze")
    selected_feature = st.selectbox("Feature", [
        "Line alignment",
        "Slant of selected letter/strokes",
        "Letter/component height and width",
        "Spacing between components",
        "Approximate word spacing",
        "Connecting strokes (manual inspection)",
        "All available features"
    ])

    base = baseline_angle(mask)
    slant = slant_angle(mask)
    comps, gaps = gap_data(mask)
    med_h = float(comps["h"].median()) if len(comps) else None
    med_w = float(comps["w"].median()) if len(comps) else None
    median_gap = float(np.median(gaps)) if gaps else None
    typical_w = float(comps["w"].median()) if len(comps) else 0
    word_gaps = [g for g in gaps if g > max(5, typical_w * 0.8)]
    word_gap = float(np.median(word_gaps)) if word_gaps else None

    overlay = rgb.copy()
    for _, r in comps.iterrows():
        cv2.rectangle(overlay, (int(r.x), int(r.y)),
                      (int(r.x+r.w), int(r.y+r.h)), (255, 145, 0), 1)

    left, right = st.columns(2)
    with left:
        st.markdown("**Selected crop**")
        st.image(rgb, use_container_width=True)
        st.markdown("**Detected components**")
        st.image(overlay, use_container_width=True)
    with right:
        rows = []
        if selected_feature in ["Line alignment", "All available features"]:
            rows.append(["Line alignment / baseline angle", show_num(base, "°"),
                         "Positive: baseline falls right; negative: baseline rises right"])
        if selected_feature in ["Slant of selected letter/strokes", "All available features"]:
            rows.append(["Estimated slant from vertical", show_num(slant, "°"),
                         "Approximate; works only when straight stroke segments are detected"])
        if selected_feature in ["Letter/component height and width", "All available features"]:
            rows += [
                ["Median component height", show_num(med_h, " px"), "For a tightly cropped letter/word; component may not equal letter"],
                ["Median component width", show_num(med_w, " px"), "For a tightly cropped letter/word"]
            ]
        if selected_feature in ["Spacing between components", "All available features"]:
            rows.append(["Median component gap", show_num(median_gap, " px"),
                         "Gap between detected ink components, not guaranteed letter spacing"])
        if selected_feature in ["Approximate word spacing", "All available features"]:
            rows.append(["Approximate word gap", show_num(word_gap, " px"),
                         "Heuristic; select one line and verify word boundaries manually"])
        if selected_feature in ["Connecting strokes (manual inspection)", "All available features"]:
            rows.append(["Connecting strokes", "Manual visual check required",
                         "Zoom the crop to inspect whether the selected letters physically connect"])
        st.markdown("**Selected-feature results**")
        st.dataframe(pd.DataFrame(rows, columns=["Feature", "Result", "Interpretation"]),
                     hide_index=True, use_container_width=True)

    st.subheader("3. Export")
    export = pd.DataFrame([
        ["Crop coordinates", f"x={x1}:{x2}; y={y1}:{y2}", "pixels"],
        ["Line alignment angle", show_num(base), "degrees; estimate"],
        ["Slant angle", show_num(slant), "degrees from vertical; estimate"],
        ["Median component height", show_num(med_h), "pixels"],
        ["Median component width", show_num(med_w), "pixels"],
        ["Median component gap", show_num(median_gap), "pixels"],
        ["Approximate word gap", show_num(word_gap), "pixels; heuristic"],
        ["Connecting strokes", "Manual visual verification", "not automatically classified"]
    ], columns=["Feature", "Value", "Note"])
    st.download_button("Download feature results (CSV)", export.to_csv(index=False).encode("utf-8"),
                       "handwriting_feature_results.csv", "text/csv")
    buf = BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    st.download_button("Download selected crop with boxes (PNG)", buf.getvalue(),
                       "handwriting_selected_crop.png", "image/png")

    st.warning("Important: this app does not reliably recognize letter names or automatically prove connecting strokes. For specific-letter measurements, crop one clearly visible letter at a time and record its identity yourself. Use the same image scale and procedure for human and AI samples.")
else:
    st.markdown("### Features included")
    st.markdown("- Baseline / line alignment angle\n- Slant estimate for selected strokes\n- Height and width of components in a selected letter or word\n- Gaps between ink components\n- Approximate word-gap estimate\n- Connecting-stroke visual inspection\n- Crop annotation and CSV export")
