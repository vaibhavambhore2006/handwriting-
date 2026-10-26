
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title="Handwriting Feature Measurement",
    page_icon="✍️",
    layout="wide"
)

st.title("Handwriting Feature Measurement Tool")
st.write(
    "Exploratory image-based measurements for handwriting research. "
    "This application does not identify authors or establish forensic conclusions."
)

st.info(
    "Use clear, consistently scanned samples. Measurements are in pixels "
    "unless otherwise stated. Individual cursive letters may not be separated "
    "correctly, so inspect the annotated image and verify results manually."
)

uploaded = st.file_uploader(
    "Upload a handwriting image",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
)


def group_rows(binary):
    """Find approximate text-line bands from horizontal ink projection."""
    projection = np.sum(binary > 0, axis=1)
    active = projection > max(2, binary.shape[1] * 0.002)

    indices = np.flatnonzero(active)
    if len(indices) == 0:
        return []

    # Join nearby active rows into line bands.
    bands = []
    start = previous = int(indices[0])

    for y in indices[1:]:
        y = int(y)
        if y - previous > 5:
            bands.append((start, previous + 1))
            start = y
        previous = y

    bands.append((start, previous + 1))
    return bands


def detect_components(binary, min_area):
    """Return filtered connected-component bounding boxes."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    components = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area >= min_area and w >= 2 and h >= 2:
            components.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": int(area),
                "cx": float(x + w / 2),
                "cy": float(y + h / 2),
                "bottom": float(y + h)
            })
    return components


def estimate_slant(binary):
    """
    Estimate the slant of detectable line segments.
    Angles are measured relative to vertical:
    positive = leaning right; negative = leaning left.
    """
    edges = cv2.Canny(binary, 40, 120)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=15,
        minLineLength=12,
        maxLineGap=5
    )

    angles = []
    lengths = []

    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = map(int, line)
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))

            # Near-vertical segments are more useful for slant estimation.
            if length >= 12 and abs(dy) >= abs(dx) * 0.8:
                # Image y increases downward. Normalize direction.
                if dy < 0:
                    dx, dy = -dx, -dy

                angle = float(np.degrees(np.arctan2(dx, dy)))
                if -60 <= angle <= 60:
                    angles.append(angle)
                    lengths.append(length)

    if not angles:
        return None, []

    return float(np.median(angles)), list(zip(angles, lengths))


def fit_baseline(items):
    """Fit a line to component bottoms in one approximate text band."""
    if len(items) < 3:
        return None

    xs = np.array([c["cx"] for c in items], dtype=float)
    ys = np.array([c["bottom"] for c in items], dtype=float)

    if np.ptp(xs) < 10:
        return None

    # Bottom edges of detached dots can distort a baseline.
    # Use the lower half of bottom-edge values as a rough robust fit.
    cutoff = np.percentile(ys, 50)
    keep = ys >= cutoff

    if np.count_nonzero(keep) < 3:
        keep = np.ones(len(xs), dtype=bool)

    slope, intercept = np.polyfit(xs[keep], ys[keep], 1)
    angle = float(np.degrees(np.arctan(slope)))

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "angle": angle
    }


if uploaded:
    try:
        pil = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("The image could not be opened. Please try another image.")
        st.stop()

    rgb = np.array(pil)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    page_h, page_w = gray.shape

    # Threshold dark ink against a light background.
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    min_area = max(8, int(page_h * page_w * 0.000005))
    components = detect_components(binary, min_area)

    if not components:
        st.error(
            "No usable handwriting components detected. "
            "Try a clearer image with better contrast."
        )
        st.stop()

    clean = np.zeros_like(binary)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area >= min_area and w >= 2 and h >= 2:
            clean[labels == i] = 255

    components = detect_components(clean, min_area)
    heights = np.array([c["h"] for c in components], dtype=float)
    widths = np.array([c["w"] for c in components], dtype=float)

    # Find approximate writing-line bands.
    bands = group_rows(clean)
    line_data = []
    used_component_ids = set()

    for line_number, (y1, y2) in enumerate(bands, start=1):
        items = [
            c for c in components
            if y1 <= c["cy"] < y2
        ]

        if not items:
            continue

        # A band containing only dots/noise is unlikely to be a text line.
        if len(items) < 2:
            continue

        items = sorted(items, key=lambda c: c["x"])
        baseline = fit_baseline(items)

        line_data.append({
            "line": line_number,
            "y1": y1,
            "y2": y2,
            "items": items,
            "baseline": baseline
        })

        for c in items:
            used_component_ids.add(id(c))

    # Slant estimation across the page.
    median_slant, stroke_angles = estimate_slant(clean)

    # Approximate spacing between neighboring components in each line.
    gaps = []
    gap_records = []

    for line in line_data:
        items = line["items"]

        for left, right in zip(items, items[1:]):
            gap = right["x"] - (left["x"] + left["w"])

            # Overlapping bounding boxes do not represent a positive gap.
            if gap >= 0:
                gaps.append(float(gap))
                gap_records.append({
                    "line": line["line"],
                    "x1": left["x"] + left["w"],
                    "x2": right["x"],
                    "y": int(
                        (left["cy"] + right["cy"]) / 2
                    ),
                    "gap": float(gap)
                })

    # A component touching another component after binarization may be
    # a connected ink region, not necessarily a single letter.
    components_per_line = [
        len(line["items"]) for line in line_data
    ]

    ink_coverage = (
        np.count_nonzero(clean) / (page_h * page_w) * 100
    )

    dist = cv2.distanceTransform(
        (clean > 0).astype(np.uint8), cv2.DIST_L2, 3
    )

    stroke_width = float("nan")
    try:
        import cv2.ximgproc
        skeleton = cv2.ximgproc.thinning(clean)
        values = dist[skeleton > 0]
        if values.size:
            stroke_width = float(np.median(values) * 2)
    except (ImportError, AttributeError, cv2.error):
        values = dist[clean > 0]
        if values.size:
            stroke_width = float(np.median(values) * 2)

    # Build results.
    metrics = [
        ("Image width", page_w, "pixels"),
        ("Image height", page_h, "pixels"),
        ("Detected connected components", len(components),
         "count; not equivalent to letters"),
        ("Median component height", round(float(np.median(heights)), 2),
         "pixels; not necessarily letter height"),
        ("Median component width", round(float(np.median(widths)), 2),
         "pixels; not necessarily letter width"),
        ("Median component height/width ratio",
         round(float(np.median(
             heights / np.maximum(widths, 1)
         )), 3), "ratio"),
        ("Foreground ink coverage", round(float(ink_coverage), 3),
         "% of image"),
        ("Approximate visible stroke width",
         round(stroke_width, 2) if np.isfinite(stroke_width)
         else "Unavailable", "pixels; rough estimate"),
        ("Estimated median stroke slant",
         round(median_slant, 2) if median_slant is not None
         else "Not detected", "degrees from vertical; rough estimate"),
        ("Approximate text lines detected", len(line_data),
         "count; may split or merge lines"),
        ("Median component gap",
         round(float(np.median(gaps)), 2) if gaps else "Unavailable",
         "pixels; not true letter spacing"),
        ("Mean component gap",
         round(float(np.mean(gaps)), 2) if gaps else "Unavailable",
         "pixels; not true letter spacing"),
    ]

    for line in line_data:
        base = line["baseline"]
        metrics.append((
            f"Line {line['line']} baseline angle",
            round(base["angle"], 3) if base else "Not enough components",
            "degrees; approximate"
        ))

    baseline_angles = [
        line["baseline"]["angle"]
        for line in line_data if line["baseline"] is not None
    ]

    if baseline_angles:
        metrics.append((
            "Baseline angle variation",
            round(float(np.std(baseline_angles)), 3),
            "degrees; SD across detected lines"
        ))

    metrics.extend([
        ("Components per detected line",
         "; ".join(map(str, components_per_line))
         if components_per_line else "Unavailable",
         "count per line; not letter count"),
        ("Connecting-stroke assessment",
         "See notes below",
         "Connected regions do not prove individual letter connections"),
    ])

    df = pd.DataFrame(
        metrics,
        columns=["Feature", "Value", "Unit / interpretation"]
    )

    # Annotated image.
    annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Draw component boxes.
    for c in components:
        cv2.rectangle(
            annotated,
            (c["x"], c["y"]),
            (c["x"] + c["w"], c["y"] + c["h"]),
            (40, 140, 255), 1
        )

    # Draw fitted baselines and line labels.
    for line in line_data:
        base = line["baseline"]
        if base is not None:
            x1 = 0
            x2 = page_w - 1
            y1 = int(base["slope"] * x1 + base["intercept"])
            y2 = int(base["slope"] * x2 + base["intercept"])

            if -page_h <= y1 <= 2 * page_h and -page_h <= y2 <= 2 * page_h:
                cv2.line(
                    annotated, (x1, y1), (x2, y2),
                    (0, 180, 0), 2
                )

            cv2.putText(
                annotated, f"Line {line['line']}",
                (10, max(20, line["y1"])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 130, 0), 1, cv2.LINE_AA
            )

    # Draw sampled detected slant segments.
    edges = cv2.Canny(clean, 40, 120)
    detected_lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 15,
        minLineLength=12, maxLineGap=5
    )

    if detected_lines is not None:
        shown = 0
        for line in detected_lines[:, 0]:
            x1, y1, x2, y2 = map(int, line)
            dx, dy = x2 - x1, y2 - y1
            length = np.hypot(dx, dy)

            if length >= 12 and abs(dy) >= abs(dx) * 0.8:
                cv2.line(
                    annotated, (x1, y1), (x2, y2),
                    (220, 40, 40), 2
                )
                shown += 1
                if shown >= 100:
                    break

    # Draw measured component gaps.
    for record in gap_records:
        if record["x2"] > record["x1"]:
            cv2.line(
                annotated,
                (record["x1"], record["y"]),
                (record["x2"], record["y"]),
                (180, 0, 180), 2
            )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded handwriting")
        st.image(pil, use_container_width=True)

        st.subheader("Annotated measurements")
        st.image(
            annotated,
            caption=(
                "Orange: component boxes | Green: approximate baselines | "
                "Red: selected near-vertical strokes | Purple: component gaps"
            ),
            use_container_width=True
        )

    with col2:
        st.subheader("Estimated measurements")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download measurements (CSV)",
            df.to_csv(index=False).encode("utf-8-sig"),
            file_name="handwriting_measurements.csv",
            mime="text/csv"
        )

    st.subheader("Detected line details")
    if line_data:
        line_rows = []
        for line in line_data:
            base = line["baseline"]
            line_rows.append({
                "Line": line["line"],
                "Components in line": len(line["items"]),
                "Baseline angle (degrees)": (
                    round(base["angle"], 3) if base else None
                ),
                "Approximate line height (pixels)": (
                    line["y2"] - line["y1"]
                )
            })
        st.dataframe(
            pd.DataFrame(line_rows),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning(
            "No reliable line groups detected. Try a clearer, "
            "higher-contrast image with more separation between lines."
        )

    st.subheader("Interpretation and limitations")
    st.markdown("""
    - **Slant:** estimated from detectable near-vertical image segments;
      it is not a measurement of every letter.
    - **Spacing:** calculated between connected-component bounding boxes.
      In cursive handwriting, these boxes may not correspond to individual letters.
    - **Baseline and alignment:** fitted from component bottoms within
      approximate horizontal ink bands. Dots and descenders can affect the fit.
    - **Connecting strokes:** connected components describe connected ink
      regions, not necessarily individual letters or actual pen connections.
    - **Stroke width:** estimated from the binary image and distance transform.
    - **Scale:** pixel measurements cannot be converted to millimetres without
      a suitable calibration reference.
    - **Forensic use:** verify measurements manually and report the method,
      image conditions, limitations and uncertainty. Do not use this tool alone
      to determine authorship or whether handwriting is AI-generated.
    """)

else:
    st.markdown("""
    ### Available measurements
    - Component height and width
    - Approximate slant of detectable strokes
    - Approximate baseline angle for detected writing lines
    - Component spacing estimates
    - Approximate stroke width and ink coverage
    - Annotated image and CSV export

    ### Recommended research workflow
    1. Assign anonymous sample IDs.
    2. Scan all samples at the same resolution and scale.
    3. Keep the original images unchanged.
    4. Review the annotated output and manually verify measurements.
    5. Record uncertain or failed measurements instead of guessing.
    6. Compare groups using appropriate statistical methods.
    """)
