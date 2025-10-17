
import pytesseract
from pytesseract import Output
from typing import Dict, Tuple
import cv2
import numpy as np
import pandas as pd
import re
pytesseract.pytesseract.tesseract_cmd = r"C:\venv\tesseract.exe"

# ---------- Color & preprocessing ----------
def magenta_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Keep magenta/pink letters in BGR image.
    Returns a binary mask (255 = keep).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # OpenCV hue in [0,179]; magenta spans near 300° -> ~150 and wraps near 0°
    lower1, upper1 = (np.array([140, 60, 60]), np.array([179, 255, 255]))  # magenta
    lower2, upper2 = (np.array([0,   60, 60]), np.array([10, 255, 255]))   # pinkish-red wrap
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
    return mask

def binarize_for_ocr(bgr_roi: np.ndarray) -> np.ndarray:
    """
    Make a high-contrast, OCR-friendly patch: black text on white background,
    optionally upsampled, with light sharpening.
    """
    mask = magenta_mask(bgr_roi)              # 255 on letters
    bin_img = 255 - mask                      # letters black (0), bg white (255)

    # Upsample small patches to help Tesseract
    h, w = bin_img.shape[:2]
    scale = 2 if max(h, w) < 120 else 1
    if scale > 1:
        bin_img = cv2.resize(bin_img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # Gentle denoise + unsharp for thin strokes
    bin_img = cv2.GaussianBlur(bin_img, (3, 3), 0)
    bin_img = cv2.addWeighted(bin_img, 1.5, cv2.GaussianBlur(bin_img, (0, 0), 1.0), -0.5, 0)
    return bin_img

# ---------- OCR helpers ----------
def run_tesseract(img_bin: np.ndarray, config: str) -> pd.DataFrame:
    return pytesseract.image_to_data(img_bin, lang="eng", output_type=Output.DATAFRAME, config=config)


def sanitize_tess_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    # Ensure expected columns & types
    for col in ["text", "conf", "block_num", "par_num", "line_num", "left"]:
        if col not in df:
            df[col] = 0
    df["text"] = df["text"].astype(str)
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce").fillna(-1)
    return df


def best_line_text(df: pd.DataFrame) -> Tuple[str, float]:
    """
    Pick the line with the highest mean confidence, return joined text and its mean conf.
    """
    df = sanitize_tess_df(df)
    if df.empty:
        return "", -1.0

    # Drop blanks like "", "nan"
    keep = df["text"].astype(str).str.strip().ne("") & df["text"].str.lower().ne("nan")  & df["text"].ne(",")
    df = df[keep]
    if df.empty:
        return "", -1.0

    df["line_id"] = list(zip(df["block_num"], df["par_num"], df["line_num"]))
    best = None
    best_conf = -1.0

    for _, g in df.groupby("line_id"):
        g = g.sort_values("left")
        text = " ".join(g["text"].tolist()).strip()
        conf = float(g["conf"].mean())
        if conf > best_conf:
            best, best_conf = text, conf

    return (best or "").strip(), best_conf

# ---------- Main: read text near a detection ----------
#  aim for uppercase A–Z and digits (component-style tags)
TAG_CFG  = r'--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -c user_defined_dpi=300'

def read_tag_near_point(img_bgr: np.ndarray, cx: int, cy: int, pad: int = 40) -> Tuple[str, float]:
    """
    Crop a square around (cx, cy), clean it, run two OCR configs,
    and return (best_text, best_conf).
    """
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, cx - pad), max(0, cy - pad)
    x2, y2 = min(w, cx + pad), min(h, cy + pad)

    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return "", -1.0

    roi_bin = binarize_for_ocr(roi)

    tA, cA = best_line_text(run_tesseract(roi_bin, TAG_CFG))

    return (tA, cA) 
