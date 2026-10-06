import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def ocr_with_confidence(img, timeout=30):
    """
    Run Tesseract OCR with a timeout.

    Returns:
        text: extracted text
        mean_conf: average OCR confidence
        n: number of detected words
    """

    # ---------------------------------------------------------
    # 1. Validate image
    # ---------------------------------------------------------
    if img is None:
        return "", 0.0, 0

    # ---------------------------------------------------------
    # 2. Resize very large images
    # ---------------------------------------------------------
    max_width = 2500

    h, w = img.shape[:2]

    if w > max_width:
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)

        img = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

    # ---------------------------------------------------------
    # 3. Convert to grayscale
    # ---------------------------------------------------------
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # ---------------------------------------------------------
    # 4. OCR with timeout
    # ---------------------------------------------------------
    try:
        d = pytesseract.image_to_data(
            gray,
            config="--psm 6",
            output_type=Output.DICT,
            timeout=timeout
        )

    except RuntimeError as e:
        # pytesseract raises RuntimeError when timeout occurs
        print(f"WARNING: OCR timed out: {e}")
        return "", 0.0, 0

    except Exception as e:
        print(f"WARNING: OCR failed: {e}")
        return "", 0.0, 0

    # ---------------------------------------------------------
    # 5. Extract text and confidence
    # ---------------------------------------------------------
    texts = []
    confidences = []

    for text, conf in zip(d["text"], d["conf"]):

        text = text.strip()

        try:
            conf = float(conf)
        except (ValueError, TypeError):
            continue

        if text and conf >= 0:
            texts.append(text)
            confidences.append(conf)

    # ---------------------------------------------------------
    # 6. Calculate results
    # ---------------------------------------------------------
    if not confidences:
        return " ".join(texts), 0.0, 0

    mean_conf = float(np.mean(confidences))
    n = len(texts)

    return " ".join(texts), mean_conf, n