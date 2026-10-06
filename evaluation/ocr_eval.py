import random
import re

import cv2
import pandas as pd
import pytesseract

from preprocessing.clean import clean_image

# Windows only: uncomment and fix the path if tesseract isn't found
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

SAMPLE_SIZE = 50
SEED = 1
CONFIG = "--psm 6"

gt = pd.read_csv("data/ground_truth.csv")
gt = gt[gt["is_duplicate"] == 0]          # originals only
sample = gt.sample(SAMPLE_SIZE, random_state=SEED)


def norm(s):
    return re.sub(r"\s+", "", s).upper()


def read(img):
    return norm(pytesseract.image_to_string(img, config=CONFIG))


def score(text, row):
    return {
        "invoice_no": norm(row.invoice_no) in text,
        "date": norm(row.invoice_date) in text,
        "gstin": norm(row.gstin) in text,
        "total": f"{row.total:,.2f}" in text,
    }


results = []
for row in sample.itertuples():
    dirty = cv2.imread(f"data/raw/{row.file}")
    perfect = cv2.imread(f"data/raw_clean/{row.file}")
    cleaned, _ = clean_image(dirty)
    for version, img in [("dirty", dirty), ("cleaned", cleaned), ("perfect", perfect)]:
        s = score(read(img), row)
        results.append({"file": row.file, "version": version, **s})
    print("done", row.file)

df = pd.DataFrame(results)
df.to_csv("data/ocr_eval_results.csv", index=False)

summary = df.groupby("version")[["invoice_no", "date", "gstin", "total"]].mean()
summary["overall"] = summary.mean(axis=1)
print("\nField accuracy by version (1.0 = 100%)")
print(summary.round(3))

b = summary.loc["dirty", "overall"]
a = summary.loc["cleaned", "overall"]
print(f"\nBaseline (dirty): {b:.3f}  New (cleaned): {a:.3f}")
if b > 0:
    print(f"Relative improvement: {(a - b) / b * 100:.1f}%")
print(f"Absolute improvement: {(a - b) * 100:.1f} percentage points")