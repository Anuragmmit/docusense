import os
import random

import cv2
import numpy as np
import pandas as pd
from faker import Faker
from PIL import Image, ImageDraw, ImageFont

SEED, N = 7, 300
random.seed(SEED); np.random.seed(SEED); Faker.seed(SEED)
fake = Faker("en_IN")
W, H = 1240, 1754
OUT = "data/doctypes"
ITEMS = ["Laptop Stand", "A4 Paper Ream", "Ethernet Cable", "Office Chair",
         "Printer Toner", "LED Monitor", "Keyboard", "Mouse", "USB Hub"]
for c in ["invoice", "purchase_order", "receipt"]:
    os.makedirs(f"{OUT}/{c}", exist_ok=True)


def get_font(size):
    for name in ["DejaVuSans.ttf", "arial.ttf", "Arial.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def to_bgr(pil):
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def to_small_gray(bgr):
    """Full-size colour -> half-size grayscale. Our CNN only needs this."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (W // 2, H // 2), interpolation=cv2.INTER_AREA)

def make_dirty(img, hard=False):
    """img is a half-size grayscale image."""
    h, w = img.shape
    lim = 12 if hard else 6
    M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-lim, lim), 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=255)
    if hard or random.random() < 0.5:
        k = 7 if hard else random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    sigma = random.uniform(25, 45) if hard else random.uniform(8, 25)
    img = np.clip(img.astype(np.float32) + np.random.normal(0, sigma, img.shape),
                  0, 255).astype(np.uint8)
    if hard or random.random() < 0.4:
        img = cv2.convertScaleAbs(img, alpha=0.5 if hard else 0.6, beta=80 if hard else 70)
    return img


def draw_po():
    img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
    big, mid, small = get_font(random.choice([40, 46])), get_font(30), get_font(26)
    d.text((80, 80), "PURCHASE ORDER", font=big, fill="black")
    d.text((80, 160), fake.company(), font=mid, fill="black")
    d.text((80, 220), f"PO No: PO-{random.randint(10000, 99999)}", font=mid, fill="black")
    d.text((80, 270), f"Order Date: {fake.date_this_year()}", font=small, fill="black")
    d.rectangle((700, 200, 1160, 380), outline="black", width=2)       # Ship-To box
    d.text((715, 210), "Ship To:", font=small, fill="black")
    d.text((715, 255), fake.street_address()[:28], font=small, fill="black")
    d.text((715, 305), f"Deliver By: {fake.date_this_year()}", font=small, fill="black")
    y = 460
    for x, t in [(80, "Item"), (600, "Qty"), (760, "Unit Price"), (980, "Line Total")]:
        d.text((x, y), t, font=mid, fill="black")
    d.line((80, y + 45, 1160, y + 45), fill="black", width=2)
    y += 70
    for _ in range(random.randint(3, 8)):
        d.text((80, y), random.choice(ITEMS), font=small, fill="black")
        d.text((600, y), str(random.randint(1, 50)), font=small, fill="black")
        d.text((760, y), f"{random.uniform(100, 20000):,.2f}", font=small, fill="black")
        d.text((980, y), f"{random.uniform(500, 90000):,.2f}", font=small, fill="black")
        y += 50
    d.text((80, 1450), "Terms: Net 30", font=small, fill="black")
    d.line((800, 1560, 1160, 1560), fill="black", width=2)
    d.text((820, 1575), "Authorised Signatory", font=small, fill="black")
    return to_bgr(img)


def draw_receipt():
    rw, rh = random.randint(480, 600), random.randint(700, 1100)
    r = Image.new("RGB", (rw, rh), "white"); d = ImageDraw.Draw(r)
    f1, f2 = get_font(34), get_font(24)
    d.text((20, 30), fake.company()[:20], font=f1, fill="black")
    d.text((20, 85), f"{fake.date_this_year()}  {random.randint(8, 21)}:{random.randint(10, 59)}",
           font=f2, fill="black")
    y = 140
    for _ in range(random.randint(3, 9)):
        d.text((20, y), random.choice(ITEMS), font=f2, fill="black")
        d.text((rw - 140, y), f"{random.uniform(20, 2000):.2f}", font=f2, fill="black")
        y += 40
    d.line((20, y + 10, rw - 20, y + 10), fill="black", width=2)
    d.text((20, y + 25), "TOTAL", font=f1, fill="black")
    d.text((rw - 170, y + 25), f"{random.uniform(200, 9000):.2f}", font=f1, fill="black")
    d.text((20, y + 90), "THANK YOU, VISIT AGAIN", font=f2, fill="black")
    x = 30                                                         # fake barcode
    while x < rw - 30 and y + 190 < rh:
        bw = random.choice([2, 3, 5])
        d.rectangle((x, y + 140, x + bw, y + 190), fill="black")
        x += bw + random.choice([2, 4])
    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(r, (random.randint(80, 500), random.randint(60, 250)))
    return to_bgr(canvas)


# Invoices come from Step 3 (originals only, no -COPY duplicates)
gt = pd.read_csv("data/ground_truth.csv")
inv_files = gt[gt.is_duplicate == 0].file.tolist()[:N]

for i in range(N):
    paths = {(c, v): f"{OUT}/{c}/{i:03d}_{v}.png"
             for c in ["invoice", "purchase_order", "receipt"]
             for v in ["normal", "hard"]}
    if all(os.path.exists(p) for p in paths.values()):
        continue                      # already done, so resume works
    sources = {
        "invoice": to_small_gray(cv2.imread(f"data/raw_clean/{inv_files[i]}")),
        "purchase_order": to_small_gray(draw_po()),
        "receipt": to_small_gray(draw_receipt()),
    }
    for cls, clean in sources.items():
        for ver, hard in [("normal", False), ("hard", True)]:
            cv2.imwrite(paths[(cls, ver)], make_dirty(clean, hard))
    if i % 50 == 0:
        print("generated", i, flush=True)
print("Done:", N, "documents per class, 2 versions each")