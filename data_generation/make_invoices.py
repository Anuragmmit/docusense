import csv, os, random
from datetime import date, timedelta

import cv2
import numpy as np
from faker import Faker
from PIL import Image, ImageDraw, ImageFont

random.seed(42)
Faker.seed(42)
fake = Faker("en_IN")

CLEAN_DIR = "data/raw_clean"   # perfect images
SCAN_DIR = "data/raw"          # messy "scanned" images
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(SCAN_DIR, exist_ok=True)

N_INVOICES = 300


def get_font(size):
    # Try a few common fonts so this works on Windows, Mac and Linux
    for name in ["DejaVuSans.ttf", "arial.ttf", "Arial.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# 1) Make 15 vendors. Each invoice will pick one of them.
def make_gstin():
    pan = "".join(random.choices("ABCDEFGHIJKLMNPQRSTUVWXYZ", k=5)) \
        + "".join(random.choices("0123456789", k=4)) \
        + random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
    return f"27{pan}1Z{random.choice('0123456789')}"

vendors = [
    {"vendor_id": f"V{i:03d}", "name": fake.company(), "gstin": make_gstin(),
     "city": fake.city()}
    for i in range(1, 16)
]

ITEMS = ["Laptop Stand", "A4 Paper Ream", "Ethernet Cable", "Office Chair",
         "Printer Toner", "LED Monitor", "Keyboard", "Mouse", "USB Hub",
         "Whiteboard Marker Box", "Server Maintenance", "Cloud Hosting"]


# 2) Make the "answer key" data for one invoice
def make_invoice(idx):
    v = random.choice(vendors)
    inv_date = date(2026, 1, 1) + timedelta(days=random.randint(0, 270))
    lines = []
    for _ in range(random.randint(1, 5)):
        qty = random.randint(1, 20)
        rate = round(random.uniform(150, 25000), 2)
        lines.append((random.choice(ITEMS), qty, rate, round(qty * rate, 2)))
    subtotal = round(sum(l[3] for l in lines), 2)
    gst = round(subtotal * 0.18, 2)
    return {
        "invoice_no": f"INV-2026-{idx:05d}",
        "vendor_id": v["vendor_id"], "vendor_name": v["name"],
        "gstin": v["gstin"], "city": v["city"],
        "invoice_date": inv_date.isoformat(),
        "lines": lines, "subtotal": subtotal, "gst": gst,
        "total": round(subtotal + gst, 2),
        "is_duplicate": 0, "is_anomaly": 0,
    }


# 3) Paint the invoice as a picture
def draw_invoice(inv, path):
    img = Image.new("RGB", (1240, 1754), "white")   # A4 size at ~150 DPI
    d = ImageDraw.Draw(img)
    big, mid, small = get_font(44), get_font(30), get_font(26)

    d.text((80, 80), inv["vendor_name"], font=big, fill="black")
    d.text((80, 140), f'{inv["city"]}   GSTIN: {inv["gstin"]}', font=small, fill="black")
    d.text((900, 80), "TAX INVOICE", font=mid, fill="black")
    d.text((80, 260), f'Invoice No: {inv["invoice_no"]}', font=mid, fill="black")
    d.text((80, 310), f'Date: {inv["invoice_date"]}', font=mid, fill="black")

    y = 440
    d.text((80, y), "Item", font=mid, fill="black")
    d.text((620, y), "Qty", font=mid, fill="black")
    d.text((760, y), "Rate", font=mid, fill="black")
    d.text((1000, y), "Amount", font=mid, fill="black")
    d.line((80, y + 45, 1160, y + 45), fill="black", width=2)
    y += 70
    for name, qty, rate, amt in inv["lines"]:
        d.text((80, y), name, font=small, fill="black")
        d.text((620, y), str(qty), font=small, fill="black")
        d.text((760, y), f"{rate:,.2f}", font=small, fill="black")
        d.text((1000, y), f"{amt:,.2f}", font=small, fill="black")
        y += 50

    y += 40
    d.line((700, y, 1160, y), fill="black", width=2)
    d.text((700, y + 20), f'Subtotal: Rs. {inv["subtotal"]:,.2f}', font=small, fill="black")
    d.text((700, y + 70), f'GST 18%:  Rs. {inv["gst"]:,.2f}', font=small, fill="black")
    d.text((700, y + 120), f'TOTAL:    Rs. {inv["total"]:,.2f}', font=mid, fill="black")
    img.save(path)


# 4) Make a picture look like a bad scan
def make_dirty(path_in, path_out):
    img = cv2.imread(path_in)
    h, w = img.shape[:2]

    # crooked paper (rotate a little)
    angle = random.uniform(-6, 6)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

    # blur (shaky scanner)
    if random.random() < 0.5:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # grainy noise (dusty glass)
    noise = np.random.normal(0, random.uniform(8, 25), img.shape)
    print("Image shape:", img.shape)
    print("Image dtype:", img.dtype)
    print("Noise shape:", noise.shape)
    print("Noise dtype:", noise.dtype)
    print("Image MB:", img.nbytes / 1024 / 1024)
    print("Noise MB:", noise.nbytes / 1024 / 1024)

    # img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    img = img.astype(np.float32)

# Make sure noise has the same shape as the image
    if noise.shape != img.shape:
        noise = np.broadcast_to(noise, img.shape)

    img += noise
    np.clip(img, 0, 255, out=img)

    img = img.astype(np.uint8)


    # faded ink (low contrast)
    if random.random() < 0.4:
        img = cv2.convertScaleAbs(img, alpha=0.6, beta=70)

    cv2.imwrite(path_out, img)


# 5) Main: build everything and write the answer key
rows = []
invoices = [make_invoice(i) for i in range(1, N_INVOICES + 1)]

# Make ~3% anomalies: total is wildly inflated (a fake "fraud" signal)
for inv in random.sample(invoices, int(0.03 * N_INVOICES)):
    inv["lines"][0] = (inv["lines"][0][0], inv["lines"][0][1],
                       inv["lines"][0][2] * 15, inv["lines"][0][3] * 15)
    inv["subtotal"] = round(sum(l[3] for l in inv["lines"]), 2)
    inv["gst"] = round(inv["subtotal"] * 0.18, 2)
    inv["total"] = round(inv["subtotal"] + inv["gst"], 2)
    inv["is_anomaly"] = 1

# Add ~5% duplicates: same invoice, scanned again with a new number
dups = []
for inv in random.sample(invoices, int(0.05 * N_INVOICES)):
    d = dict(inv)
    d["invoice_no"] = inv["invoice_no"] + "-COPY"
    d["is_duplicate"] = 1
    dups.append(d)
invoices += dups

for inv in invoices:
    name = inv["invoice_no"]
    clean_path = f"{CLEAN_DIR}/{name}.png"
    scan_path = f"{SCAN_DIR}/{name}.png"
    draw_invoice(inv, clean_path)
    make_dirty(clean_path, scan_path)
    rows.append({
        "file": f"{name}.png", "invoice_no": inv["invoice_no"],
        "vendor_id": inv["vendor_id"], "vendor_name": inv["vendor_name"],
        "gstin": inv["gstin"], "invoice_date": inv["invoice_date"],
        "subtotal": inv["subtotal"], "gst": inv["gst"], "total": inv["total"],
        "is_duplicate": inv["is_duplicate"], "is_anomaly": inv["is_anomaly"],
    })

with open("data/ground_truth.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"Done! Made {len(rows)} invoices.")