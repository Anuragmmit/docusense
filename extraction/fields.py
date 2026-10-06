import re
from difflib import SequenceMatcher

AMOUNT = r"(?:Rs[.,]?)?\s*([\d,]+\.\d{2})"


def to_float(s):
    return float(s.replace(",", "")) if s else None


def find_vendor(text, master, min_score=0.8):
    """
    The vendor name is at the top. OCR may misspell it, so we don't need
    an exact match: we pick the closest name from the vendor master list
    (in a real company this list comes from the ERP/vendor master).
    """
    first_line = text.splitlines()[0] if text else ""
    top = re.sub(r"TAX\s*INVOICE", "", first_line, flags=re.I).strip().lower()
    best_id, best_name, best_score = None, None, 0.0
    for vid, name in master:
        score = SequenceMatcher(None, name.lower(), top).ratio()
        if score > best_score:
            best_id, best_name, best_score = vid, name, score
    if best_score < min_score:
        return None, None, best_score
    return best_id, best_name, best_score


def extract(text, master):
    inv = re.search(r"INV-\d{4}-\d{5}(?:-COPY)?", text)
    date = re.search(r"\d{4}-\d{2}-\d{2}", text)
    gstin = re.search(r"\b27[A-Z]{5}\d{4}[A-Z]1Z\d\b", text)
    # (?<!SUB) stops "SUBTOTAL" from being mistaken for "TOTAL"
    total = re.findall(r"(?<!SUB)TOTAL\s*:?\s*" + AMOUNT, text, flags=re.I)
    subtotal = re.search(r"SUBTOTAL\s*:?\s*" + AMOUNT, text, flags=re.I)
    gst = re.search(r"GST\s*18%\s*:?\s*" + AMOUNT, text, flags=re.I)
    vid, vname, vscore = find_vendor(text, master)
    return {
        "invoice_no": inv.group(0) if inv else None,
        "invoice_date": date.group(0) if date else None,
        "gstin": gstin.group(0) if gstin else None,
        "vendor_id": vid, "vendor_name": vname, "vendor_match": vscore,
        "subtotal": to_float(subtotal.group(1)) if subtotal else None,
        "gst": to_float(gst.group(1)) if gst else None,
        "total": to_float(total[-1]) if total else None,
    }


def validate(f, mean_conf, min_conf=60):
    """Return a list of problems. Empty list = safe to auto-approve."""
    flags = []
    for k in ["invoice_no", "invoice_date", "gstin", "vendor_id",
              "subtotal", "gst", "total"]:
        if f[k] is None:
            flags.append(f"missing_{k}")
    if None not in (f["subtotal"], f["gst"], f["total"]):
        if abs(f["subtotal"] + f["gst"] - f["total"]) > 0.02:
            flags.append("total_mismatch")
        if abs(f["subtotal"] * 0.18 - f["gst"]) > 0.02:
            flags.append("gst_rate_mismatch")
    if mean_conf < min_conf:
        flags.append("low_ocr_confidence")
    return flags