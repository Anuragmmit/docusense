import logging
import sqlite3
from pathlib import Path

import cv2

from preprocessing.clean import clean_image

DB_PATH = "data/docusense.db"
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("cleaning")

conn = sqlite3.connect(DB_PATH)
conn.execute("""CREATE TABLE IF NOT EXISTS image_quality (
    doc_id TEXT PRIMARY KEY, skew_angle REAL, blur_score REAL,
    processed_path TEXT)""")

rows = conn.execute("""SELECT doc_id, bronze_path FROM documents
                       WHERE status='INGESTED'
                       AND doc_id NOT IN (SELECT doc_id FROM image_quality)""").fetchall()
log.info("%d documents to clean", len(rows))

for doc_id, bronze_path in rows:
    try:
        img = cv2.imread(bronze_path)
        cleaned, info = clean_image(img)
        out = OUT_DIR / f"{doc_id}.png"
        cv2.imwrite(str(out), cleaned)
        conn.execute("INSERT INTO image_quality VALUES (?,?,?,?)",
                     (doc_id, info["skew_angle"], info["blur_score"], str(out)))
        conn.commit()
    except Exception as e:
        log.error("Failed %s: %s", doc_id, e)

log.info("Cleaning finished")
conn.close()