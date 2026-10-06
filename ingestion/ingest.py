import hashlib
import logging
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

RAW_DIR = Path("data/raw")
BRONZE_DIR = Path("data/bronze")
DB_PATH = Path("data/docusense.db")
ALLOWED = {".png", ".jpg", ".jpeg", ".pdf", ".tif", ".tiff"}

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("logs/ingestion.log"), logging.StreamHandler()],
)
log = logging.getLogger("ingestion")


def sha256_of_file(path: Path) -> str:
    """Read the file in small chunks so huge files don't fill memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def init_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id        TEXT PRIMARY KEY,
            file_name     TEXT,
            source_path   TEXT,
            bronze_path   TEXT,
            sha256        TEXT,
            size_bytes    INTEGER,
            source        TEXT,
            ingested_at   TEXT,
            status        TEXT,      -- INGESTED / DUPLICATE_EXACT / FAILED
            duplicate_of  TEXT,
            error_message TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sha ON documents(sha256)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_src ON documents(source_path)")
    conn.commit()
    return conn


def insert_row(conn, **r):
    conn.execute(
        """INSERT INTO documents VALUES
           (:doc_id,:file_name,:source_path,:bronze_path,:sha256,:size_bytes,
            :source,:ingested_at,:status,:duplicate_of,:error_message)""", r)
    conn.commit()


def ingest_file(conn, path: Path, source: str) -> str:
    now = datetime.now(timezone.utc)
    base = dict(
        doc_id=str(uuid.uuid4()), file_name=path.name, source_path=str(path),
        bronze_path=None, sha256=None, size_bytes=path.stat().st_size,
        source=source, ingested_at=now.isoformat(),
        status=None, duplicate_of=None, error_message=None,
    )
    try:
        digest = sha256_of_file(path)
        base["sha256"] = digest

        # Incremental: did we already handle this exact path + content?
        seen_same_path = conn.execute(
            "SELECT 1 FROM documents WHERE source_path=? AND sha256=? LIMIT 1",
            (str(path), digest)).fetchone()
        if seen_same_path:
            return "SKIPPED"

        # Exact duplicate: same fingerprint, different file
        original = conn.execute(
            "SELECT doc_id FROM documents WHERE sha256=? AND status='INGESTED' LIMIT 1",
            (digest,)).fetchone()
        if original:
            base.update(status="DUPLICATE_EXACT", duplicate_of=original[0])
            insert_row(conn, **base)
            log.warning("Exact duplicate: %s of %s", path.name, original[0])
            return "DUPLICATE_EXACT"

        # New file: copy into a dated Bronze drawer
        drawer = BRONZE_DIR / f"ingest_date={now:%Y-%m-%d}"
        drawer.mkdir(parents=True, exist_ok=True)
        dest = drawer / f"{base['doc_id']}{path.suffix.lower()}"
        shutil.copy2(path, dest)
        base.update(status="INGESTED", bronze_path=str(dest))
        insert_row(conn, **base)
        return "INGESTED"

    except Exception as e:  # one bad file must not stop the whole run
        base.update(status="FAILED", error_message=str(e))
        insert_row(conn, **base)
        log.error("Failed on %s: %s", path.name, e)
        return "FAILED"


def main(source: str = "scanner_folder"):
    conn = init_db()
    counts = {"INGESTED": 0, "DUPLICATE_EXACT": 0, "SKIPPED": 0, "FAILED": 0}
    files = sorted(p for p in RAW_DIR.iterdir() if p.suffix.lower() in ALLOWED)
    log.info("Found %d files in %s", len(files), RAW_DIR)
    for p in files:
        counts[ingest_file(conn, p, source)] += 1
    log.info("Run finished: %s", counts)
    conn.close()


if __name__ == "__main__":
    main()