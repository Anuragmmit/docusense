import random
import sqlite3
from pathlib import Path

import pandas as pd
import spacy
from spacy.training import Example


SEED = 42
random.seed(SEED)

LABELS = [
    "INV_NO",
    "DATE",
    "GSTIN",
    "VENDOR",
    "SUBTOTAL",
    "GST",
    "TOTAL",
]


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
# train_ner.py is in:
# docusense/nlp/train_ner.py
#
# parents[1] = docusense/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = PROJECT_ROOT / "data" / "docusense.db"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "ground_truth.csv"
SPLIT_PATH = PROJECT_ROOT / "data" / "split.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "ner"


# ---------------------------------------------------------
# Load OCR data
# ---------------------------------------------------------
conn = sqlite3.connect(DB_PATH)

df = pd.read_sql(
    """
    SELECT d.doc_id, d.file_name, o.text
    FROM documents d
    JOIN ocr_results o USING(doc_id)
    """,
    conn,
)

conn.close()

gt = pd.read_csv(GROUND_TRUTH_PATH)

df = df.merge(
    gt,
    left_on="file_name",
    right_on="file",
)


# ---------------------------------------------------------
# 1) Split by BASE invoice number
# ---------------------------------------------------------
# This ensures that duplicate invoices stay in the same
# train/test split.

df["base"] = df.invoice_no.str.replace(
    "-COPY",
    "",
    regex=False,
)

bases = sorted(df.base.unique())

random.shuffle(bases)

test_bases = set(
    bases[: int(0.2 * len(bases))]
)

df["split"] = df.base.apply(
    lambda b: "test" if b in test_bases else "train"
)

df[["doc_id", "split"]].to_csv(
    SPLIT_PATH,
    index=False,
)

print(
    df["split"].value_counts().to_string(),
    "\n",
)


# ---------------------------------------------------------
# 2) Auto-label: find true values inside OCR text
# ---------------------------------------------------------
def find_spans(text, row):
    spans = []

    def add(label, value, last=False):
        # Convert everything to string because OCR text
        # is always a string.
        value = str(value)

        i = (
            text.rfind(value)
            if last
            else text.find(value)
        )

        if i >= 0:
            spans.append(
                (
                    i,
                    i + len(value),
                    label,
                )
            )

    add(
        "INV_NO",
        row.invoice_no,
    )

    add(
        "DATE",
        row.invoice_date,
    )

    add(
        "GSTIN",
        row.gstin,
    )

    add(
        "VENDOR",
        row.vendor_name,
    )

    # Amounts:
    # Summary values usually occur near the bottom,
    # so use the LAST occurrence.
    add(
        "SUBTOTAL",
        f"{row.subtotal:,.2f}",
        last=True,
    )

    add(
        "GST",
        f"{row.gst:,.2f}",
        last=True,
    )

    add(
        "TOTAL",
        f"{row.total:,.2f}",
        last=True,
    )

    return spans


# ---------------------------------------------------------
# Create spaCy NER examples
# ---------------------------------------------------------
nlp = spacy.blank("en")

ner = nlp.add_pipe("ner")

for label in LABELS:
    ner.add_label(label)


examples = []
skipped = 0

train_df = df[df["split"] == "train"]

for row in train_df.itertuples():

    doc = nlp.make_doc(row.text)

    ents = []

    for s, e, label in sorted(
        find_spans(row.text, row)
    ):

        span = doc.char_span(
            s,
            e,
            label=label,
            alignment_mode="strict",
        )

        if (
            span is not None
            and (
                not ents
                or s >= ents[-1][1]
            )
        ):
            ents.append(
                (s, e, label)
            )

    # Only train on documents where all labels
    # were successfully found.
    if len(ents) < len(LABELS):
        skipped += 1
        continue

    examples.append(
        Example.from_dict(
            doc,
            {
                "entities": ents
            },
        )
    )


print(
    f"Training documents: {len(examples)} "
    f"(skipped {skipped} where OCR garbled "
    f"a value so it couldn't be auto-labelled)\n"
)


# ---------------------------------------------------------
# Safety check
# ---------------------------------------------------------
if not examples:
    raise RuntimeError(
        "No training examples were created. "
        "Check OCR text and ground_truth.csv."
    )


# ---------------------------------------------------------
# 3) Train
# ---------------------------------------------------------
optimizer = nlp.initialize(
    lambda: examples
)

for epoch in range(20):

    random.shuffle(examples)

    losses = {}

    for batch in spacy.util.minibatch(
        examples,
        size=8,
    ):

        nlp.update(
            batch,
            sgd=optimizer,
            drop=0.2,
            losses=losses,
        )

    if epoch % 5 == 0 or epoch == 19:
        print(
            f"epoch {epoch:2d}  "
            f"loss {losses['ner']:.1f}"
        )


# ---------------------------------------------------------
# 4) Save trained model
# ---------------------------------------------------------
MODEL_PATH.mkdir(
    parents=True,
    exist_ok=True,
)

nlp.to_disk(MODEL_PATH)

print(
    f"\nSaved model to: {MODEL_PATH}"
)

## The immediate fix


from pathlib import Path

Path("models/ner").mkdir(parents=True, exist_ok=True)

nlp.to_disk("models/ner")
print("Saved models/ner")
