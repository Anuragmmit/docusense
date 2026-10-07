import copy
import time

import cv2
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

CLASSES = ["invoice", "purchase_order", "receipt"]
N, SIZE, EPOCHS = 300, 128, 15
SPLIT_SEED, SEEDS = 42, [0, 1, 2]


def read(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return 1.0 - g.astype(np.float32) / 255.0      # ink = 1, paper = 0


def build(idx, ver):
    X, y = [], []
    for ci, c in enumerate(CLASSES):
        for i in idx:
            X.append(read(f"data/doctypes/{c}/{i:03d}_{ver}.png")); y.append(ci)
    return np.stack(X)[:, None], np.array(y)       # (n, 1, 128, 128)


# Split by document index: train 70% / val 15% / test 15%, fixed seed
perm = np.random.RandomState(SPLIT_SEED).permutation(N)
tr, va, te = perm[:210], perm[210:255], perm[255:]
Xtr, ytr = build(tr, "normal")
Xva, yva = build(va, "normal")
tests = {"normal": build(te, "normal"), "hard": build(te, "hard")}
print(f"train {len(ytr)} | val {len(yva)} | test {len(tests['normal'][1])} per condition\n")


class SmallCNN(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(64, n))

    def forward(self, x):
        return self.head(self.features(x))


def predict(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            out.append(model(torch.from_numpy(X[i:i + 64])).argmax(1).numpy())
    return np.concatenate(out)


def train(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state = -1, None
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(ytr))
        for i in range(0, len(order), 32):
            b = order[i:i + 32]
            opt.zero_grad()
            loss = loss_fn(model(torch.from_numpy(Xtr[b.numpy()])),
                           torch.from_numpy(ytr[b.numpy()]))
            loss.backward(); opt.step()
        val_acc = accuracy_score(yva, predict(model, Xva))
        if val_acc > best_acc:                       # keep the best epoch by VALIDATION
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_acc


def scores(y, p):
    return {"acc": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro")}


# ---------- Baseline: logistic regression on 32x32 pixels ----------
small = lambda X: X.reshape(len(X), 32, 4, 32, 4).mean((2, 4)).reshape(len(X), -1)
base = LogisticRegression(max_iter=3000).fit(small(Xtr), ytr)
rows = []
for cond, (X, y) in tests.items():
    rows.append({"model": "logreg_pixels", "condition": cond, "seed": "-",
                 **scores(y, base.predict(small(X)))})

# ---------- CNN, three seeds ----------
latencies, best_model = [], None
for seed in SEEDS:
    model, val_acc = train(seed)
    for cond, (X, y) in tests.items():
        p = predict(model, X)
        rows.append({"model": "cnn", "condition": cond, "seed": seed, **scores(y, p)})
        if seed == SEEDS[0]:
            print(f"Confusion matrix, CNN seed {seed}, {cond} test "
                  f"(rows = truth, cols = predicted; {CLASSES}):")
            print(confusion_matrix(y, p), "\n")
    if seed == SEEDS[0]:
        best_model = model
    try:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("document_classifier")
        with mlflow.start_run(run_name=f"cnn_seed{seed}"):
            mlflow.log_params({"epochs": EPOCHS, "img_size": SIZE, "seed": seed, "lr": 1e-3})
            mlflow.log_metric("val_acc", val_acc)
            for r in [r for r in rows if r["model"] == "cnn" and r["seed"] == seed]:
                mlflow.log_metric(f"test_{r['condition']}_macro_f1", r["macro_f1"])
    except Exception as e:
        print("MLflow skipped:", e)

df = pd.DataFrame(rows)
df.to_csv("data/dl_results.csv", index=False)
summary = df.groupby(["model", "condition"])[["acc", "macro_f1"]].agg(["mean", "std"]).round(3)
print("Results (std is across seeds; logreg has one run so std is blank)")
print(summary, "\n")

# ---------- Cost: size and speed ----------
torch.save(best_model.state_dict(), "models/doc_cnn.pt")
n_params = sum(p.numel() for p in best_model.parameters())
x = torch.from_numpy(Xtr[:1])
best_model.eval()
with torch.no_grad():
    for _ in range(20):
        best_model(x)
    t = time.perf_counter()
    for _ in range(100):
        best_model(x)
cnn_ms = (time.perf_counter() - t) / 100 * 1000
x_small = small(Xtr[:1])
t = time.perf_counter()
for _ in range(100):
    base.predict(x_small)
base_ms = (time.perf_counter() - t) / 100 * 1000
print(f"CNN parameters: {n_params:,} | latency: {cnn_ms:.2f} ms/image (CPU, batch=1)")
print(f"Logreg latency: {base_ms:.2f} ms/image")

# ---------- Improvement formulas (hard condition, macro F1) ----------
b = df[(df.model == "logreg_pixels") & (df.condition == "hard")].macro_f1.mean()
a = df[(df.model == "cnn") & (df.condition == "hard")].macro_f1.mean()
print(f"\nHard test, macro F1: baseline {b:.3f}, CNN {a:.3f}")
print(f"Absolute: {(a - b) * 100:+.1f} points | Relative: {(a - b) / b * 100:+.1f}%")
if b < 1:
    print(f"Error reduction: {((1 - b) - (1 - a)) / (1 - b) * 100:.1f}%")