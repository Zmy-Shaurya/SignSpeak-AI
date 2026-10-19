"""Train the sign classifier on data/keypoints/<sign>/*.npy and save models/sign_model.pt.

Usage: python train.py [--epochs 400] [--cv]
Labels are the folder names. 20% of each sign is held out as a test set the model never trains on.
--cv: 5-fold cross-validation instead. Every clip is tested exactly once, which gives a far steadier accuracy
than a single 20% split. Use it to compare data or training changes. It doesn't save a model.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from keypoints import FEATURES, normalize_keypoints


class SignModel(nn.Module):
    """(batch, 30 frames, 258) -> one score per sign. LSTM reads the frames in order; last state decides."""

    def __init__(self, n_classes, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(FEATURES, hidden, batch_first=True)
        self.drop = nn.Dropout(0.3)  # randomly zeroes 30% of the summary during training only
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.out(self.drop(h[-1]))


def augment(X, rng):
    """Random variation of normalized clips (N, 30, 258), new every epoch, so the model can't memorize clips.

    Speed: crop up to 15% off each end and stretch back to 30 frames.
    Space: rotate up to 10 degrees, scale 0.9-1.1, shift 0.1 shoulder-widths, plus tiny jitter.
    Missing parts (all-zero) stay zero. Visibility is left alone.
    """
    N, T = X.shape[:2]
    X = np.stack([
        x[np.linspace(rng.uniform(0, 0.15) * (T - 1), (1 - rng.uniform(0, 0.15)) * (T - 1), T).round().astype(int)]
        for x in X
    ])
    pose = X[..., :132].reshape(N, T, 33, 4).copy()
    hands = X[..., 132:].reshape(N, T, 42, 3).copy()

    a = rng.uniform(-0.17, 0.17, N)
    rot = np.stack([np.stack([np.cos(a), -np.sin(a)], -1), np.stack([np.sin(a), np.cos(a)], -1)], -2)  # (N, 2, 2)
    scale = rng.uniform(0.9, 1.1, (N, 1, 1, 1))
    shift = rng.uniform(-0.1, 0.1, (N, 1, 1, 2))

    def move(xyz):
        xy = np.einsum("nij,ntpj->ntpi", rot, xyz[..., :2]) * scale + shift
        out = np.concatenate([xy, xyz[..., 2:3] * scale], -1)
        out += rng.normal(0, 0.01, out.shape)
        return np.where(xyz.any(-1, keepdims=True), out, 0)

    pose[..., :3] = move(pose[..., :3])
    hands = move(hands)
    return np.concatenate([pose.reshape(N, T, 132), hands.reshape(N, T, 126)], -1).astype(np.float32)


def load_dataset():
    labels = sorted(p.name for p in Path("data/keypoints").iterdir() if p.is_dir())
    X, y = [], []
    for i, label in enumerate(labels):
        for f in sorted(Path("data/keypoints", label).glob("*.npy")):
            X.append(np.stack([normalize_keypoints(frame) for frame in np.load(f)]))
            y.append(i)
    return np.array(X), np.array(y), labels


def fit(X, y, n_classes, rng, epochs, report=None):
    """Train a fresh model on (X, y) with augmentation. report(epoch, loss, model) is called every 10 epochs."""
    model = SignModel(n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    yt = torch.tensor(y)
    # ponytail: full-batch training (whole train set per step). Switch to a DataLoader with mini-batches
    # once the dataset is a few thousand clips.
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(torch.tensor(augment(X, rng))), yt)
        loss.backward()
        opt.step()
        if report and (epoch % 10 == 0 or epoch == 1):
            model.eval()
            report(epoch, loss, model)
    model.eval()
    return model


def predict(model, X):
    with torch.no_grad():
        return model(torch.tensor(X)).argmax(1).numpy()


def print_confusion(y_true, y_pred, labels, title):
    cm = np.zeros((len(labels), len(labels)), int)
    np.add.at(cm, (y_true, y_pred), 1)
    width = max(len(l) for l in labels) + 2
    print(f"\n{title} (rows = true, columns = predicted)")
    print(" " * width + "".join(f"{l:>{width}}" for l in labels))
    for l, row in zip(labels, cm):
        print(f"{l:>{width}}" + "".join(f"{v:>{width}}" for v in row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--cv", action="store_true", help="5-fold cross-validation report; no model saved")
    args = parser.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    X, y, labels = load_dataset()
    print(f"{len(X)} samples, {len(labels)} signs: " + ", ".join(f"{l}={int((y == i).sum())}" for i, l in enumerate(labels)))

    probe = X[:8].copy()
    probe[..., 195:] = 0
    assert not augment(probe, rng)[..., 195:].any(), "augment must keep missing hands at zero"

    if args.cv:
        # Stratified folds: each sign's clips are dealt round-robin into 5 folds.
        fold = np.empty(len(y), int)
        for c in range(len(labels)):
            idx = rng.permutation(np.where(y == c)[0])
            fold[idx] = np.arange(len(idx)) % 5
        assert all((fold[y == c] == k).any() for c in range(len(labels)) for k in range(5)), "every sign needs 5+ clips"
        pred = np.empty_like(y)
        for k in range(5):
            model = fit(X[fold != k], y[fold != k], len(labels), rng, args.epochs)
            pred[fold == k] = predict(model, X[fold == k])
            print(f"fold {k + 1}/5: {(pred[fold == k] == y[fold == k]).mean():.0%}")
        accs = [(pred[fold == k] == y[fold == k]).mean() for k in range(5)]
        print(f"\n5-fold accuracy: {np.mean(accs):.1%} (std {np.std(accs):.1%}), every clip tested once")
        per = sorted((float((pred[y == c] == c).mean()), labels[c]) for c in range(len(labels)))
        print("per sign: " + ", ".join(f"{l} {a:.0%}" for a, l in per))
        print_confusion(y, pred, labels, "Cross-validation confusion matrix, all clips")
        raise SystemExit

    # Stratified split: every sign gets ~20% in test.
    test_idx = np.concatenate([
        rng.permutation(np.where(y == c)[0])[:max(1, round((y == c).sum() * 0.2))] for c in range(len(labels))
    ])
    train_mask = np.ones(len(X), bool)
    train_mask[test_idx] = False
    Xte, yte = X[~train_mask], y[~train_mask]

    def report(epoch, loss, model):
        train_acc = (predict(model, X[train_mask]) == y[train_mask]).mean()
        print(f"epoch {epoch:4d}  train loss {loss:.3f} acc {train_acc:.0%}  |  test acc {(predict(model, Xte) == yte).mean():.0%}")

    model = fit(X[train_mask], y[train_mask], len(labels), rng, args.epochs, report)
    print_confusion(yte, predict(model, Xte), labels, "Test confusion matrix")

    Path("models").mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "labels": labels}, "models/sign_model.pt")
    print("\nSaved models/sign_model.pt")
