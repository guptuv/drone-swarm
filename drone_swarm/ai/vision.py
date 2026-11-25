"""Computer vision: CNN that detects people in aerial camera patches.

A procedural renderer creates 32x32 RGB aerial patches - grass / soil textures, trees,
rocks, colourful debris (hard negatives) and people (bright clothing + head, random
orientation).  Flying altitude controls the ground resolution: the higher the drone,
the smaller the person, the more blur and sensor noise.

The CNN is trained on mixed altitudes; then detection performance vs altitude gives the
classic search trade-off: high = big camera footprint but worse detection.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter, rotate, zoom

PATCH = 32
CLOTHES = np.array([[0.9, 0.1, 0.1], [1.0, 0.55, 0.0], [0.1, 0.3, 0.95], [0.95, 0.9, 0.1], [0.9, 0.2, 0.8], [0.95, 0.95, 0.95]])


def _background(rng, S):
    base = np.array([0.25, 0.45, 0.18]) if rng.random() < 0.7 else np.array([0.5, 0.4, 0.28])
    tex = gaussian_filter(rng.normal(0, 1, (S, S)), 2.0)
    img = base[None, None] + 0.08 * tex[..., None] + rng.normal(0, 0.02, (S, S, 3))
    for _ in range(rng.integers(0, 4)):                                         # trees
        c = rng.uniform(0, S, 2); r = rng.uniform(S * 0.06, S * 0.16)
        yy, xx = np.mgrid[:S, :S]
        m = ((yy - c[0]) ** 2 + (xx - c[1]) ** 2) < r ** 2
        img[m] = np.array([0.08, 0.25, 0.08]) + rng.normal(0, 0.03, 3)
    for _ in range(rng.integers(0, 3)):                                         # rocks
        c = rng.uniform(0, S, 2); r = rng.uniform(S * 0.03, S * 0.08)
        yy, xx = np.mgrid[:S, :S]
        m = ((yy - c[0]) / 1.5) ** 2 + (xx - c[1]) ** 2 < r ** 2
        img[m] = 0.55 + rng.normal(0, 0.03)
    return img


def _debris(img, rng, scale):
    """Hard negative: colourful square object (tarp, car part) - not person shaped."""
    S = img.shape[0]
    w = max(2, int(rng.uniform(3, 7) * scale)); h = max(2, int(rng.uniform(3, 7) * scale))
    y, x = rng.integers(0, S - h), rng.integers(0, S - w)
    img[y:y + h, x:x + w] = CLOTHES[rng.integers(len(CLOTHES))] * rng.uniform(0.7, 1.0)


def _person(img, rng, scale):
    S = img.shape[0]
    body_h, body_w = max(3, int(round(8 * scale))), max(1, int(round(3 * scale)))
    sprite = np.zeros((body_h + 4, body_h + 4, 4))
    c = (body_h + 4) // 2
    col = CLOTHES[rng.integers(len(CLOTHES))]
    sprite[c - body_h // 2:c + body_h // 2, c - body_w // 2:c - body_w // 2 + body_w, :3] = col
    sprite[c - body_h // 2:c + body_h // 2, c - body_w // 2:c - body_w // 2 + body_w, 3] = 1
    hr = max(1, int(round(1.3 * scale)))
    sprite[c - body_h // 2 - hr:c - body_h // 2 + 1, c - hr // 2:c - hr // 2 + hr, :3] = [0.85, 0.65, 0.5]
    sprite[c - body_h // 2 - hr:c - body_h // 2 + 1, c - hr // 2:c - hr // 2 + hr, 3] = 1
    sprite = rotate(sprite, rng.uniform(0, 360), reshape=False, order=1)
    a = np.clip(sprite[..., 3:], 0, 1)
    n = sprite.shape[0]
    hi = max(S - n - 1, 2)
    y, x = rng.integers(1, hi), rng.integers(1, hi)
    img[y:y + n, x:x + n] = img[y:y + n, x:x + n] * (1 - a) + sprite[..., :3] * a


def render(rng, person, altitude=30.0):
    """altitude in metres (10 .. 80). Reference resolution at 30 m."""
    scale = np.clip(30.0 / altitude, 0.3, 1.8)
    img = _background(rng, PATCH)
    if rng.random() < 0.35:
        _debris(img, rng, scale)
    if person:
        _person(img, rng, scale)
    blur = 0.3 + altitude / 80
    img = gaussian_filter(img, (blur, blur, 0))
    img = img + rng.normal(0, 0.015 + altitude / 4000, img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


def make_dataset(n, rng, altitudes=(10, 80)):
    y = rng.random(n) < 0.5
    alt = rng.uniform(*altitudes, n)
    X = np.stack([render(rng, bool(y[i]), alt[i]) for i in range(n)])
    return X.transpose(0, 3, 1, 2), y.astype(np.float32), alt


class DetectorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def train_detector(n_train=12000, epochs=8, seed=0, verbose=True):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    X, y, _ = make_dataset(n_train, rng)
    Xv, yv, _ = make_dataset(2000, np.random.default_rng(seed + 1))
    X, y, Xv, yv = map(torch.tensor, (X, y, Xv, yv))
    m = DetectorCNN(); opt = torch.optim.Adam(m.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = []
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(X)); tot = 0
        for s in range(0, len(X), 128):
            i = perm[s:s + 128]
            xb = X[i]
            if torch.rand(1) < 0.5: xb = xb.flip(-1)                      # augmentation
            if torch.rand(1) < 0.5: xb = xb.flip(-2)
            loss = nn.functional.binary_cross_entropy_with_logits(m(xb), y[i])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(i)
        sched.step()
        m.eval()
        with torch.no_grad():
            acc = ((m(Xv) > 0).float() == yv).float().mean().item()
        hist.append(dict(epoch=ep + 1, loss=tot / len(X), val_acc=acc))
        if verbose:
            print(f"epoch {ep + 1} | loss {tot / len(X):.4f} | val acc {acc:.3f}")
    return m, hist


@torch.no_grad()
def performance_vs_altitude(model, altitudes=(10, 20, 30, 40, 50, 60, 70, 80), n=600, seed=5, thr=0.5):
    model.eval(); rows = []
    for a in altitudes:
        rng = np.random.default_rng(seed + int(a))
        X, y, _ = make_dataset(n, rng, altitudes=(a, a))
        p = torch.sigmoid(model(torch.tensor(X))).numpy()
        pred = p > thr
        tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = (~pred & (y == 1)).sum(); tn = (~pred & (y == 0)).sum()
        rows.append(dict(altitude_m=a, accuracy=(tp + tn) / n, recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1),
                         false_pos_rate=fp / max(fp + tn, 1)))
    return rows


def roc_curve(model, n=2000, seed=11):
    model.eval()
    X, y, _ = make_dataset(n, np.random.default_rng(seed))
    with torch.no_grad():
        s = torch.sigmoid(model(torch.tensor(X))).numpy()
    order = np.argsort(-s); y = y[order]
    tpr = np.r_[0, np.cumsum(y) / y.sum()]; fpr = np.r_[0, np.cumsum(1 - y) / (1 - y).sum()]
    return fpr, tpr, float(np.trapezoid(tpr, fpr) if hasattr(np, 'trapezoid') else np.trapz(tpr, fpr))


def search_tradeoff(rows, fov_deg=70.0, area_side=300.0, speed=8.0):
    """Time to sweep an area once vs altitude, and expected detections per sweep."""
    out = []
    for r in rows:
        swath = 2 * r['altitude_m'] * np.tan(np.radians(fov_deg / 2))
        t = area_side * area_side / (swath * speed)
        out.append(dict(altitude_m=r['altitude_m'], swath_m=swath, sweep_time_s=t, recall=r['recall'],
                        detections_per_min=r['recall'] / (t / 60)))
    return out
