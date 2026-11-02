"""Find problems in labelled data: wrong labels, duplicates and disagreeing annotators."""
import hashlib
from collections import Counter, defaultdict

import numpy as np


def class_thresholds(probs, labels):
    """t_j: the model's average confidence in class j over examples labelled j."""
    probs, labels = np.asarray(probs, float), np.asarray(labels, int)
    out = np.zeros(probs.shape[1])
    for c in range(probs.shape[1]):
        m = labels == c
        out[c] = probs[m, c].mean() if m.any() else 1.0
    return out


def confident_joint(probs, labels):
    """Count how often an example labelled i confidently looks like class j (Northcutt et al. 2021).
    Each example counts once, for the qualifying class with the highest probability."""
    probs, labels = np.asarray(probs, float), np.asarray(labels, int)
    t = class_thresholds(probs, labels)
    k = probs.shape[1]
    above = probs >= t - 1e-12
    masked = np.where(above, probs, -1.0)
    best = masked.argmax(axis=1)
    has = above.any(axis=1)
    cj = np.zeros((k, k), int)
    np.add.at(cj, (labels[has], best[has]), 1)
    return cj


def calibrated_joint(cj, labels, k):
    """Adjust counts so each row matches the true label counts, then normalise to a distribution."""
    counts = np.bincount(np.asarray(labels, int), minlength=k).astype(float)
    rows = cj.sum(axis=1, keepdims=True).astype(float)
    rows[rows == 0] = 1
    q = cj / rows * counts[:, None]
    return q / max(q.sum(), 1)


def noise_matrix(probs, labels):
    """P(given label = i | true label = j), estimated from the data."""
    probs, labels = np.asarray(probs, float), np.asarray(labels, int)
    k = probs.shape[1]
    q = calibrated_joint(confident_joint(probs, labels), labels, k)
    cols = q.sum(axis=0, keepdims=True)
    cols[cols == 0] = 1
    return q / cols


def noise_rate(probs, labels):
    k = np.asarray(probs).shape[1]
    q = calibrated_joint(confident_joint(probs, labels), labels, k)
    return float(1 - np.trace(q))


def find_issues(probs, labels, method="prune"):
    """Return suspect example indices, most suspicious first.

    prune: for each off-diagonal cell (i, j) of the joint, remove the n examples of label i
    with the largest gap p_j - p_i, where n comes from the calibrated joint.
    off_diagonal: everything the confident joint puts off the diagonal.
    Use out-of-fold probabilities, because a model that saw an example memorises its label.
    """
    probs, labels = np.asarray(probs, float), np.asarray(labels, int)
    n, k = probs.shape
    if method == "off_diagonal":
        t = class_thresholds(probs, labels)
        above = probs >= t - 1e-12
        best = np.where(above, probs, -1.0).argmax(axis=1)
        hit = above.any(axis=1) & (best != labels)
        idx = np.nonzero(hit)[0]
    elif method == "prune":
        q = calibrated_joint(confident_joint(probs, labels), labels, k)
        chosen = set()
        for i in range(k):
            members = np.nonzero(labels == i)[0]
            for j in range(k):
                count = int(round(q[i, j] * n))
                if i != j and count > 0 and len(members):
                    gap = probs[members, j] - probs[members, i]
                    chosen.update(members[np.argsort(-gap, kind="stable")[:count]].tolist())
        idx = np.array(sorted(chosen), int)
    else:
        raise ValueError("method must be prune or off_diagonal")
    margin = probs.max(axis=1) - probs[np.arange(n), labels]
    return idx[np.argsort(-margin[idx], kind="stable")].tolist()


def quality_scores(probs, labels):
    """Self-confidence: low means the label looks wrong."""
    return np.asarray(probs, float)[np.arange(len(labels)), np.asarray(labels, int)]


def cross_val_probs(fit_predict, x, labels, folds=5, seed=0):
    """Out-of-fold probabilities. fit_predict(x_train, y_train, x_test) returns N×C."""
    labels = np.asarray(labels, int)
    rng = np.random.default_rng(seed)
    fold = np.empty(len(labels), int)
    for c in np.unique(labels):
        idx = rng.permutation(np.nonzero(labels == c)[0])
        fold[idx] = np.arange(len(idx)) % folds
    out = None
    for f in range(folds):
        te = fold == f
        p = np.asarray(fit_predict(x[~te], labels[~te], x[te]), float)
        if out is None:
            out = np.zeros((len(labels), p.shape[1]))
        out[te] = p
    return out


def average_hash(image, size=8):
    """A perceptual hash: shrink, then compare each cell with the mean. `image` is a 2-D or 3-D array."""
    a = np.asarray(image, float)
    if a.ndim == 3:
        a = a.mean(axis=2)
    h, w = a.shape
    ys = np.linspace(0, h, size + 1).astype(int)
    xs = np.linspace(0, w, size + 1).astype(int)
    cells = np.array([[a[ys[i]:max(ys[i + 1], ys[i] + 1), xs[j]:max(xs[j + 1], xs[j] + 1)].mean() for j in range(size)] for i in range(size)])
    bits = (cells > cells.mean()).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def difference_hash(image, size=8):
    """dHash: compare neighbouring cells, which survives brightness changes better."""
    a = np.asarray(image, float)
    if a.ndim == 3:
        a = a.mean(axis=2)
    h, w = a.shape
    ys = np.linspace(0, h, size + 1).astype(int)
    xs = np.linspace(0, w, size + 2).astype(int)
    cells = np.array([[a[ys[i]:max(ys[i + 1], ys[i] + 1), xs[j]:max(xs[j + 1], xs[j] + 1)].mean() for j in range(size + 1)] for i in range(size)])
    bits = (cells[:, 1:] > cells[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return (a ^ b).bit_count()


class BKTree:
    """Find values within a Hamming distance without comparing against every item."""

    def __init__(self):
        self.root = None
        self.comparisons = 0

    def add(self, value, key):
        if self.root is None:
            self.root = [value, [key], {}]
            return
        node = self.root
        while True:
            d = hamming(value, node[0])
            if d == 0:
                node[1].append(key)
                return
            if d in node[2]:
                node = node[2][d]
            else:
                node[2][d] = [value, [key], {}]
                return

    def search(self, value, radius):
        out, todo = [], [self.root] if self.root else []
        while todo:
            node = todo.pop()
            self.comparisons += 1
            d = hamming(value, node[0])
            if d <= radius:
                out.extend(node[1])
            todo.extend(child for edge, child in node[2].items() if d - radius <= edge <= d + radius)
        return out


class Groups:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def near_duplicates(hashes, radius=5):
    """hashes: {name: hash}. Returns groups of near-identical items, largest first."""
    tree = BKTree()
    for k, h in hashes.items():
        tree.add(h, k)
    g = Groups()
    for k, h in hashes.items():
        for other in tree.search(h, radius):
            if other != k:
                g.union(k, other)
    groups = defaultdict(list)
    for k in hashes:
        groups[g.find(k)].append(k)
    return sorted((sorted(v) for v in groups.values() if len(v) > 1), key=lambda v: (-len(v), v))


def exact_duplicates(blobs):
    """blobs: {name: bytes}. Returns groups with identical content."""
    seen = defaultdict(list)
    for k, b in blobs.items():
        seen[hashlib.sha256(b).hexdigest()].append(k)
    return sorted((sorted(v) for v in seen.values() if len(v) > 1), key=lambda v: (-len(v), v))


def embedding_duplicates(vectors, threshold=0.98):
    """Pairs of rows with cosine similarity above a threshold, as (i, j, similarity)."""
    v = np.asarray(vectors, float)
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    out = []
    for start in range(0, len(v), 512):
        sims = v[start:start + 512] @ v.T
        for i, j in zip(*np.nonzero(sims >= threshold)):
            a = start + int(i)
            if a < int(j):
                out.append((a, int(j), float(sims[i, j])))
    return sorted(out, key=lambda t: -t[2])


def cohen_kappa(a, b, weights=None):
    """Agreement between two annotators beyond chance. weights: None, linear or quadratic."""
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    m = np.zeros((k, k))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    expected = np.outer(m.sum(axis=1), m.sum(axis=0)) / n
    if weights is None:
        w = 1 - np.eye(k)
    else:
        grid = np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
        w = grid if weights == "linear" else grid ** 2
    denom = (w * expected).sum()
    return 1.0 if denom == 0 else float(1 - (w * m).sum() / denom)


def fleiss_kappa(table):
    """table: subjects × categories, each cell the number of raters who chose it."""
    t = np.asarray(table, float)
    n = t.sum(axis=1)[0]
    if not np.allclose(t.sum(axis=1), n):
        raise ValueError("every subject needs the same number of ratings")
    N = len(t)
    p = t.sum(axis=0) / (N * n)
    P = ((t ** 2).sum(axis=1) - n) / (n * (n - 1))
    pe = (p ** 2).sum()
    return 1.0 if pe == 1 else float((P.mean() - pe) / (1 - pe))


def krippendorff_alpha(units, kind="nominal"):
    """units: one list per item of the labels given, with None for missing. Handles any number of raters."""
    pairs = Counter()
    for values in units:
        vals = [v for v in values if v is not None]
        m = len(vals)
        if m < 2:
            continue
        for i, a in enumerate(vals):
            for j, b in enumerate(vals):
                if i != j:
                    pairs[(a, b)] += 1 / (m - 1)
    totals = Counter()
    for (a, _), w in pairs.items():
        totals[a] += w
    n = sum(totals.values())
    if n == 0:
        return float("nan")

    def distance(a, b):
        if kind == "nominal":
            return 0 if a == b else 1
        if kind == "interval":
            return (a - b) ** 2
        if kind == "ordinal":
            lo, hi = min(a, b), max(a, b)
            s = sum(totals[v] for v in totals if lo <= v <= hi) - (totals[lo] + totals[hi]) / 2
            return s ** 2
        raise ValueError("kind must be nominal, ordinal or interval")
    observed = sum(w * distance(a, b) for (a, b), w in pairs.items())
    expected = sum(totals[a] * totals[b] * distance(a, b) for a in totals for b in totals) / (n - 1)
    return 1.0 if expected == 0 else float(1 - observed / expected)


def box_iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    w = max(0, min(ax2, bx2) - max(a[0], b[0]))
    h = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = w * h
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def box_agreement(a, b, threshold=0.5):
    """Compare two annotators' boxes on one image. a and b: lists of (label, x, y, w, h)."""
    pairs = sorted(((box_iou(p[1:], q[1:]), i, j) for i, p in enumerate(a) for j, q in enumerate(b)), reverse=True)
    used_a, used_b, matched = set(), set(), []
    for v, i, j in pairs:
        if v >= threshold and i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            matched.append((v, i, j))
    same = sum(1 for _, i, j in matched if a[i][0] == b[j][0])
    union = len(a) + len(b) - len(matched)
    return {"matched": len(matched), "only_a": len(a) - len(matched), "only_b": len(b) - len(matched), "label_agreement": same / len(matched) if matched else None, "mean_iou": float(np.mean([m[0] for m in matched])) if matched else None, "f1": 2 * len(matched) / (len(a) + len(b)) if a or b else 1.0, "jaccard": len(matched) / union if union else 1.0}
