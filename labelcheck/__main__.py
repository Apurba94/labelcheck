"""Audit labels from a .npz file holding out-of-fold `probs` and `labels`.

Usage: python -m labelcheck results.npz [how many to list]
"""
import sys

import numpy as np

from . import find_issues, noise_rate, quality_scores


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    d = np.load(argv[1])
    probs, labels = d["probs"], d["labels"]
    show = int(argv[2]) if len(argv) > 2 else 20
    issues = find_issues(probs, labels)
    print(f"{len(labels)} examples, estimated label noise {noise_rate(probs, labels):.1%}, {len(issues)} suspects")
    q = quality_scores(probs, labels)
    for i in issues[:show]:
        print(f"  #{i}: labelled {labels[i]}, the model says {probs[i].argmax()} (label confidence {q[i]:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
