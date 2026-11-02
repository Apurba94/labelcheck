# LabelCheck

Bad labels quietly limit every model and every paper. LabelCheck finds them: likely mislabelled examples, duplicate images that leak between splits, and annotators who disagree.

```python
from labelcheck import cross_val_probs, find_issues, noise_rate

probs = cross_val_probs(my_fit_predict, x, labels)     # out-of-fold, so the model has not memorised the labels
print(f"estimated label noise: {noise_rate(probs, labels):.1%}")
for i in find_issues(probs, labels)[:50]:
    print(i, labels[i], probs[i].argmax())
```

## What it does

**Label errors (confident learning, Northcutt et al. 2021).** Per-class thresholds make the method robust to models that are too confident or too shy. It estimates the joint distribution of given and true labels, ranks suspects, and gives a noise matrix that shows which classes get mixed up.

**Duplicates.** Exact duplicates by SHA-256, near-duplicates by perceptual hashes (aHash and dHash) with a BK-tree search, and embedding duplicates by cosine similarity. A near-duplicate across train and test inflates your results without warning.

**Annotator agreement.**

| Measure | Use |
|---|---|
| Cohen's kappa (with linear or quadratic weights) | Two annotators |
| Fleiss' kappa | Many annotators, each rating every item |
| Krippendorff's alpha | Any number of annotators, with missing ratings; nominal, ordinal or interval data |
| `box_agreement` | Detection boxes: matches, IoU, F1 and label agreement |

## How it is tested

- On simulated data with known label flips, both methods must reach at least 80% recall and precision, and the noise-rate estimate must land near the truth at 5%, 20% and 30%.
- Cohen's kappa matches scikit-learn on 100 random cases, when scikit-learn is installed.
- Fleiss' kappa reproduces the widely used 14-rater worked example (0.210), and Krippendorff's alpha reproduces the published four-observer data (0.743 nominal, 0.849 interval).
- The BK-tree search returns exactly what a full scan returns, with fewer comparisons.

## Limits

Always review suspects by hand: a flagged image may be a hard case and not an error. Results depend on the quality of the model that gives the probabilities, and an in-sample model will hide errors, so use `cross_val_probs`. Perceptual hashes catch resizes and brightness changes, but not crops or rotations.

## Tests

```
pip install -r requirements.txt
python -m unittest discover -s tests
```

MIT licence.
