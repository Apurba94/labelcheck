import random
import unittest

import numpy as np

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    cohen_kappa_score = None

from labelcheck import BKTree, average_hash, box_agreement, class_thresholds, cohen_kappa, confident_joint, cross_val_probs, difference_hash, embedding_duplicates, exact_duplicates, find_issues, fleiss_kappa, hamming, krippendorff_alpha, near_duplicates, noise_matrix, noise_rate, quality_scores


def noisy_data(n=6000, k=4, noise=0.15, seed=0, sharp=4.0):
    """A model with honest probabilities on true labels, and labels corrupted at a known rate."""
    rng = np.random.default_rng(seed)
    true = rng.integers(0, k, n)
    logits = rng.normal(size=(n, k))
    logits[np.arange(n), true] += sharp
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    given = true.copy()
    flip = rng.random(n) < noise
    given[flip] = (true[flip] + rng.integers(1, k, flip.sum())) % k
    return p, given, true, flip


class TestConfidentLearning(unittest.TestCase):
    def test_thresholds(self):
        p = np.array([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]])
        self.assertTrue(np.allclose(class_thresholds(p, [0, 0, 1]), [0.7, 0.9]))

    def test_confident_joint_counts(self):
        # thresholds are 0.85 and 0.525, so the second row qualifies for neither class
        p = np.array([[0.9, 0.1], [0.8, 0.2], [0.05, 0.95], [0.9, 0.1]])
        labels = np.array([0, 0, 1, 1])
        self.assertEqual(confident_joint(p, labels).tolist(), [[1, 0], [1, 1]])

    def test_finds_most_flips_with_few_false_alarms(self):
        p, given, _, flip = noisy_data()
        for method in ("prune", "off_diagonal"):
            found = np.zeros(len(given), bool)
            found[find_issues(p, given, method)] = True
            recall = (found & flip).sum() / flip.sum()
            precision = (found & flip).sum() / found.sum()
            self.assertGreater(recall, 0.8, method)
            self.assertGreater(precision, 0.8, method)

    def test_clean_data_gives_few_suspects(self):
        p, given, _, _ = noisy_data(noise=0.0, sharp=6)
        self.assertLess(len(find_issues(p, given)), 0.02 * len(given))

    def test_noise_rate_is_close(self):
        for rate in (0.05, 0.2, 0.3):
            p, given, _, _ = noisy_data(n=10000, noise=rate, seed=1)
            self.assertAlmostEqual(noise_rate(p, given), rate, delta=0.04)

    def test_noise_matrix_finds_a_systematic_swap(self):
        p, given, true, _ = noisy_data(noise=0.0, seed=2, sharp=5)
        rng = np.random.default_rng(3)
        swap = (true == 1) & (rng.random(len(true)) < 0.3)
        given[swap] = 2
        m = noise_matrix(p, given)
        self.assertAlmostEqual(m[2, 1], 0.3, delta=0.05)
        self.assertLess(m[3, 0], 0.05)

    def test_most_suspicious_come_first(self):
        p, given, _, flip = noisy_data()
        top = find_issues(p, given)[:100]
        self.assertGreater(flip[top].mean(), 0.95)

    def test_quality_scores(self):
        self.assertEqual(quality_scores(np.array([[0.2, 0.8]]), [1]).tolist(), [0.8])

    def test_bad_method(self):
        with self.assertRaises(ValueError):
            find_issues(np.ones((2, 2)) / 2, [0, 1], "nope")

    def test_out_of_fold_catches_what_a_memorising_model_would_miss(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(800, 2)) + np.repeat(np.array([[3, 0], [-3, 0]]), 400, axis=0)
        labels = np.repeat([0, 1], 400)
        wrong = rng.choice(800, 40, replace=False)
        labels[wrong] = 1 - labels[wrong]

        def nearest_centroid(xtr, ytr, xte):
            c = np.stack([xtr[ytr == k].mean(axis=0) for k in (0, 1)])
            d = -((xte[:, None] - c[None]) ** 2).sum(axis=2)
            e = np.exp(d - d.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        p = cross_val_probs(nearest_centroid, x, labels)
        found = set(find_issues(p, labels))
        self.assertGreater(len(found & set(wrong.tolist())), 32)


class TestHashes(unittest.TestCase):
    def image(self, seed):
        rng = np.random.default_rng(seed)
        base = rng.random((6, 6))
        return np.kron(base, np.ones((16, 16))) * 255

    def test_same_image_same_hash(self):
        self.assertEqual(average_hash(self.image(1)), average_hash(self.image(1)))

    def test_brightness_noise_and_resize_stay_close(self):
        a = self.image(1)
        rng = np.random.default_rng(9)
        for fn in (difference_hash, average_hash):
            self.assertLessEqual(hamming(fn(a), fn(np.clip(a * 0.7 + 30, 0, 255))), 6)
            self.assertLessEqual(hamming(fn(a), fn(a + rng.normal(0, 5, a.shape))), 6)
            self.assertLessEqual(hamming(fn(a), fn(a[::2, ::2])), 6)

    def test_different_images_are_far(self):
        self.assertGreater(hamming(difference_hash(self.image(1)), difference_hash(self.image(2))), 15)

    def test_colour_images_work(self):
        self.assertIsInstance(difference_hash(np.stack([self.image(1)] * 3, axis=2)), int)

    def test_bk_tree_matches_a_full_scan_and_does_less_work(self):
        rng = random.Random(1)
        values = list({rng.getrandbits(64) for _ in range(3000)})
        t = BKTree()
        for i, v in enumerate(values):
            t.add(v, i)
        for _ in range(30):
            q = values[rng.randrange(len(values))] ^ rng.getrandbits(4)
            want = sorted(i for i, v in enumerate(values) if hamming(v, q) <= 12)
            self.assertEqual(sorted(t.search(q, 12)), want)
        self.assertLess(t.comparisons, 30 * len(values))

    def test_groups(self):
        h = {"a": 0b0000, "b": 0b0001, "c": 0b0011, "d": 0xFFFF00}
        self.assertEqual(near_duplicates(h, 1), [["a", "b", "c"]])

    def test_exact(self):
        self.assertEqual(exact_duplicates({"x": b"1", "y": b"2", "z": b"1"}), [["x", "z"]])

    def test_embeddings(self):
        v = np.array([[1, 0, 0], [0.999, 0.01, 0], [0, 1, 0]])
        self.assertEqual([t[:2] for t in embedding_duplicates(v, 0.99)], [(0, 1)])


class TestAgreement(unittest.TestCase):
    def test_kappa_values(self):
        self.assertEqual(cohen_kappa([1, 2, 3], [1, 2, 3]), 1)
        self.assertAlmostEqual(cohen_kappa([0, 0, 1, 1], [0, 1, 0, 1]), 0)
        self.assertAlmostEqual(cohen_kappa(["a"] * 20 + ["b"] * 5 + ["a"] * 10 + ["b"] * 15, ["a"] * 20 + ["a"] * 5 + ["b"] * 10 + ["b"] * 15), 0.4)

    @unittest.skipIf(cohen_kappa_score is None, "scikit-learn is not installed")
    def test_kappa_matches_scikit_learn(self):
        rng = random.Random(2)
        for _ in range(100):
            a = [rng.randint(0, 3) for _ in range(40)]
            b = [x if rng.random() < 0.6 else rng.randint(0, 3) for x in a]
            for w in (None, "linear", "quadratic"):
                self.assertAlmostEqual(cohen_kappa(a, b, w), cohen_kappa_score(a, b, weights=w))

    def test_fleiss_published_example(self):
        table = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
        self.assertAlmostEqual(fleiss_kappa(table), 0.2099, places=3)

    def test_fleiss_needs_equal_raters(self):
        with self.assertRaises(ValueError):
            fleiss_kappa([[1, 1], [1, 0]])

    def test_krippendorff_published_example(self):
        # the reliability data from Krippendorff (2011), four observers with gaps; nominal alpha = 0.743
        n = None
        rows = [[1, 2, 3, 3, 2, 1, 4, 1, 2, n, n, n], [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, n, 3], [n, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, n], [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, n]]
        units = list(zip(*rows))
        self.assertAlmostEqual(krippendorff_alpha(units), 0.743, places=3)
        self.assertAlmostEqual(krippendorff_alpha(units, "interval"), 0.849, places=3)

    def test_krippendorff_edges(self):
        self.assertEqual(krippendorff_alpha([[1, 1], [2, 2]]), 1)
        self.assertTrue(np.isnan(krippendorff_alpha([[1, None]])))

    def test_boxes(self):
        a = [("cell", 0, 0, 10, 10), ("cell", 20, 0, 10, 10), ("wbc", 50, 0, 10, 10)]
        b = [("cell", 1, 0, 10, 10), ("rbc", 20, 0, 10, 10)]
        r = box_agreement(a, b)
        self.assertEqual((r["matched"], r["only_a"], r["only_b"], r["label_agreement"]), (2, 1, 0, 0.5))
        self.assertAlmostEqual(r["f1"], 0.8)

    def test_boxes_with_nothing(self):
        self.assertEqual(box_agreement([], [])["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
