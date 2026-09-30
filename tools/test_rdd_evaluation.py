"""Small regression tests for matching correctness and CPU/CUDA conversion."""
import unittest

import numpy as np
import torch

from evaluate_rdd_examples import box_iou, match_predictions
from infer import to_numpy


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.gt = [dict(id=1, category_id=0, bbox=[0, 0, 10, 10])]

    def test_duplicate_is_false_positive_and_highest_score_wins(self):
        result = match_predictions([0, 0], [[0, 0, 10, 10]] * 2,
                                   [0.7, 0.9], self.gt, 0.6, 0.5)
        self.assertEqual([r['true_positive'] for r in result], [True, False])
        self.assertEqual(result[0]['score'], 0.9)

    def test_wrong_class_and_low_overlap_are_false_positives(self):
        result = match_predictions([1, 0], [[0, 0, 10, 10], [20, 20, 30, 30]],
                                   [0.9, 0.8], self.gt, 0.6, 0.5)
        self.assertFalse(any(r['true_positive'] for r in result))

    def test_threshold_boundaries_and_below_threshold(self):
        result = match_predictions([0, 0], [[0, 0, 5, 10]] * 2,
                                   [0.6, 0.59], self.gt, 0.6, 0.5)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]['true_positive'])
        self.assertEqual(result[0]['iou'], 0.5)

    def test_empty_ground_truth_and_predictions(self):
        result = match_predictions([0], [[0, 0, 10, 10]], [0.9], [], 0.6, 0.5)
        self.assertFalse(result[0]['true_positive'])
        self.assertEqual(match_predictions([], [], [], self.gt, 0.6, 0.5), [])
        self.assertEqual(box_iou([0, 0, 0, 0], [0, 0, 0, 0]), 0)

    def test_each_ground_truth_can_be_matched_once(self):
        gt = self.gt + [dict(id=2, category_id=0, bbox=[20, 20, 10, 10])]
        result = match_predictions([0, 0], [[0, 0, 10, 10], [20, 20, 30, 30]],
                                   [0.9, 0.8], gt, 0.6, 0.5)
        self.assertEqual([r['annotation_id'] for r in result], [1, 2])

    def test_numpy_and_tensor_conversion(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            value = torch.tensor([1.0], device=device, requires_grad=True)
            np.testing.assert_array_equal(to_numpy(value), [1.0])
        np.testing.assert_array_equal(to_numpy(np.array([1.])), [1.])


if __name__ == '__main__':
    unittest.main()
