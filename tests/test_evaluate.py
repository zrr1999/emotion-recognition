from __future__ import annotations

import unittest

from recognize.evaluate import (
    TrainingResult,
    calculate_class_accuracies,
    calculate_class_f1_scores,
    calculate_class_precision_scores,
    calculate_class_recall_scores,
    calculate_metrics_summary,
    calculate_overall_accuracy,
    calculate_weighted_f1_score,
    calculate_weighted_precision_score,
    calculate_weighted_recall_score,
)


class EvaluateMetricsTest(unittest.TestCase):
    def setUp(self):
        self.conf_matrix = [
            [8, 2],
            [1, 9],
        ]
        self.class_weights = [0.4, 0.6]

    def test_overall_and_weighted_metrics(self):
        self.assertAlmostEqual(calculate_overall_accuracy(self.conf_matrix), 85.0)
        self.assertAlmostEqual(calculate_weighted_precision_score(self.conf_matrix, self.class_weights), 84.64646465)
        self.assertAlmostEqual(calculate_weighted_recall_score(self.conf_matrix, self.class_weights), 86.0)
        self.assertAlmostEqual(calculate_weighted_f1_score(self.conf_matrix, self.class_weights), 85.11278195)

    def test_class_metrics(self):
        self.assertEqual(len(calculate_class_accuracies(self.conf_matrix)), 2)
        self.assertAlmostEqual(calculate_class_accuracies(self.conf_matrix)[0], 80.0)
        self.assertAlmostEqual(calculate_class_accuracies(self.conf_matrix)[1], 90.0)
        self.assertAlmostEqual(calculate_class_precision_scores(self.conf_matrix)[0], 88.88888889)
        self.assertAlmostEqual(calculate_class_precision_scores(self.conf_matrix)[1], 81.81818182)
        self.assertAlmostEqual(calculate_class_recall_scores(self.conf_matrix)[0], 80.0)
        self.assertAlmostEqual(calculate_class_recall_scores(self.conf_matrix)[1], 90.0)
        self.assertAlmostEqual(calculate_class_f1_scores(self.conf_matrix)[0], 84.21052632)
        self.assertAlmostEqual(calculate_class_f1_scores(self.conf_matrix)[1], 85.71428571)

    def test_metrics_summary_and_typst_output(self):
        metrics_summary = calculate_metrics_summary(self.conf_matrix, self.class_weights)
        self.assertAlmostEqual(metrics_summary.weighted_precision_score, 84.64646465)
        self.assertAlmostEqual(metrics_summary.weighted_recall_score, 86.0)

        result = TrainingResult(**metrics_summary.model_dump(), confusion_matrix=self.conf_matrix)
        self.assertEqual(
            result.gen_typst_code(gen_accuracy=False, gen_precision=True, gen_recall=True, gen_f1=True),
            "[88.89], [80.00], [84.21], [81.82], [90.00], [85.71]",
        )


if __name__ == "__main__":
    unittest.main()
