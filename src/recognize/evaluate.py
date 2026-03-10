from __future__ import annotations

import torch
from loguru import logger
from pydantic import BaseModel, Field
from rich.table import Table
from torch.utils.data import DataLoader

from .dataset import MultimodalDataset
from .model import ClassifierModel

EPSILON = 1e-10


def confusion_matrix(model: ClassifierModel, data_loader: DataLoader) -> list[list[int]]:
    y_true, y_pred = get_outputs(model, data_loader)

    num_classes = model.num_classes
    if num_classes == 1:
        num_classes = 2
    matrix = [[0] * num_classes for _ in range(num_classes)]

    for true_label, pred_label in zip(y_true, y_pred, strict=True):
        matrix[true_label][pred_label] += 1

    return matrix


def calculate_overall_accuracy(conf_matrix: list[list[int]]) -> float:
    num_classes = len(conf_matrix)
    total = sum(sum(row) for row in conf_matrix)
    correct = sum(conf_matrix[i][i] for i in range(num_classes))
    return calculate_percentage(correct, total)


def calculate_percentage(numerator: float, denominator: float) -> float:
    return 100 * numerator / (denominator + EPSILON)


class ClassMetrics(BaseModel):
    accuracy: float = 0
    precision: float = 0
    recall: float = 0
    f1_score: float = 0


def calculate_class_metrics(conf_matrix: list[list[int]]) -> list[ClassMetrics]:
    num_classes = len(conf_matrix)
    class_metrics = []

    for i in range(num_classes):
        true_positives = conf_matrix[i][i]
        false_positives = sum(conf_matrix[j][i] for j in range(num_classes) if j != i)
        false_negatives = sum(conf_matrix[i][j] for j in range(num_classes) if j != i)
        total = sum(conf_matrix[i][j] for j in range(num_classes))

        precision = true_positives / (true_positives + false_positives + EPSILON)
        recall = true_positives / (true_positives + false_negatives + EPSILON)
        f1_score = 2 * (precision * recall) / (precision + recall + EPSILON)

        class_metrics.append(
            ClassMetrics(
                accuracy=calculate_percentage(true_positives, total),
                precision=100 * precision,
                recall=100 * recall,
                f1_score=100 * f1_score,
            )
        )

    return class_metrics


def calculate_weighted_metric(metric_scores: list[float], class_weights: list[float]) -> float:
    return sum(score * weight for score, weight in zip(metric_scores, class_weights, strict=True))


def calculate_weighted_f1_score(conf_matrix: list[list[int]], class_weights: list[float]) -> float:
    class_metrics = calculate_class_metrics(conf_matrix)
    class_f1_scores = [metric.f1_score / 100 for metric in class_metrics]
    return 100 * calculate_weighted_metric(class_f1_scores, class_weights)


def calculate_weighted_precision_score(conf_matrix: list[list[int]], class_weights: list[float]) -> float:
    class_metrics = calculate_class_metrics(conf_matrix)
    class_precision_scores = [metric.precision / 100 for metric in class_metrics]
    return 100 * calculate_weighted_metric(class_precision_scores, class_weights)


def calculate_weighted_recall_score(conf_matrix: list[list[int]], class_weights: list[float]) -> float:
    class_metrics = calculate_class_metrics(conf_matrix)
    class_recall_scores = [metric.recall / 100 for metric in class_metrics]
    return 100 * calculate_weighted_metric(class_recall_scores, class_weights)


def calculate_class_accuracies(conf_matrix: list[list[int]]) -> list[float]:
    return [metric.accuracy for metric in calculate_class_metrics(conf_matrix)]


def calculate_class_precision_scores(conf_matrix: list[list[int]]) -> list[float]:
    return [metric.precision for metric in calculate_class_metrics(conf_matrix)]


def calculate_class_recall_scores(conf_matrix: list[list[int]]) -> list[float]:
    return [metric.recall for metric in calculate_class_metrics(conf_matrix)]


def calculate_class_f1_scores(conf_matrix: list[list[int]]) -> list[float]:
    return [metric.f1_score for metric in calculate_class_metrics(conf_matrix)]


def get_outputs(model: ClassifierModel, data_loader: DataLoader) -> tuple[list[int], list[int]]:
    model.eval()
    predicted_list: list[int] = []
    labels_list: list[int] = []
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch)
            if outputs.logits.shape[1] == 1:
                predicted = (outputs.logits > 0).int().view(-1)
                labels = (batch.labels > 0).int()
            else:
                _, predicted = torch.max(outputs.logits, 1)
                labels = batch.labels
            predicted_list.extend(predicted.cpu().numpy().tolist())
            labels_list.extend(labels.cpu().numpy().tolist())
    return predicted_list, labels_list


class MetricsSummary(BaseModel):
    overall_accuracy: float = 0
    weighted_precision_score: float = 0
    weighted_recall_score: float = 0
    weighted_f1_score: float = 0
    class_accuracies: list[float] = Field(default_factory=list)
    class_precision_scores: list[float] = Field(default_factory=list)
    class_recall_scores: list[float] = Field(default_factory=list)
    class_f1_scores: list[float] = Field(default_factory=list)


def calculate_metrics_summary(conf_matrix: list[list[int]], class_weights: list[float]) -> MetricsSummary:
    class_metrics = calculate_class_metrics(conf_matrix)
    class_accuracies = [metric.accuracy for metric in class_metrics]
    class_precision_scores = [metric.precision for metric in class_metrics]
    class_recall_scores = [metric.recall for metric in class_metrics]
    class_f1_scores = [metric.f1_score for metric in class_metrics]

    return MetricsSummary(
        overall_accuracy=calculate_overall_accuracy(conf_matrix),
        weighted_precision_score=calculate_weighted_precision_score(conf_matrix, class_weights),
        weighted_recall_score=calculate_weighted_recall_score(conf_matrix, class_weights),
        weighted_f1_score=calculate_weighted_f1_score(conf_matrix, class_weights),
        class_accuracies=class_accuracies,
        class_precision_scores=class_precision_scores,
        class_recall_scores=class_recall_scores,
        class_f1_scores=class_f1_scores,
    )


class TrainingResult(MetricsSummary):
    confusion_matrix: list[list[int]] | None = None

    def print(self, *, print_table: bool = False):
        logger.info(
            "Overall Accuracy: "
            f"{self.overall_accuracy:.2f}%, "
            f"Weighted Precision: {self.weighted_precision_score:.2f}%, "
            f"Weighted Recall: {self.weighted_recall_score:.2f}%, "
            f"Weighted F1 Score: {self.weighted_f1_score:.2f}%"
        )
        if len(self.class_accuracies) > 0:
            for i, (acc, precision, recall, f1) in enumerate(
                zip(
                    self.class_accuracies,
                    self.class_precision_scores,
                    self.class_recall_scores,
                    self.class_f1_scores,
                    strict=True,
                )
            ):
                logger.info(
                    f"Class {i} - Accuracy: {acc:.2f}%, "
                    f"Precision: {precision:.2f}%, "
                    f"Recall: {recall:.2f}%, "
                    f"F1 Score: {f1:.2f}%"
                )
        if not print_table or self.confusion_matrix is None:
            return

        from rich import print

        logger.info("Confusion Matrix:")
        table = Table(show_header=False, show_lines=True)
        for i, row in enumerate(self.confusion_matrix):
            str_row = [f"[red]{v}" if i == j else f"{v}" for j, v in enumerate(row)]
            table.add_row(*str_row)
        print(table)

    def gen_typst_code(
        self,
        gen_accuracy: bool = True,
        gen_precision: bool = False,
        gen_recall: bool = False,
        gen_f1: bool = True,
    ) -> str:
        items = []
        for acc, precision, recall, f1 in zip(
            self.class_accuracies,
            self.class_precision_scores,
            self.class_recall_scores,
            self.class_f1_scores,
            strict=True,
        ):
            if gen_accuracy:
                items.append(f"[{acc:.2f}]")
            if gen_precision:
                items.append(f"[{precision:.2f}]")
            if gen_recall:
                items.append(f"[{recall:.2f}]")
            if gen_f1:
                items.append(f"[{f1:.2f}]")
        return ", ".join(items)

    @classmethod
    def auto_compute(cls, model: ClassifierModel, data_loader: DataLoader, *, output_path: str | None = None):
        dataset = data_loader.dataset
        assert isinstance(dataset, MultimodalDataset)
        class_weights = dataset.class_weights

        conf_matrix = confusion_matrix(model, data_loader)
        metrics_summary = calculate_metrics_summary(conf_matrix, class_weights)

        from utils.visualization import plot_confusion_matrix

        if output_path is not None:
            plot_confusion_matrix(
                confusion_matrix=conf_matrix,
                class_names=list(dataset.emotion_class_names_mapping.keys()),
                output_path=output_path,
                normalize=True,
            )
        return cls(
            **metrics_summary.model_dump(),
            confusion_matrix=conf_matrix,
        )
