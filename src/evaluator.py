import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from src.config import Config
from src.utils import ensure_dir


class Evaluator:
    """Evaluates a trained model and produces diagnostic outputs.

    Computes test loss/accuracy, generates a confusion matrix plot, and
    saves a full classification report.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        ensure_dir(config.OUTPUT_PATH)

    def evaluate(
        self,
        model: tf.keras.Model,
        test_ds: tf.data.Dataset,
        class_names: List[str],
    ) -> Dict[str, float]:
        """Evaluate the model on the test set and produce diagnostic outputs.

        Args:
            model: The trained Keras model.
            test_ds: Test dataset.
            class_names: List of class label names.

        Returns:
            A dictionary containing 'loss' and 'accuracy' metrics.
        """
        loss, accuracy = model.evaluate(test_ds, verbose=1)
        print(f"\nTest Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        y_true, y_pred = self._get_predictions(model, test_ds)

        self._plot_confusion_matrix(y_true, y_pred, class_names)
        self._save_classification_report(y_true, y_pred, class_names)

        return {"loss": loss, "accuracy": accuracy}

    def _get_predictions(
        self,
        model: tf.keras.Model,
        test_ds: tf.data.Dataset,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute true labels and predicted labels for the test set.

        Args:
            model: The trained Keras model.
            test_ds: Test dataset.

        Returns:
            A tuple of (y_true, y_pred) numpy arrays.
        """
        y_true: List[int] = []
        y_pred: List[int] = []
        for images, labels in test_ds:
            preds = model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(preds, axis=1))
        return np.array(y_true), np.array(y_pred)

    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
    ) -> None:
        """Plot and save a confusion matrix heatmap.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            class_names: List of class label names for axis ticks.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.OUTPUT_PATH, "confusion_matrix.png"))
        plt.close()

    def _save_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
    ) -> None:
        """Generate and save a text classification report.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            class_names: List of class label names.
        """
        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
        )
        report_path = os.path.join(
            self.config.OUTPUT_PATH,
            "classification_report.txt",
        )
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nClassification Report:\n{report}")
