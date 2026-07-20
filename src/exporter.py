import os
from typing import List

import tensorflow as tf

from src.config import Config
from src.utils import ensure_dir


class Exporter:
    def __init__(self, config: Config) -> None:
        self.config = config

    def export_all(
        self,
        model: tf.keras.Model,
        class_names: List[str],
    ) -> None:
        self.export_savedmodel(model)
        self.export_tflite(model, class_names)
        self.export_tfjs(model)
        self.save_labels(class_names)

    def export_savedmodel(self, model: tf.keras.Model) -> None:
        path = ensure_dir(self.config.MODEL_PATH)
        model.save(path)
        print(f"SavedModel exported to: {path}")

    def export_tflite(
        self,
        model: tf.keras.Model,
        class_names: List[str],
    ) -> None:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        path = ensure_dir(self.config.TFLITE_PATH)
        tflite_path = os.path.join(path, "model.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"TFLite exported to: {tflite_path}")

    def export_tfjs(self, model: tf.keras.Model) -> None:
        try:
            import tensorflowjs as tfjs
            path = ensure_dir(self.config.TFJS_PATH)
            tfjs.converters.save_keras_model(model, path)
            print(f"TFJS exported to: {path}")
        except ImportError:
            print("tensorflowjs not installed. Skipping TFJS export.")

    def save_labels(self, class_names: List[str]) -> None:
        labels_path = os.path.join(self.config.MODEL_PATH, "labels.txt")
        with open(labels_path, "w") as f:
            for name in class_names:
                f.write(f"{name}\n")
        print(f"Labels saved to: {labels_path}")
