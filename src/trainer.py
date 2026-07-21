import os
from typing import List, Optional, Tuple

import tensorflow as tf

from src.config import Config
from src.callbacks import CallbackFactory


class Trainer:
    """Trains a Keras model using the provided datasets and callbacks."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def _get_last_epoch(self) -> int:
        csv_path = os.path.join(self.config.LOG_PATH, "training_log.csv")
        if not os.path.isfile(csv_path):
            return 0
        with open(csv_path) as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) < 2:
            return 0
        last_line = lines[-1]
        try:
            return int(last_line.split(",")[0]) + 1
        except (ValueError, IndexError):
            return 0

    def train(
        self,
        model: tf.keras.Model,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    ) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
        """Run model training with automatic resume support.

        If a checkpoint exists in checkpoints/latest_checkpoint.keras,
        the model is restored and training continues from that epoch.

        Args:
            model: The Keras model to train.
            train_ds: Training dataset.
            val_ds: Validation dataset.
            callbacks: Optional list of Keras callbacks. If None, default
                callbacks are created via CallbackFactory.

        Returns:
            A tuple of (model, history) where model is the trained/resumed model
            and history is the History object containing training metrics per epoch.
        """
        if callbacks is None:
            factory = CallbackFactory(self.config)
            callbacks = factory.get_callbacks()

        resume_path = os.path.join(self.config.CHECKPOINT_PATH, "latest_checkpoint.keras")
        initial_epoch = self._get_last_epoch()

        # Fallback to legacy path config.MODEL_PATH if not found in CHECKPOINT_PATH
        if not os.path.isfile(resume_path):
            fallback_path = os.path.join(self.config.MODEL_PATH, "latest_checkpoint.keras")
            if os.path.isfile(fallback_path):
                print(f"Latest checkpoint not found in {self.config.CHECKPOINT_PATH}. Using fallback from {self.config.MODEL_PATH}...")
                resume_path = fallback_path

        if os.path.isfile(resume_path) and initial_epoch > 0:
            print(f"Resuming training from epoch {initial_epoch}...")
            model = tf.keras.models.load_model(resume_path)
        else:
            print("Starting training from scratch...")

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.EPOCHS,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=1,
        )
        return model, history
