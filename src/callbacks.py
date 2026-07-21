import os
from typing import List

import tensorflow as tf

from src.config import Config


class CallbackFactory:
    """Creates a list of Keras callbacks for training.

    Includes early stopping, learning rate reduction on plateau, model
    checkpointing, NaN termination, backup/restore, and CSV logging.
    """

    def __init__(self, config: Config, log_dir: str = "logs") -> None:
        self.config = config
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(config.MODEL_PATH, exist_ok=True)
        self.log_dir = log_dir

    def get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Create and return a list of standard training callbacks.

        Returns:
            A list of tf.keras.callbacks.Callback instances for early
            stopping, LR reduction, model checkpointing, NaN termination,
            backup/restore, and CSV logging.
        """
        checkpoint_path = os.path.join(
            self.config.MODEL_PATH,
            "best_model.keras"
        )
        log_file = os.path.join(self.log_dir, "training_log.csv")

        resume_checkpoint_path = os.path.join(
            self.config.MODEL_PATH,
            "latest_checkpoint.keras"
        )

        return [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=resume_checkpoint_path,
                save_best_only=False,
                verbose=0,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=self.config.PATIENCE_EARLY,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.config.LR_FACTOR,
                patience=self.config.PATIENCE_LR,
                min_lr=self.config.MIN_LR,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.CSVLogger(
                filename=log_file,
                append=True,
            ),
        ]
