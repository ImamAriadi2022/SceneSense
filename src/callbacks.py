import os
from typing import List, Optional

import tensorflow as tf

from src.config import Config


class CallbackFactory:
    """Creates a list of Keras callbacks for training.

    Includes early stopping, learning rate reduction on plateau, model
    checkpointing, NaN termination, backup/restore, CSV logging, and TensorBoard.
    """

    def __init__(self, config: Config, log_dir: Optional[str] = None) -> None:
        self.config = config
        self.log_dir = log_dir if log_dir is not None else config.LOG_PATH
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(config.MODEL_PATH, exist_ok=True)
        os.makedirs(config.CHECKPOINT_PATH, exist_ok=True)

    def get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Create and return a list of standard training callbacks.

        Returns:
            A list of tf.keras.callbacks.Callback instances for early
            stopping, LR reduction, model checkpointing, NaN termination,
            backup/restore, CSV logging, and TensorBoard.
        """
        checkpoint_path = os.path.join(
            self.config.CHECKPOINT_PATH,
            "best_model.keras"
        )
        log_file = os.path.join(self.log_dir, "training_log.csv")

        resume_checkpoint_path = os.path.join(
            self.config.CHECKPOINT_PATH,
            "latest_checkpoint.keras"
        )

        return [
            tf.keras.callbacks.TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=1,
            ),
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
