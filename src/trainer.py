from typing import List, Optional

import tensorflow as tf

from src.config import Config
from src.callbacks import CallbackFactory


class Trainer:
    def __init__(self, config: Config) -> None:
        self.config = config

    def train(
        self,
        model: tf.keras.Model,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    ) -> tf.keras.callbacks.History:
        if callbacks is None:
            factory = CallbackFactory(self.config)
            callbacks = factory.get_callbacks()

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.EPOCHS,
            callbacks=callbacks,
            verbose=1,
        )
        return history
