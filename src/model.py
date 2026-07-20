import tensorflow as tf

from src.config import Config


class SceneSenseModel:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.model: tf.keras.Sequential = self._build()

    def _build(self) -> tf.keras.Sequential:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.config.MODEL_INPUT_SHAPE),
            tf.keras.layers.Rescaling(1.0 / 255, name="rescaling"),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", name="conv1"),
            tf.keras.layers.BatchNormalization(name="bn1"),
            tf.keras.layers.ReLU(name="relu1"),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool1"),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", name="conv2"),
            tf.keras.layers.BatchNormalization(name="bn2"),
            tf.keras.layers.ReLU(name="relu2"),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool2"),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", name="conv3"),
            tf.keras.layers.BatchNormalization(name="bn3"),
            tf.keras.layers.ReLU(name="relu3"),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool3"),
            tf.keras.layers.Dropout(0.3, name="dropout1"),
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(256, name="dense1"),
            tf.keras.layers.Dropout(0.5, name="dropout2"),
            tf.keras.layers.Dense(
                self.config.NUM_CLASSES,
                activation="softmax",
                name="output",
            ),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        return model

    def summary(self) -> None:
        self.model.summary()

    def get_model(self) -> tf.keras.Sequential:
        return self.model
