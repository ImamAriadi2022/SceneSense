import tensorflow as tf


class AugmentationPipeline:
    def __init__(self, config) -> None:
        self.config = config
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        self.pipeline = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        ])

    def apply(self, images: tf.Tensor, labels: tf.Tensor) -> tuple:
        augmented = self.pipeline(images, training=True)
        return augmented, labels

    def get_augmentation_layer(self) -> tf.keras.Sequential:
        return self.pipeline
