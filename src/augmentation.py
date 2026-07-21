import tensorflow as tf

from src.config import Config


class AugmentationPipeline:
    """Builds and applies random image augmentations for training.

    Uses Keras preprocessing layers to apply flips, rotations, zooms,
    translations, and brightness/contrast adjustments.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Construct the sequential augmentation layer stack."""
        self.pipeline = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            tf.keras.layers.RandomBrightness(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ])

    def apply(self, images: tf.Tensor, labels: tf.Tensor) -> tuple:
        """Apply augmentations to a batch of images.

        Args:
            images: Batch of input image tensors.
            labels: Corresponding label tensors (passed through unchanged).

        Returns:
            A tuple of (augmented_images, labels).
        """
        augmented = self.pipeline(images, training=True)
        return augmented, labels

    def get_augmentation_layer(self) -> tf.keras.Sequential:
        """Return the internal Keras Sequential augmentation layer.

        Returns:
            The augmentation pipeline as a tf.keras.Sequential instance.
        """
        return self.pipeline
