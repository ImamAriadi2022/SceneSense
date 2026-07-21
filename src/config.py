import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class Config:
    """Immutable configuration for the SceneSense pipeline.

    All hyperparameters, paths, and class definitions are centralised here.
    Override any field via ``Config(IMAGE_SIZE=(...))``.
    """
    IMAGE_SIZE: Tuple[int, int] = (150, 150)
    BATCH_SIZE: int = 32
    EPOCHS: int = 50
    LEARNING_RATE: float = 0.001
    RANDOM_SEED: int = 42
    DATASET_PATH: str = field(default_factory=lambda: os.getenv("SCENESENSE_DATASET_PATH", "dataset"))
    OUTPUT_PATH: str = field(default_factory=lambda: os.getenv("SCENESENSE_OUTPUT_PATH", "outputs"))
    MODEL_PATH: str = field(default_factory=lambda: os.getenv("SCENESENSE_MODEL_PATH", "saved_model"))
    TFLITE_PATH: str = field(default_factory=lambda: os.getenv("SCENESENSE_TFLITE_PATH", "tflite"))
    TFJS_PATH: str = field(default_factory=lambda: os.getenv("SCENESENSE_TFJS_PATH", "tfjs_model"))
    CHECKPOINT_PATH: str = field(default_factory=lambda: os.getenv("SCENESENSE_CHECKPOINT_PATH", "checkpoints"))
    LOG_PATH: str = field(default_factory=lambda: os.getenv("SCENESENSE_LOG_PATH", "logs"))
    CLASSES: Tuple[str, ...] = field(default_factory=lambda: (
        "buildings", "forest", "glacier", "mountain", "sea", "street"
    ))
    TRAIN_SPLIT: float = 0.7
    VAL_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.15
    PATIENCE_EARLY: int = 20
    PATIENCE_LR: int = 5
    LR_FACTOR: float = 0.5
    MIN_LR: float = 1e-7
    MODEL_INPUT_SHAPE: Tuple[int, ...] = field(default_factory=lambda: (150, 150, 3))
    NUM_CLASSES: int = 6
