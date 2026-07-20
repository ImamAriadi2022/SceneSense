from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class Config:
    IMAGE_SIZE: Tuple[int, int] = (64, 64)
    BATCH_SIZE: int = 32
    EPOCHS: int = 20
    LEARNING_RATE: float = 0.0005
    RANDOM_SEED: int = 42
    DATASET_PATH: str = "dataset"
    OUTPUT_PATH: str = "outputs"
    MODEL_PATH: str = "saved_model"
    TFLITE_PATH: str = "tflite"
    TFJS_PATH: str = "tfjs_model"
    CLASSES: Tuple[str, ...] = field(default_factory=lambda: (
        "buildings", "forest", "glacier", "mountain", "sea", "street"
    ))
    TRAIN_SPLIT: float = 0.7
    VAL_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.15
    PATIENCE_EARLY: int = 10
    PATIENCE_LR: int = 5
    LR_FACTOR: float = 0.5
    MIN_LR: float = 1e-7
    MODEL_INPUT_SHAPE: Tuple[int, ...] = field(default_factory=lambda: (64, 64, 3))
    NUM_CLASSES: int = 6
