import os
import random
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def plot_history(
    history: tf.keras.callbacks.History,
    output_path: str = "outputs",
    timestamp: str = "",
) -> None:
    ensure_dir(output_path)

    acc: List[float] = history.history["accuracy"]
    val_acc: List[float] = history.history["val_accuracy"]
    loss: List[float] = history.history["loss"]
    val_loss: List[float] = history.history["val_loss"]
    epochs: range = range(1, len(acc) + 1)

    suffix: str = f"_{timestamp}" if timestamp else ""
    plt.style.use("ggplot")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, acc, "b-", label="Training Accuracy")
    ax1.plot(epochs, val_acc, "r-", label="Validation Accuracy")
    ax1.set_title("Training & Validation Accuracy")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, loss, "b-", label="Training Loss")
    ax2.plot(epochs, val_loss, "r-", label="Validation Loss")
    ax2.set_title("Training & Validation Loss")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(output_path, f"accuracy{suffix}.png"))
    fig.savefig(os.path.join(output_path, f"loss{suffix}.png"))
    plt.close(fig)


def load_image_for_inference(
    image_path: str,
    target_size: tuple = (150, 150),
) -> np.ndarray:
    img: tf.Tensor = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0
    return np.expand_dims(img.numpy(), axis=0)
