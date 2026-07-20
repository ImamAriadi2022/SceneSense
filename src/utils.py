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

    fig_acc, ax_acc = plt.subplots(figsize=(7, 5))
    ax_acc.plot(epochs, acc, "b-", label="Training Accuracy")
    ax_acc.plot(epochs, val_acc, "r-", label="Validation Accuracy")
    ax_acc.set_title("Training & Validation Accuracy")
    ax_acc.set_xlabel("Epochs")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True)
    fig_acc.tight_layout()
    fig_acc.savefig(os.path.join(output_path, f"accuracy{suffix}.png"))
    plt.close(fig_acc)

    fig_loss, ax_loss = plt.subplots(figsize=(7, 5))
    ax_loss.plot(epochs, loss, "b-", label="Training Loss")
    ax_loss.plot(epochs, val_loss, "r-", label="Validation Loss")
    ax_loss.set_title("Training & Validation Loss")
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True)
    fig_loss.tight_layout()
    fig_loss.savefig(os.path.join(output_path, f"loss{suffix}.png"))
    plt.close(fig_loss)


def load_image_for_inference(
    image_path: str,
    target_size: tuple = (128, 128),
) -> np.ndarray:
    img: tf.Tensor = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32)
    return np.expand_dims(img.numpy(), axis=0)
