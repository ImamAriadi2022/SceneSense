import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf

from src.config import Config
from src.utils import load_image_for_inference


def load_savedmodel(model_path: str) -> tf.keras.Model:
    try:
        return tf.keras.models.load_model(model_path)
    except ValueError:
        layer = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
        inputs = tf.keras.Input(shape=Config().MODEL_INPUT_SHAPE)
        outputs = layer(inputs)
        return tf.keras.Model(inputs, outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with SceneSense model")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to SavedModel directory. Defaults to saved_model/",
    )
    parser.add_argument(
        "--tflite_path",
        type=str,
        default=None,
        help="Path to TFLite model file. Mutually exclusive with --model_path.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input image.",
    )
    return parser.parse_args()


def load_labels(path: str) -> list:
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def predict_with_savedmodel(model_path: str, image_path: str, class_names: list) -> None:
    model = load_savedmodel(model_path)
    input_image = load_image_for_inference(image_path, target_size=Config().IMAGE_SIZE)
    preds = model.predict(input_image, verbose=0)
    if isinstance(preds, dict):
        preds = list(preds.values())[0]
    preds = np.array(preds).squeeze()
    predicted_idx = int(np.argmax(preds))
    confidence = float(preds[predicted_idx])
    predicted_class = class_names[predicted_idx] if class_names else str(predicted_idx)

    print(f"\nPredicted Class: {predicted_class}")
    print(f"Confidence:      {confidence:.4f} ({confidence * 100:.2f}%)")

    print("\nTop-3 predictions:")
    top3 = np.argsort(preds)[-3:][::-1]
    for idx in top3:
        cls_name = class_names[idx] if class_names else str(idx)
        print(f"  {cls_name}: {preds[idx]:.4f} ({preds[idx] * 100:.2f}%)")


def predict_with_tflite(tflite_path: str, image_path: str, class_names: list) -> None:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_image = load_image_for_inference(image_path, target_size=Config().IMAGE_SIZE)
    input_image = input_image.astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], input_image)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]
    predicted_idx = int(np.argmax(preds))
    confidence = float(preds[predicted_idx])
    predicted_class = class_names[predicted_idx] if class_names else str(predicted_idx)

    print(f"\nPredicted Class: {predicted_class}")
    print(f"Confidence:      {confidence:.4f} ({confidence * 100:.2f}%)")

    print("\nTop-3 predictions:")
    top3 = np.argsort(preds)[-3:][::-1]
    for idx in top3:
        cls_name = class_names[idx] if class_names else str(idx)
        print(f"  {cls_name}: {preds[idx]:.4f} ({preds[idx] * 100:.2f}%)")


def main() -> None:
    args = parse_args()

    config = Config()
    model_path = args.model_path or config.MODEL_PATH
    labels_path = os.path.join(model_path, "labels.txt")
    class_names = load_labels(labels_path) if os.path.isfile(labels_path) else list(config.CLASSES)

    if not os.path.isfile(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)

    if args.tflite_path:
        predict_with_tflite(args.tflite_path, args.image_path, class_names)
    else:
        predict_with_savedmodel(model_path, args.image_path, class_names)


if __name__ == "__main__":
    main()
