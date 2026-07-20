import argparse
import os
import sys

import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.utils import set_seed, ensure_dir, plot_history, get_timestamp
from src.dataset import SceneSenseDataset
from src.augmentation import AugmentationPipeline
from src.model import SceneSenseModel
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.exporter import Exporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SceneSense model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to raw dataset directory (optional). If provided, splits and saves to dataset/.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()
    if args.epochs:
        config = Config(EPOCHS=args.epochs)
    if args.batch_size:
        config = Config(BATCH_SIZE=args.batch_size)

    set_seed(config.RANDOM_SEED)
    timestamp = get_timestamp()

    print("=" * 60)
    print("SceneSense - Image Classification Training")
    print("=" * 60)

    dataset = SceneSenseDataset(config)

    if args.data_dir:
        print(f"\n[1/7] Splitting dataset from: {args.data_dir}")
        dataset.split_and_save(args.data_dir)
    else:
        base = config.DATASET_PATH
        train_dir = os.path.join(base, "train")
        if not os.path.isdir(train_dir):
            print(f"Error: {train_dir} not found. Use --data_dir to specify raw data.")
            sys.exit(1)

    print("\n[2/7] Loading datasets...")
    dataset.load_datasets()

    stats = dataset.get_dataset_statistics()
    print(f"Train samples: {stats.get('train_total', 0)}")
    print(f"Val samples:   {stats.get('val_total', 0)}")
    print(f"Test samples:  {stats.get('test_total', 0)}")
    print(f"Classes: {dataset.class_names}")

    print("\n[3/7] Setting up augmentation...")
    aug_pipeline = AugmentationPipeline(config)
    train_ds = dataset.train_ds.map(aug_pipeline.apply, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = dataset.val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = dataset.test_ds.prefetch(tf.data.AUTOTUNE)

    print("\n[4/7] Building model...")
    model_builder = SceneSenseModel(config)
    model = model_builder.get_model()
    model.summary()

    print("\n[5/7] Training model...")
    trainer = Trainer(config)
    history = trainer.train(model, train_ds, val_ds)

    print("\n[6/7] Evaluating model...")
    evaluator = Evaluator(config)
    results = evaluator.evaluate(model, test_ds, dataset.class_names)

    print(f"\n  Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"  Final Test Loss:     {results['loss']:.4f}")

    print("\n[7/7] Exporting model...")
    exporter = Exporter(config)
    exporter.export_all(model, dataset.class_names)

    print("\nPlotting training history...")
    plot_history(history, output_path=config.OUTPUT_PATH, timestamp=timestamp)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
