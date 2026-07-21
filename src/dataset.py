import os
import shutil
from typing import Dict, List, Optional, Tuple

import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import Config


class SceneSenseDataset:
    """Handles dataset loading, splitting, and statistics for scene images.

    Organises images from a raw directory into train/val/test splits and
    loads them as TensorFlow Dataset objects.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.class_names: List[str] = list(config.CLASSES)
        self.train_ds: Optional[tf.data.Dataset] = None
        self.val_ds: Optional[tf.data.Dataset] = None
        self.test_ds: Optional[tf.data.Dataset] = None

    def _get_file_paths(self, data_dir: str) -> Tuple[List[str], List[int]]:
        """Collect image file paths and their class indices from a directory.

        Args:
            data_dir: Path to the root directory containing class subdirectories.

        Returns:
            A tuple of (paths, labels) where paths is a list of image file
            paths and labels is a list of corresponding integer class indices.
        """
        paths: List[str] = []
        labels: List[int] = []
        for idx, cls_name in enumerate(self.class_names):
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(os.path.join(cls_dir, fname))
                    labels.append(idx)
        return paths, labels

    def _copy_files(
        self,
        paths: List[str],
        labels: List[int],
        target_dir: str,
    ) -> None:
        """Copy image files into class subdirectories under target_dir.

        Args:
            paths: List of source image file paths.
            labels: List of integer class indices corresponding to each path.
            target_dir: Destination root directory (class folders created inside).
        """
        for path, label in zip(paths, labels):
            cls_name = self.class_names[label]
            dest_dir = os.path.join(target_dir, cls_name)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(path, os.path.join(dest_dir, os.path.basename(path)))

    def split_and_save(self, data_dir: str) -> None:
        """Split images into train/val/test sets and save to structured directories.

        Args:
            data_dir: Path to the raw dataset with class subdirectories.

        Raises:
            ValueError: If no valid image files are found in data_dir.
        """
        paths, labels = self._get_file_paths(data_dir)
        if not paths:
            raise ValueError(f"No images found in {data_dir}")

        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            paths, labels,
            test_size=self.config.VAL_SPLIT + self.config.TEST_SPLIT,
            random_state=self.config.RANDOM_SEED,
            stratify=labels,
        )
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=self.config.TEST_SPLIT / (self.config.VAL_SPLIT + self.config.TEST_SPLIT),
            random_state=self.config.RANDOM_SEED,
            stratify=temp_labels,
        )

        base = self.config.DATASET_PATH
        self._copy_files(train_paths, train_labels, os.path.join(base, "train"))
        self._copy_files(val_paths, val_labels, os.path.join(base, "val"))
        self._copy_files(test_paths, test_labels, os.path.join(base, "test"))

        print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    def load_datasets(self) -> None:
        """Load train, validation, and test datasets from the split directory.

        Populates train_ds, val_ds, and test_ds as TensorFlow Dataset objects.

        Raises:
            FileNotFoundError: If the train directory does not exist.
        """
        base = self.config.DATASET_PATH
        train_dir = os.path.join(base, "train")
        val_dir = os.path.join(base, "val")
        test_dir = os.path.join(base, "test")

        if not os.path.isdir(train_dir):
            raise FileNotFoundError(
                f"Train directory not found: {train_dir}. Run split_and_save first."
            )

        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            image_size=self.config.IMAGE_SIZE,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            seed=self.config.RANDOM_SEED,
        )
        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            val_dir,
            image_size=self.config.IMAGE_SIZE,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            seed=self.config.RANDOM_SEED,
        )
        self.test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            test_dir,
            image_size=self.config.IMAGE_SIZE,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            seed=self.config.RANDOM_SEED,
        )
        self.class_names = self.train_ds.class_names

    def get_dataset_statistics(self) -> Dict[str, int]:
        """Compute per-class and total image counts for each split.

        Returns:
            A dictionary mapping split_class keys (e.g. 'train_buildings')
            and split_total keys (e.g. 'train_total') to image counts.
        """
        stats = {}
        for split_name in ["train", "val", "test"]:
            split_dir = os.path.join(self.config.DATASET_PATH, split_name)
            total = 0
            if os.path.isdir(split_dir):
                for cls_name in self.class_names:
                    cls_dir = os.path.join(split_dir, cls_name)
                    if os.path.isdir(cls_dir):
                        count = len([
                            f for f in os.listdir(cls_dir)
                            if f.lower().endswith((".jpg", ".jpeg", ".png"))
                        ])
                        total += count
                        stats[f"{split_name}_{cls_name}"] = count
            stats[f"{split_name}_total"] = total
        return stats
