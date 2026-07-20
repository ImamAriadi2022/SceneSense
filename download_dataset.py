import os
import shutil
import sys

KAGGLEDATASETS = True
try:
    import kagglehub
except ImportError:
    KAGGLEDATASETS = False


DATASET_SLUG = "puneet6060/intel-image-classification"
RAW_DIR = "dataset/raw"


def download_with_kagglehub() -> str:
    print("Downloading dataset via kagglehub...")
    path = kagglehub.dataset_download(DATASET_SLUG)
    print(f"Dataset downloaded to: {path}")
    return path


def merge_to_raw(source_path: str) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)

    for split in ["seg_train", "seg_test"]:
        split_dir = os.path.join(source_path, split, split)
        if not os.path.isdir(split_dir):
            split_dir = os.path.join(source_path, split)
        if not os.path.isdir(split_dir):
            continue

        for cls_name in os.listdir(split_dir):
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            dest_cls = os.path.join(RAW_DIR, cls_name)
            os.makedirs(dest_cls, exist_ok=True)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    src_file = os.path.join(cls_dir, fname)
                    dst_file = os.path.join(dest_cls, fname)
                    if not os.path.isfile(dst_file):
                        shutil.copy2(src_file, dst_file)

    total = sum(
        len(files)
        for _, _, files in os.walk(RAW_DIR)
    )
    print(f"Merged {total} images into {RAW_DIR}/")
    return RAW_DIR


def manual_instructions() -> None:
    print("=" * 70)
    print("Kaggle API / kagglehub not available or download failed.")
    print()
    print("Manual download instructions:")
    print("1. Go to: https://www.kaggle.com/datasets/puneet6060/intel-image-classification")
    print("2. Click 'Download' and save the ZIP file.")
    print("3. Extract the ZIP file.")
    print("4. You will get directories: seg_train/, seg_test/, seg_pred/")
    print("5. Run the following to prepare the dataset:")
    print()
    print("   mkdir -p dataset/raw/buildings dataset/raw/forest dataset/raw/glacier \\")
    print("       dataset/raw/mountain dataset/raw/sea dataset/raw/street")
    print("   cp seg_train/seg_train/buildings/* dataset/raw/buildings/")
    print("   cp seg_test/seg_test/buildings/* dataset/raw/buildings/")
    print("   # repeat for all 6 classes")
    print()
    print("Then run: python train.py --data_dir dataset/raw")
    print("=" * 70)


def main() -> None:
    if os.path.isdir(RAW_DIR) and len(os.listdir(RAW_DIR)) > 0:
        print(f"Raw dataset already exists at {RAW_DIR}/")
        cls_count = sum(
            len(files) for _, _, files in os.walk(RAW_DIR)
        )
        print(f"Total images: {cls_count}")
        return

    try:
        source = download_with_kagglehub()
        merge_to_raw(source)
        print("Dataset ready for training!")
        print(f"Run: python train.py --data_dir {RAW_DIR}")
    except Exception as e:
        print(f"Error: {e}")
        manual_instructions()
        sys.exit(1)


if __name__ == "__main__":
    main()
