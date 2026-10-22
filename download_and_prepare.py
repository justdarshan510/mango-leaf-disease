import os
import sys
import zipfile
import urllib.request
import time

def report_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_tot = total_size / (1024 * 1024)
        sys.stdout.write(f"\rDownloading: {mb_down:.1f} MB / {mb_tot:.1f} MB ({percent:.1f}%)")
        sys.stdout.flush()

def download_file(url, target_path):
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000000:
        print(f"Already downloaded: {target_path} ({os.path.getsize(target_path)/(1024*1024):.1f} MB)")
        return
    print(f"\nStarting download: {url} -> {target_path}")
    urllib.request.urlretrieve(url, target_path, reporthook=report_progress)
    print(f"\nCompleted: {target_path}")

def extract_zip(zip_path, target_dir):
    print(f"\nExtracting {zip_path} into {target_dir}...")
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(target_dir)
    print(f"Extraction complete for {target_dir}")

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    zips_dir = os.path.join(base_dir, "cashew_data")
    dataset_dir = os.path.join(base_dir, "cashew_dataset")
    os.makedirs(zips_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    splits = [
        ("test", "https://huggingface.co/datasets/KaraAgro-AI/CADI-AI/resolve/main/Data/test.zip", "test.zip"),
        ("valid", "https://huggingface.co/datasets/KaraAgro-AI/CADI-AI/resolve/main/Data/val.zip", "val.zip"),
        ("train", "https://huggingface.co/datasets/KaraAgro-AI/CADI-AI/resolve/main/Data/train.zip", "train.zip"),
    ]

    # 1. Download
    for split_name, url, filename in splits:
        zip_path = os.path.join(zips_dir, filename)
        download_file(url, zip_path)

    # 2. Extract
    for split_name, _, filename in splits:
        zip_path = os.path.join(zips_dir, filename)
        target_split_dir = os.path.join(dataset_dir, split_name)
        extract_zip(zip_path, target_split_dir)

    # 3. Create data.yaml
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    # Use forward slashes for cross-platform YOLO compatibility
    dataset_dir_fwd = dataset_dir.replace("\\", "/")
    yaml_content = f"""path: {dataset_dir_fwd}
train: train/images
val: valid/images
test: test/images

nc: 3
names:
  0: abiotic
  1: disease
  2: insect
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"\nCreated YOLO data configuration: {yaml_path}")

    # 4. Summary verification
    print("\n--- Dataset Summary ---")
    for split_name, _, _ in splits:
        img_dir = os.path.join(dataset_dir, split_name, "images")
        lbl_dir = os.path.join(dataset_dir, split_name, "labels")
        n_imgs = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
        n_lbls = len(os.listdir(lbl_dir)) if os.path.exists(lbl_dir) else 0
        print(f"Split [{split_name}]: {n_imgs} images, {n_lbls} label files")

if __name__ == "__main__":
    main()
