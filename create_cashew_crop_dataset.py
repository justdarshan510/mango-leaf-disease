import os
import glob
import cv2
from tqdm import tqdm

CLASSES = {
    0: "abiotic",
    1: "disease",
    2: "insect"
}

def crop_instances(dataset_root="cashew_dataset", output_root="cashew_cropped_dataset", max_crops_per_class=2000):
    """
    Extracts labeled bounding box crops from YOLO annotations and organizes
    them into class directories for MobileNetV2 image classification.
    """
    print(f"Generating classification crop dataset in '{output_root}'...")
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(dataset_root, split, "images")
        lbl_dir = os.path.join(dataset_root, split, "labels")
        if not os.path.exists(img_dir):
            continue

        counts = {name: 0 for name in CLASSES.values()}

        # Create target directories
        for cname in CLASSES.values():
            os.makedirs(os.path.join(output_root, split, cname), exist_ok=True)

        img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
        for img_path in tqdm(img_files, desc=f"Processing {split} crops"):
            base = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, base + ".txt")
            if not os.path.exists(lbl_path):
                continue

            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if not lines:
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                cname = CLASSES.get(cls_id)
                if not cname:
                    continue

                if counts[cname] >= max_crops_per_class:
                    continue

                xc, yc, nw, nh = map(float, parts[1:])
                # Convert normalized YOLO bbox to pixel coords
                x1 = max(0, int((xc - nw / 2) * w))
                y1 = max(0, int((yc - nh / 2) * h))
                x2 = min(w, int((xc + nw / 2) * w))
                y2 = min(h, int((yc + nh / 2) * h))

                # Discard degenerate zero-size boxes
                if (x2 - x1) < 15 or (y2 - y1) < 15:
                    continue

                crop = img[y1:y2, x1:x2]
                out_name = f"{base}_crop{idx}.jpg"
                out_path = os.path.join(output_root, split, cname, out_name)
                cv2.imwrite(out_path, crop)
                counts[cname] += 1

        print(f"[{split}] Cropped counts: {counts}")

if __name__ == "__main__":
    crop_instances()
