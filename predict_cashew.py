import os
import argparse
import glob
from ultralytics import YOLO

def predict(
    image_path,
    model_path="cashew_models/cadi_yolo_pretrained.pt",
    conf=0.25,
    save_dir="cashew_predictions"
):
    """
    Runs inference on cashew images using the trained YOLO model.
    """
    if not os.path.exists(model_path):
        # Fallback to local best weights if available
        if os.path.exists("cashew_models/cashew_yolo_best.pt"):
            model_path = "cashew_models/cashew_yolo_best.pt"
        else:
            raise FileNotFoundError(f"Model weights not found at: {model_path}")

    print(f"Loading Cashew Detection Model from: {model_path}")
    model = YOLO(model_path)
    os.makedirs(save_dir, exist_ok=True)

    # Support single image or folder of images
    if os.path.isdir(image_path):
        images = glob.glob(os.path.join(image_path, "*.jpg")) + glob.glob(os.path.join(image_path, "*.png"))
    else:
        images = [image_path]

    print(f"Found {len(images)} image(s) for inference.")
    for img in images:
        results = model.predict(img, conf=conf, imgsz=640, device="cpu")
        for res in results:
            print(f"\nImage: {img}")
            n_boxes = len(res.boxes)
            print(f"Detected {n_boxes} instances:")
            for b in res.boxes:
                cls_id = int(b.cls[0])
                cls_name = model.names.get(cls_id, str(cls_id))
                confidence = float(b.conf[0])
                box = [round(x, 1) for x in b.xyxy[0].tolist()]
                print(f"  - [{cls_name.upper()}]: confidence={confidence:.2f}, bbox={box}")

            # Save annotated image
            base_name = os.path.basename(img)
            out_file = os.path.join(save_dir, f"pred_{base_name}")
            res.save(filename=out_file)
            print(f"  -> Saved visualization: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cashew Disease & Pest Detection")
    parser.add_argument("--image", default="cashew_dataset/test/images", help="Path to image file or directory")
    parser.add_argument("--model", default="cashew_models/cadi_yolo_pretrained.pt", help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--out", default="cashew_predictions", help="Directory to save visual outputs")
    args = parser.parse_args()

    predict(
        image_path=args.image,
        model_path=args.model,
        conf=args.conf,
        save_dir=args.out
    )
