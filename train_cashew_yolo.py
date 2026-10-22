import os
import argparse
from ultralytics import YOLO

def train(
    data_yaml="cashew_dataset/data.yaml",
    model_name="yolov8n.pt",  # can use 'yolov8n.pt', 'yolo11n.pt', or 'cashew_models/cadi_yolo_pretrained.pt'
    epochs=10,
    imgsz=640,
    batch=8,
    project="runs/cashew_train",
    name="cashew_detector",
    device="cpu"
):
    """
    Trains YOLO on the Cashew Disease Identification (CADI-AI) dataset.
    Classes:
      0: abiotic
      1: disease
      2: insect
    """
    data_path = os.path.abspath(data_yaml)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data.yaml not found at: {data_path}")

    print("=" * 60)
    print("🚀 Cashew Disease & Pest Detection (CADI-AI) Training")
    print("=" * 60)
    print(f"Dataset config: {data_path}")
    print(f"Base model:     {model_name}")
    print(f"Epochs:         {epochs}")
    print(f"Batch size:     {batch}")
    print(f"Image size:     {imgsz}x{imgsz}")
    print(f"Device:         {device}")
    print("=" * 60)

    # Initialize YOLO model
    model = YOLO(model_name)

    # Train model
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        plots=True,
        save=True,
        val=True
    )

    print("\n✅ Training complete!")
    best_weight = os.path.join(project, name, "weights", "best.pt")
    if os.path.exists(best_weight):
        print(f"🏆 Best model weights saved to: {best_weight}")
        # Copy to cashew_models/ for easy access
        os.makedirs("cashew_models", exist_ok=True)
        import shutil
        target_copy = os.path.join("cashew_models", "cashew_yolo_best.pt")
        shutil.copy(best_weight, target_copy)
        print(f"📦 Model copied to: {target_copy}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO on CADI-AI Cashew Dataset")
    parser.add_argument("--data", default="cashew_dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained weights or model architecture (e.g. yolov8n.pt or cashew_models/cadi_yolo_pretrained.pt)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (640, 320)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Device to use ('cpu', '0', etc.)")
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )
