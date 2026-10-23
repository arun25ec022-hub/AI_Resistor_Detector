"""Train a YOLO resistor detector.

Examples:
    python train.py
    python train.py --epochs 100 --batch 16
    python train.py --weights runs/detect/models/resistor_detector_fast-3/weights/best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train a resistor-detection model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_DIR / "dataset" / "data.yaml",
        help="Path to the YOLO dataset YAML file.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_DIR / "yolo11n.pt",
        help="Model weights to start from (pretrained or a previous checkpoint).",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default=None,
        help="Training device, such as 'cpu' or '0'. Defaults to automatic selection.",
    )
    parser.add_argument("--project", type=Path, default=PROJECT_DIR / "runs" / "detect")
    parser.add_argument("--name", default="resistor_detector")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    weights_path = args.weights.expanduser().resolve()
    output_dir = args.project.expanduser().resolve()

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset configuration was not found: {data_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights were not found: {weights_path}")
    if args.epochs < 1 or args.imgsz < 1 or args.batch < 1 or args.workers < 0:
        raise ValueError("epochs, imgsz, and batch must be positive; workers cannot be negative.")

    model = YOLO(str(weights_path))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(output_dir),
        name=args.name,
    )

    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    print(f"Training completed. Best model: {best_weights}")


if __name__ == "__main__":
    main()
