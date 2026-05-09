import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train DGF-YOLO.")
    parser.add_argument("--model", default="ultralytics/cfg/models/v12/yolov12-GCF-SSFM-DGRM.yaml")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/dgf_yolo.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
