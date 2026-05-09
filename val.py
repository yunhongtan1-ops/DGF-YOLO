import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Validate DGF-YOLO.")
    parser.add_argument("--weights", required=True, help="Path to trained weights, for example runs/detect/train/weights/best.pt")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/dgf_yolo.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=args.device)


if __name__ == "__main__":
    main()
