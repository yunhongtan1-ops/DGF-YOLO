import argparse
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run DGF-YOLO inference and report FPS.")
    parser.add_argument("--weights", required=True, help="Path to trained weights.")
    parser.add_argument("--source", required=True, help="Image file or directory.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def collect_sources(source):
    path = Path(source)
    if path.is_file():
        return [path]
    suffixes = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    images = []
    for suffix in suffixes:
        images.extend(path.glob(suffix))
    return sorted(images)


def main():
    args = parse_args()
    image_paths = collect_sources(args.source)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.source}")

    model = YOLO(args.weights)

    for i in range(args.warmup):
        model.predict(
            source=str(image_paths[i % len(image_paths)]),
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            half=args.half,
            save=False,
            verbose=False,
        )
    sync_cuda()

    inference_times = []
    for image_path in image_paths:
        sync_cuda()
        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            half=args.half,
            save=args.save,
            verbose=False,
        )
        sync_cuda()
        inference_times.append(results[0].speed["inference"])

    avg_ms = float(np.mean(inference_times))
    fps = 1000.0 / avg_ms
    print(f"Images: {len(image_paths)}")
    print(f"Average inference time: {avg_ms:.3f} ms/image")
    print(f"Inference FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
