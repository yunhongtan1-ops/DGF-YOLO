# DGF-YOLO

DGF-YOLO is an object detection project built on YOLOv12. The core model in this repository is:

**GCF + SSFM + DGRM**

The default model configuration is located at:

```text
ultralytics/cfg/models/v12/yolov12-GCF-SSFM-DGRM.yaml
```

## Highlights

- Uses `GCF` blocks for feature extraction and global-context feature modeling.
- Uses `SSFM` for scale-sequence feature fusion across P3, P4, and P5 features.
- Uses `DGRM` for detail-guided refinement before detection.

## Project Structure

```text
DGF-YOLO/
|-- train.py
|-- val.py
|-- predict.py
|-- requirements.txt
|-- ultralytics/
|   |-- cfg/
|   |   |-- datasets/
|   |   |   `-- dgf_yolo.yaml
|   |   `-- models/v12/
|   |       `-- yolov12-GCF-SSFM-DGRM.yaml
|   `-- nn/
|       `-- AddModules/
|           |-- GCF.py
|           `-- SSFM_DGRM.py
```

## Installation

Create a Python environment and install dependencies:

```bash
conda create -n dgf-yolo python=3.11
conda activate dgf-yolo
pip install -r requirements.txt
pip install -e .
```

`flash-attn` is optional and depends on your CUDA, PyTorch, Python, and Linux environment. Install a matching wheel manually if your environment requires it.

## Dataset

Experiments in the paper are conducted on two public water-surface floating debris datasets:

1. **Flow-Img**

   Flow-Img is the vision-based subset of the FloW dataset for floating waste detection in inland waters. It can be accessed from the official ORCA-Uboat dataset page:

   https://orca-tech.cn/datasets/FloW/FloW-Img

   The dataset repository and paper information are also available at:

   https://github.com/ORCA-Uboat/FloW-Dataset

2. **IWHR_AI_Lable_Floater_V1**

   IWHR_AI_Lable_Floater_V1 is a public floater detection dataset collected from real inland-water scenes. It can be downloaded from Figshare:

   https://doi.org/10.6084/m9.figshare.27376851.v1

   The corresponding data descriptor paper is available at:

   https://www.nature.com/articles/s41597-025-04594-9

The datasets are not included in this repository. Please download them from the official sources above and convert or arrange them in YOLO format before training.

Prepare your dataset in YOLO format:

```text
datasets/DGF-YOLO/
|-- images/
|   |-- train/
|   |-- val/
|   `-- test/
`-- labels/
    |-- train/
    |-- val/
    `-- test/
```

Then edit:

```text
ultralytics/cfg/datasets/dgf_yolo.yaml
```

Example:

```yaml
path: ../datasets/DGF-YOLO
train: images/train
val: images/val
test: images/test

names:
  0: object
```

## Training

```bash
python train.py --data ultralytics/cfg/datasets/dgf_yolo.yaml --epochs 300 --imgsz 640 --batch 16 --device 0
```

The default model used by `train.py` is `GCF + SSFM + DGRM`.

You can also specify the model explicitly:

```bash
python train.py \
  --model ultralytics/cfg/models/v12/yolov12-GCF-SSFM-DGRM.yaml \
  --data ultralytics/cfg/datasets/dgf_yolo.yaml
```

## Validation

```bash
python val.py \
  --weights runs/detect/train/weights/best.pt \
  --data ultralytics/cfg/datasets/dgf_yolo.yaml \
  --device 0
```

## Prediction and FPS Test

```bash
python predict.py \
  --weights runs/detect/train/weights/best.pt \
  --source path/to/images \
  --device 0
```

Use `--save` to save prediction visualizations.

## Weights

Model weights are not committed to this repository. Put trained weights under `runs/` locally, or publish them separately with GitHub Releases, Google Drive, Hugging Face, or another model hosting service.

## Acknowledgements

This project is based on the YOLOv12 codebase:

```bibtex
@article{tian2025yolov12,
  title={YOLOv12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.12524},
  year={2025}
}
```

## License

This repository follows the AGPL-3.0 license inherited from the original YOLOv12/Ultralytics codebase. See [LICENSE](LICENSE) for details.
