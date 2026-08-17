# NormVein

Official implementation and pretrained models for **NormVein: Normalizing
Finger-Vein Tasks with Large-Scale Synthetic Pretraining**.

NormVein uses [FingerVeinSyn-5M](https://github.com/EvanWang98/FingerVeinSyn-5M)
to pretrain models for rotated ROI detection, finger-shape and vein-pattern
segmentation, and finger-vein recognition.

![NormVein overview used in the paper](assets/normvein_overview.png)

## Pretrained models

The model files are published in the private candidate
[v0.1.0-candidate.1 Release](https://github.com/EvanWang98/NormVein/releases/tag/v0.1.0-candidate.1).
Each file is a model-only PyTorch artifact. Exact sizes, parameter counts,
source epochs, and SHA-256 digests are recorded in `weights/manifest.json`.

| Task | Model ID | Architecture | Checkpoint |
|---|---|---|---|
| ROI detection | `roi_rotated_faster_rcnn_r50_fpn` | Rotated Faster R-CNN, ResNet-50-FPN | `normvein_roi_rotated_faster_rcnn_r50_fpn_e006.pth` |
| ROI detection | `roi_rotated_ssd_r50_fpn` | Rotated SSD, ResNet-50-FPN | `normvein_roi_rotated_ssd_r50_fpn_e010.pth` |
| ROI detection | `roi_rotated_yolo_r50_fpn` | Rotated YOLO-style, ResNet-50-FPN | `normvein_roi_rotated_yolo_r50_fpn_e010.pth` |
| Segmentation | `seg_finger_shape_unet_r50` | U-Net, ResNet-50 encoder | `normvein_seg_finger_shape_unet_r50_e010.pth` |
| Segmentation | `seg_vein_pattern_unet_r50` | U-Net, ResNet-50 encoder | `normvein_seg_vein_pattern_unet_r50_e010.pth` |
| Recognition | `rec_mobilefacenet_large` | MobileFaceNet-Large, 512-D | `normvein_rec_mobilefacenet_large_e050.pth` |
| Recognition | `rec_iresnet100` | iResNet-100, 512-D | `normvein_rec_iresnet100_e044.pth` |
| Recognition | `rec_convnext_small` | ConvNeXt-Small, 512-D | `normvein_rec_convnext_small_e100.pth` |

The custom rotated YOLO-style model is not an Ultralytics YOLOv8 checkpoint.

## Installation and weight download

Python 3.9+ and PyTorch 2.x are recommended.

```bash
git clone https://github.com/EvanWang98/NormVein.git
cd NormVein
pip install -e .[all]
```

Download the Release assets into the expected task directories:

```bash
gh release download v0.1.0-candidate.1 --repo EvanWang98/NormVein --pattern "normvein_roi_*" --dir weights/detection
gh release download v0.1.0-candidate.1 --repo EvanWang98/NormVein --pattern "normvein_seg_*" --dir weights/segmentation
gh release download v0.1.0-candidate.1 --repo EvanWang98/NormVein --pattern "normvein_rec_*" --dir weights/recognition
gh release download v0.1.0-candidate.1 --repo EvanWang98/NormVein --pattern "manifest.json" --dir weights --clobber
python scripts/verify_weights.py --load-models
```

The last command verifies every SHA-256 digest, builds the eight architectures
shipped in `src/normvein`, and loads every `state_dict` with `strict=True`.

## Quick inference

Recognition (512-D L2-normalized embedding):

```bash
normvein recognize \
  --model rec_iresnet100 \
  --checkpoint weights/recognition/normvein_rec_iresnet100_e044.pth \
  --input path/to/roi.png \
  --output embedding.npy
```

Vein-pattern or finger-shape segmentation:

```bash
normvein segment \
  --model seg_vein_pattern_unet_r50 \
  --checkpoint weights/segmentation/normvein_seg_vein_pattern_unet_r50_e010.pth \
  --input path/to/roi.png \
  --output vein_mask.png
```

Rotated ROI detection:

```bash
normvein detect \
  --model roi_rotated_faster_rcnn_r50_fpn \
  --checkpoint weights/detection/normvein_roi_rotated_faster_rcnn_r50_fpn_e006.pth \
  --input path/to/finger.png \
  --output detections.json
```

Use `--device cpu`, `--device cuda`, or `--device auto`.

## Model implementation used by each script

Training and inference use the same model implementations from this repository:

| Entry point | Bundled implementation |
|---|---|
| `normvein recognize` | `normvein.models.build_model` → MobileFaceNet-Large, iResNet-100, or ConvNeXt-Small |
| `normvein segment` | `normvein.models.build_model` → local U-Net/ResNet-50 |
| `normvein detect` | `normvein.models.build_model` → the three local rotated detectors |
| `scripts/train_recognition.py` | the same recognition builders in `normvein.models` |
| `scripts/train_segmentation.py` | the same local U-Net/ResNet-50 implementation |
| `scripts/train_detection_*.py` | the same rotated detector builders in `normvein.models.detection` |

No external Ultralytics, MMDetection, or model-hub implementation is substituted
when loading the released weights.

## Train from scratch on FingerVeinSyn-5M

Download FingerVeinSyn-5M from its official
[repository](https://github.com/EvanWang98/FingerVeinSyn-5M). The expected layout is:

```text
FingerVeinSyn-5M/
├── raw_images/<identity>/*.png
├── roi_images/<identity>/*.png
├── annotations/<identity>/*.xml
├── shape_masks/<identity>/*.png
└── pattern_masks/<identity>/*.png
```

Prepare deterministic 500K recognition and detection subsets:

```bash
python scripts/prepare_recognition_subset.py \
  --data-root /data/FingerVeinSyn-5M \
  --identities 10000 \
  --images-per-identity 50 \
  --output data/normvein-recognition-500k.json

python scripts/prepare_detection_data.py \
  --dataset-root /data/FingerVeinSyn-5M \
  --out-dir data/detection-500k \
  --limit 500000
```

Recognition uses a 512-D embedding and additive angular-margin parameters
`(m1, m2, m3) = (1.0, 0.7, 0.0)`:

```bash
python scripts/train_recognition.py --data-root /data/FingerVeinSyn-5M/roi_images --manifest data/normvein-recognition-500k.json --model rec_mobilefacenet_large --epochs 200 --batch-size 16 --lr 0.01
python scripts/train_recognition.py --data-root /data/FingerVeinSyn-5M/roi_images --manifest data/normvein-recognition-500k.json --model rec_iresnet100 --epochs 100 --batch-size 16 --lr 0.01
python scripts/train_recognition.py --data-root /data/FingerVeinSyn-5M/roi_images --manifest data/normvein-recognition-500k.json --model rec_convnext_small --epochs 300 --batch-size 16 --lr 0.001
```

Segmentation:

```bash
python scripts/train_segmentation.py --data-root /data/FingerVeinSyn-5M --task finger-shape --epochs 10 --train-subset-ratio 0.1
python scripts/train_segmentation.py --data-root /data/FingerVeinSyn-5M --task vein-pattern --epochs 10 --train-subset-ratio 0.1
```

Rotated ROI detection:

```bash
python scripts/train_detection_faster_rcnn.py --ann-file data/detection-500k/annotations/train.json --image-root /data/FingerVeinSyn-5M/raw_images --epochs 10 --subset-ratio 1.0
python scripts/train_detection_ssd.py --ann-file data/detection-500k/annotations/train.json --image-root /data/FingerVeinSyn-5M/raw_images --epochs 10 --subset-ratio 1.0
python scripts/train_detection_yolo.py --ann-file data/detection-500k/annotations/train.json --image-root /data/FingerVeinSyn-5M/raw_images --epochs 10 --subset-ratio 1.0
```

All trainers default to one GPU. Increase `--gpus` and set
`--cuda-visible-devices` for distributed detector training.

## Reproducibility notes

- The available Faster R-CNN artifact is epoch 6; the detector schedule in the
  paper is 10 epochs.
- The available MobileFaceNet-Large, iResNet-100, and ConvNeXt-Small artifacts
  are epochs 50, 44, and 100; the paper schedules are 200, 100, and 300.
- Both downloaded segmentation checkpoints record `train_subset_ratio=0.01`.
  If the source was the complete 5M dataset, this would be about 50K samples;
  the paper describes 500K per task. The public training command above follows
  the paper's 500K setting (`0.1`).
- The recognition trainer uses a full 10K-class angular-margin head. The
  original multi-GPU experiments used distributed PartialFC, so optimization is
  not bitwise identical even though the backbone architectures are identical.

## Checkpoint format

```python
artifact = torch.load(path, map_location="cpu")
model_id = artifact["model_id"]
state_dict = artifact["state_dict"]
metadata = artifact["metadata"]
```

Only load PyTorch files from a trusted source and verify them against
`weights/manifest.json`.

## License

NormVein code and released model artifacts are provided under the
[MIT License](LICENSE). Third-party dependencies and FingerVeinSyn-5M retain
their respective licenses.
