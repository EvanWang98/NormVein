# NormVein

Official implementation and pretrained models for **NormVein: Normalizing
Finger-Vein Tasks with Large-Scale Synthetic Pretraining**.

NormVein uses large-scale synthetic data to initialize three families of
finger-vein tasks: rotated ROI detection, anatomical segmentation, and identity
recognition. The paper uses 500,000-image subsets from
[FingerVeinSyn-5M](https://github.com/EvanWang98/FingerVeinSyn-5M); release
metadata records any discrepancy found in an actual checkpoint.

> Release status: **candidate-v1**. The extracted files faithfully record the
> epochs found in the evaluated checkpoints. See [Model zoo](#model-zoo) before
> using them in a paper or benchmark.

![NormVein overview](assets/normvein_overview.png)

[中文说明](README_zh-CN.md) · [Model details](docs/MODELS.md) ·
[Dataset preparation](docs/DATA.md) · [Training from scratch](docs/TRAINING.md)

## Model zoo

The model files are attached to the GitHub Release rather than committed to Git.
After the first Release is published, download the eight files into the paths
shown below. `weights/manifest.json` contains the exact size and SHA-256 digest.

| Task | Model ID | Architecture | Actual epoch | Release file |
|---|---|---|---:|---|
| ROI detection | `roi_rotated_faster_rcnn_r50_fpn` | Rotated Faster R-CNN, R50-FPN | 6 | `weights/detection/normvein_roi_rotated_faster_rcnn_r50_fpn_e006.pth` |
| ROI detection | `roi_rotated_ssd_r50_fpn` | Rotated SSD, R50-FPN | 10 | `weights/detection/normvein_roi_rotated_ssd_r50_fpn_e010.pth` |
| ROI detection | `roi_rotated_yolo_r50_fpn` | Rotated YOLO-style, R50-FPN | 10 | `weights/detection/normvein_roi_rotated_yolo_r50_fpn_e010.pth` |
| Segmentation | `seg_finger_shape_unet_r50` | U-Net, R50 encoder | 10 | `weights/segmentation/normvein_seg_finger_shape_unet_r50_e010.pth` |
| Segmentation | `seg_vein_pattern_unet_r50` | U-Net, R50 encoder | 10 | `weights/segmentation/normvein_seg_vein_pattern_unet_r50_e010.pth` |
| Recognition | `rec_mobilefacenet_large` | MobileFaceNet-Large, 512-D | 50 | `weights/recognition/normvein_rec_mobilefacenet_large_e050.pth` |
| Recognition | `rec_iresnet100` | iResNet-100, 512-D | 44 | `weights/recognition/normvein_rec_iresnet100_e044.pth` |
| Recognition | `rec_convnext_small` | ConvNeXt-Small, 512-D | 100 | `weights/recognition/normvein_rec_convnext_small_e100.pth` |

The `roi_rotated_yolo_r50_fpn` checkpoint belongs to the custom ResNet-50-FPN
implementation in this repository; it is **not** an Ultralytics YOLOv8 checkpoint.

## Quick start

Python 3.9+ and PyTorch 2.x are recommended.

```bash
git clone https://github.com/<OWNER>/NormVein.git
cd NormVein
pip install -e .[all]
python scripts/verify_weights.py
```

Extract a 512-D recognition embedding:

```bash
normvein recognize \
  --model rec_iresnet100 \
  --checkpoint weights/recognition/normvein_rec_iresnet100_e044.pth \
  --input path/to/roi.png \
  --output embedding.npy
```

Segment a vein pattern or full finger shape:

```bash
normvein segment \
  --model seg_vein_pattern_unet_r50 \
  --checkpoint weights/segmentation/normvein_seg_vein_pattern_unet_r50_e010.pth \
  --input path/to/roi.png \
  --output vein_mask.png
```

Detect a rotated finger ROI:

```bash
normvein detect \
  --model roi_rotated_faster_rcnn_r50_fpn \
  --checkpoint weights/detection/normvein_roi_rotated_faster_rcnn_r50_fpn_e006.pth \
  --input path/to/finger.png \
  --output detections.json
```

All commands accept `--device cpu`, `--device cuda`, or `--device auto`.

## Train from scratch on synthetic data

1. Download FingerVeinSyn-5M from its official
   [repository](https://github.com/EvanWang98/FingerVeinSyn-5M). The dataset
   provides raw images, ROI images, XML annotations, finger-shape masks, and
   vein-pattern masks.
2. Prepare a deterministic 500K recognition subset (10,000 identities × 50
   images) and COCO-style rotated detection annotations.
3. Run the task-specific trainer.

```bash
python scripts/prepare_recognition_subset.py \
  --data-root /data/FingerVeinSyn-5M \
  --output data/normvein-recognition-500k.json

python training/detection/tools/prepare_data.py \
  --dataset-root /data/FingerVeinSyn-5M \
  --out-dir data/detection-500k

python scripts/train_recognition.py \
  --data-root /data/FingerVeinSyn-5M/roi_images \
  --manifest data/normvein-recognition-500k.json \
  --model rec_mobilefacenet_large --epochs 200

python scripts/train_segmentation.py \
  --data-root /data/FingerVeinSyn-5M --task vein-pattern --epochs 10

python training/detection/tools/train.py \
  --ann-file data/detection-500k/annotations/train.json \
  --image-root /data/FingerVeinSyn-5M/raw_images --epochs 10
```

The complete settings and commands for all eight models are in
[docs/TRAINING.md](docs/TRAINING.md). Dataset layout and subset rules are in
[docs/DATA.md](docs/DATA.md).

## Checkpoint format

Every Release asset is a model-only PyTorch artifact:

```python
artifact = torch.load(path, map_location="cpu")
state_dict = artifact["state_dict"]
print(artifact["model_id"], artifact["metadata"])
```

Only load PyTorch checkpoints from a trusted source. Verify downloaded files
against `weights/manifest.json` first.

## Citation

The paper identifier will be added after publication. Until then, use the
metadata in `CITATION.cff` and cite FingerVeinSyn-5M as requested by its authors.

## License

The public code and model license is pending author confirmation. Third-party
dependencies and FingerVeinSyn-5M remain under their respective licenses.

