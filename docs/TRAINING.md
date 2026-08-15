# Training from scratch

Install the training dependencies and download FingerVeinSyn-5M first:

```bash
pip install -e .[all]
```

## Recognition

Prepare the 500K manifest as described in `DATA.md`, then run one of:

```bash
python scripts/train_recognition.py --data-root /data/FingerVeinSyn-5M/roi_images --manifest data/normvein-recognition-500k.json --model rec_mobilefacenet_large --epochs 200 --batch-size 16 --lr 0.01
python scripts/train_recognition.py --data-root /data/FingerVeinSyn-5M/roi_images --manifest data/normvein-recognition-500k.json --model rec_iresnet100 --epochs 100 --batch-size 16 --lr 0.01
python scripts/train_recognition.py --data-root /data/FingerVeinSyn-5M/roi_images --manifest data/normvein-recognition-500k.json --model rec_convnext_small --epochs 300 --batch-size 16 --lr 0.001
```

The trainer uses a 512-D embedding and the additive angular-margin tuple
`(m1, m2, m3) = (1.0, 0.7, 0.0)`. MB and R100 use SGD with momentum 0.9;
ConvNeXt-S uses AdamW. The bundled trainer uses a full 10K-class head, while the
original multi-GPU experiments used distributed PartialFC. This distinction can
change optimization dynamics and is not bitwise reproduction.

## Segmentation

```bash
python scripts/train_segmentation.py --data-root /data/FingerVeinSyn-5M --task finger-shape --epochs 10 --train-subset-ratio 0.1
python scripts/train_segmentation.py --data-root /data/FingerVeinSyn-5M --task vein-pattern --epochs 10 --train-subset-ratio 0.1
```

Both tasks use the local U-Net with a single-channel ResNet-50 encoder and a
BCE+Dice loss. The paper's 500K/5M subset ratio is `0.1`; use `0.01` only when
auditing the exact setting stored in the downloaded checkpoint.

## Rotated ROI detection

Prepare annotations first (see `DATA.md`). The three original research trainers
are preserved under `training/detection`:

```bash
python training/detection/tools/train.py --ann-file data/detection-500k/annotations/train.json --image-root /data/FingerVeinSyn-5M/raw_images --epochs 10
python training/detection/tools/train_ssd.py --ann-file data/detection-500k/annotations/train.json --image-root /data/FingerVeinSyn-5M/raw_images --epochs 10
python training/detection/tools/train_yolo.py --ann-file data/detection-500k/annotations/train.json --image-root /data/FingerVeinSyn-5M/raw_images --epochs 10
```

These are custom PyTorch implementations with a one-channel ResNet-50-FPN
backbone. The YOLO-style trainer does not produce an Ultralytics checkpoint.

## Turning a training checkpoint into a Release asset

Update the explicit `SPECS` table in `scripts/prepare_release_weights.py`, then:

```bash
python scripts/prepare_release_weights.py --source-root /path/to/original/project
python scripts/verify_weights.py
pytest
```

Never overwrite a stable Release asset under the same filename. If a different
epoch or tensor set is selected, update the epoch-bearing filename and manifest.

