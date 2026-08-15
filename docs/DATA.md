# FingerVeinSyn-5M data preparation

Use the official [FingerVeinSyn-5M repository](https://github.com/EvanWang98/FingerVeinSyn-5M)
for dataset access and licensing information. It describes five million PNG
samples from 50,000 identities, with 100 samples per identity at 300 × 600
pixels. XML annotations and pixel masks support all NormVein tasks.

Expected layout:

```text
FingerVeinSyn-5M/
├── raw_images/<identity>/*.png
├── roi_images/<identity>/*.png
├── annotations/<identity>/*.xml
├── shape_masks/<identity>/*.png
└── pattern_masks/<identity>/*.png
```

The upstream repository may distribute the large files through Kaggle. Do not
commit dataset images to this repository.

## Recognition subset

The paper specifies 10,000 identities and 50 images per identity. Generate an
explicit manifest so every run selects the same files:

```bash
python scripts/prepare_recognition_subset.py \
  --data-root /data/FingerVeinSyn-5M \
  --identities 10000 \
  --images-per-identity 50 \
  --output data/normvein-recognition-500k.json
```

By default, identities and images are sorted lexicographically and the first
requested entries are selected. Archive this manifest with experiment logs.

## Detection subset

The XML-to-COCO converter preserves the extra `rotation_angle` field expected by
the custom detectors:

```bash
python training/detection/tools/prepare_data.py \
  --dataset-root /data/FingerVeinSyn-5M \
  --out-dir data/detection-500k \
  --workers 16 \
  --limit 500000
```

Record whether the `--limit` is applied before or after shuffling. The bundled
converter uses sorted XML paths, so it is deterministic.

## Segmentation subset

Full-finger shape uses `raw_images` paired with `shape_masks` at 300 × 600.
Vein-pattern segmentation uses `roi_images` paired with `pattern_masks` at
100 × 300 and percentile normalization `(10, 90)`.

The paper says 500K samples per task, which is 10% of 5M. The downloaded
segmentation checkpoints instead record `train_subset_ratio=0.01`; this
provenance discrepancy must be resolved before claiming exact reproduction.

