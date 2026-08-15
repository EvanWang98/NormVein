# Model release notes

`candidate-v1` contains eight model-only artifacts. Each artifact stores a
`state_dict`, a stable `model_id`, the task and architecture, and provenance
metadata. `weights/manifest.json` is the machine-readable source of truth for
file names, sizes, epochs, parameter counts, and SHA-256 digests.

## Artifact provenance

| Model ID | Source checkpoint | Source epoch | Paper schedule | Parameters |
|---|---|---:|---:|---:|
| `roi_rotated_faster_rcnn_r50_fpn` | `ckpt/Det/epoch_6.pth` | 6 | 10 | 41,403,802 |
| `roi_rotated_ssd_r50_fpn` | `ckpt/Det/ssd_latest.pth` | 10 | 10 | 27,222,017 |
| `roi_rotated_yolo_r50_fpn` | `ckpt/Det/yolo_latest.pth` | 10 | 10 | 36,360,097 |
| `seg_finger_shape_unet_r50` | `ckpt/Seg/FingerShape.pt` | 10 | 10 | 43,911,614 |
| `seg_vein_pattern_unet_r50` | `ckpt/Seg/FingerPattern.pt` | 10 | 10 | 43,911,614 |
| `rec_mobilefacenet_large` | rank-0 `checkpoint_gpu_0MB_500k.pt` | 50 | 200 | 6,363,099 |
| `rec_iresnet100` | rank-0 `checkpoint_gpu_0R100_500k.pt` | 44 | 100 | 78,069,850 |
| `rec_convnext_small` | rank-0 `checkpoint_gpu_0vit_500k.pt` | 100 | 300 | 49,846,881 |

## Items to resolve before a stable release

1. The Faster R-CNN file is epoch 6 although the detector schedule is 10
   epochs. Replace it with the evaluated epoch-10/best file if one exists.
2. The recognition files stop at epochs 50, 44, and 100, while the paper lists
   schedules of 200, 100, and 300. Confirm that these are the exact files used
   to obtain the reported tables, or recover the final checkpoints.
3. Both segmentation checkpoints record `train_subset_ratio=0.01`. If their
   source is the published 5M dataset, that would mean about 50K samples rather
   than the paper's 500K. Confirm the source dataset size and sampling procedure.
4. The custom `rotated_yolo` implementation is a ResNet-50-FPN YOLO-style
   detector. It must not be advertised as an Ultralytics YOLOv8-OBB checkpoint.
5. `checkpoint_gpu_1vit_500k.pt` is corrupt (missing the PyTorch archive central
   directory). The release uses rank 0. Healthy rank files differ only in
   BatchNorm running statistics; trainable tensors are identical.

## Why model-only artifacts

The source training checkpoints also contain optimizer, scheduler, AMP scaler,
and rank-local distributed state. The Release files deliberately contain only
the model state and public metadata. This reduces file size, avoids coupling
inference to the original trainer, and makes the format consistent across all
tasks. Source files are retained outside the public repository.

