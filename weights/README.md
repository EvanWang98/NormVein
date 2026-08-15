# NormVein release weights

Model binaries are ignored by Git and published as GitHub Release assets. The
machine-readable `manifest.json` records the exact file name, byte size,
SHA-256 digest, source checkpoint, source epoch, and parameter count.

| Task | Release file |
|---|---|
| Rotated Faster R-CNN | `detection/normvein_roi_rotated_faster_rcnn_r50_fpn_e006.pth` |
| Rotated SSD | `detection/normvein_roi_rotated_ssd_r50_fpn_e010.pth` |
| Rotated YOLO-style | `detection/normvein_roi_rotated_yolo_r50_fpn_e010.pth` |
| Finger-shape U-Net | `segmentation/normvein_seg_finger_shape_unet_r50_e010.pth` |
| Vein-pattern U-Net | `segmentation/normvein_seg_vein_pattern_unet_r50_e010.pth` |
| MobileFaceNet-Large | `recognition/normvein_rec_mobilefacenet_large_e050.pth` |
| iResNet-100 | `recognition/normvein_rec_iresnet100_e044.pth` |
| ConvNeXt-Small | `recognition/normvein_rec_convnext_small_e100.pth` |

Verify all local assets before inference or upload:

```bash
python scripts/verify_weights.py
```

These are model-only artifacts produced by `scripts/prepare_release_weights.py`.
The original full trainer checkpoints remain private and are never committed.
The custom rotated YOLO-style artifact is not an Ultralytics YOLOv8 checkpoint.
See `docs/MODELS.md` for known provenance discrepancies.
