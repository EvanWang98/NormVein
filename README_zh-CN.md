# NormVein

这是论文 **NormVein: Normalizing Finger-Vein Tasks with Large-Scale Synthetic Pretraining** 的官方代码与预训练模型仓库候选版。

NormVein 使用 [FingerVeinSyn-5M](https://github.com/EvanWang98/FingerVeinSyn-5M) 合成样本，对指静脉 ROI 检测、解剖结构分割和身份识别三类任务进行预训练。论文计划每个任务使用 50 万张；发布元数据会如实记录实际 checkpoint 中与此不一致的设置。

> 当前状态为 `candidate-v1`。文件名中的 epoch 是从实际下载权重中读取的真实轮次；与论文计划不一致的地方见 `docs/MODELS.md`。

![NormVein 总览](assets/normvein_overview.png)

## 快速使用

```bash
pip install -e .[all]
python scripts/verify_weights.py
```

提取 512 维识别特征：

```bash
normvein recognize \
  --model rec_iresnet100 \
  --checkpoint weights/recognition/normvein_rec_iresnet100_e044.pth \
  --input path/to/roi.png \
  --output embedding.npy
```

分割静脉纹理：

```bash
normvein segment \
  --model seg_vein_pattern_unet_r50 \
  --checkpoint weights/segmentation/normvein_seg_vein_pattern_unet_r50_e010.pth \
  --input path/to/roi.png \
  --output vein_mask.png
```

检测旋转 ROI：

```bash
normvein detect \
  --model roi_rotated_faster_rcnn_r50_fpn \
  --checkpoint weights/detection/normvein_roi_rotated_faster_rcnn_r50_fpn_e006.pth \
  --input path/to/finger.png \
  --output detections.json
```

8 个权重文件不会直接提交到 Git，而是作为 GitHub Release 附件发布。统一命名、文件大小和 SHA-256 校验值见 `weights/manifest.json`。

## 从头训练

先从官方仓库下载 FingerVeinSyn-5M。其目录包括 `raw_images`、`roi_images`、`annotations`、`shape_masks` 和 `pattern_masks`。

```bash
python scripts/prepare_recognition_subset.py \
  --data-root /data/FingerVeinSyn-5M \
  --output data/normvein-recognition-500k.json

python scripts/train_recognition.py \
  --data-root /data/FingerVeinSyn-5M/roi_images \
  --manifest data/normvein-recognition-500k.json \
  --model rec_mobilefacenet_large --epochs 200

python scripts/train_segmentation.py \
  --data-root /data/FingerVeinSyn-5M --task vein-pattern --epochs 10
```

检测数据转换和三种检测器的完整命令见 `docs/TRAINING.md`；数据目录与 50 万样本选择规则见 `docs/DATA.md`。

## 重要说明

- 当前 Faster R-CNN 发布候选是第 6 epoch，而不是论文计划的第 10 epoch。
- 当前 MobileFaceNet-Large、iResNet-100、ConvNeXt-S 权重分别来自第 50、44、100 epoch。
- 两个分割 checkpoint 记录的训练子集比例是 `0.01`；若源数据确为 5M，则约为 5 万张，与论文 50 万张描述不一致，需作者在正式发布前确认。
- `roi_rotated_yolo_r50_fpn` 是本仓库自定义 ResNet-50-FPN 检测器，不是 Ultralytics YOLOv8 权重。
- NormVein 代码与发布的模型参数采用 [MIT License](LICENSE)；第三方依赖和
  FingerVeinSyn-5M 仍遵循各自许可证。
