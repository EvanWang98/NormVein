import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler


TOOL_DIR = Path(__file__).resolve().parent
DETECTOR_DIR = TOOL_DIR.parent
PROJECT_DIR = DETECTOR_DIR.parent


def ensure_import_path():
    for item in [str(PROJECT_DIR), str(DETECTOR_DIR)]:
        if item and item not in sys.path:
            sys.path.insert(0, item)


def resolve_path(path):
    path = Path(path)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([DETECTOR_DIR / path, PROJECT_DIR / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed():
    if not is_distributed():
        return 0, 0, 1
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_main_process():
    return not is_distributed() or torch.distributed.get_rank() == 0


def reduce_losses(losses):
    reduced = {key: value.detach() for key, value in losses.items()}
    if not is_distributed():
        return reduced
    keys = sorted(reduced.keys())
    values = torch.stack([reduced[key] for key in keys])
    torch.distributed.all_reduce(values)
    values /= torch.distributed.get_world_size()
    return {key: value for key, value in zip(keys, values)}


def move_target_to_device(target, device):
    moved = {}
    for key, value in target.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


class RandomSubsetDistributedSampler(Sampler):
    def __init__(self, dataset, subset_ratio=1.0, shuffle=True, seed=2048, rank=0, world_size=1):
        self.dataset = dataset
        self.subset_ratio = float(subset_ratio)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        self.dataset_size = len(dataset)
        self.subset_size = max(1, int(math.ceil(self.dataset_size * min(max(self.subset_ratio, 0.0), 1.0))))
        self.num_samples = int(math.ceil(self.subset_size / self.world_size))
        self.total_size = self.num_samples * self.world_size

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_size, generator=generator).tolist()[: self.subset_size]
        if self.shuffle:
            order = torch.randperm(len(indices), generator=generator).tolist()
            indices = [indices[i] for i in order]
        if len(indices) < self.total_size:
            repeat = math.ceil((self.total_size - len(indices)) / max(1, len(indices)))
            indices += (indices * repeat)[: self.total_size - len(indices)]
        indices = indices[self.rank : self.total_size : self.world_size]
        return iter(indices)

    def __len__(self):
        return self.num_samples


def build_dataloader(args, rank, world_size):
    ensure_import_path()
    from detector_pretrain.datasets import FVSynRotatedCocoDataset, collate_fn

    dataset = FVSynRotatedCocoDataset(
        ann_file=args.ann_file,
        image_root=args.image_root,
        image_size=(args.image_width, args.image_height),
        angle_sign=args.angle_sign,
        train=True,
        random_flip=not args.no_random_flip,
        root_dir=DETECTOR_DIR,
    )
    sampler = RandomSubsetDistributedSampler(
        dataset,
        subset_ratio=args.subset_ratio,
        shuffle=True,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    return dataset, sampler, loader


def build_model(args, device):
    ensure_import_path()
    from detector_pretrain.models import build_rotated_yolo

    weights_path = resolve_path(args.pretrained_backbone)
    model = build_rotated_yolo(
        image_size=(args.image_width, args.image_height),
        pretrained_backbone_path=str(weights_path) if weights_path else None,
        trainable_backbone_layers=args.trainable_backbone_layers,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        detections_per_img=args.detections_per_img,
        bbox_loss_weight=args.bbox_loss_weight,
        obj_loss_weight=args.obj_loss_weight,
        cls_loss_weight=args.cls_loss_weight,
        pre_nms_topk=args.pre_nms_topk,
        center_radius=args.center_radius,
    )
    return model.to(device)

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, args):
    raw_model = model.module if hasattr(model, "module") else model
    payload = {
        "epoch": epoch,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    checkpoint = torch.load(str(path), map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("epoch", 0))


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args):
    model.train()
    start = time.time()
    running = {}
    for step, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [move_target_to_device(target, device) for target in targets]

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp):
            losses = model(images, targets)
            loss = sum(value for value in losses.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch={epoch} step={step}: {loss.item()}")

        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        reduced = reduce_losses(losses)
        for key, value in reduced.items():
            running[key] = running.get(key, 0.0) + float(value.item())

        if is_main_process() and (step % args.log_interval == 0 or step == len(loader)):
            count = float(step)
            stats = "  ".join(f"{key}={running[key] / count:.4f}" for key in sorted(running))
            lr = optimizer.param_groups[0]["lr"]
            print(f"epoch={epoch} step={step}/{len(loader)} lr={lr:.6g} {stats}", flush=True)

    elapsed = time.time() - start
    return {key: value / max(1, len(loader)) for key, value in running.items()}, elapsed


def launch_distributed_if_needed(args):
    if is_distributed() or args.gpus <= 1:
        return False
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    pythonpath = str(PROJECT_DIR)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={args.gpus}",
        f"--master_port={args.master_port}",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    if args.dry_run:
        print("CUDA_VISIBLE_DEVICES=" + args.cuda_visible_devices + " PYTHONPATH=" + pythonpath + " " + " ".join(cmd))
        return True
    raise SystemExit(subprocess.call(cmd, env=env))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", default="./tools/data/fvsyn50k_coco/annotations/train.json")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default="./work_dirs/rotated_yolo")
    parser.add_argument("--pretrained-backbone", default="./weight/resnet50-0676ba61.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--subset-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--lr-step", type=int, default=7)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--clip-grad-norm", type=float, default=10.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--cuda-visible-devices", default="2,3,4,5")
    parser.add_argument("--master-port", type=int, default=15658)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-width", type=int, default=600)
    parser.add_argument("--image-height", type=int, default=300)
    parser.add_argument("--angle-sign", type=float, default=-1.0)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--nms-thresh", type=float, default=0.5)
    parser.add_argument("--detections-per-img", type=int, default=100)
    parser.add_argument("--bbox-loss-weight", type=float, default=1.0)
    parser.add_argument("--trainable-backbone-layers", type=int, default=None)
    parser.add_argument("--no-random-flip", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--obj-loss-weight", type=float, default=1.0)
    parser.add_argument("--cls-loss-weight", type=float, default=0.5)
    parser.add_argument("--pre-nms-topk", type=int, default=1000)
    parser.add_argument("--center-radius", type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    if launch_distributed_if_needed(args):
        return

    rank, local_rank, world_size = setup_distributed()
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if is_distributed() else "cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    output_dir = resolve_path(args.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"model: rotated_yolo", flush=True)
        print(f"ann_file: {resolve_path(args.ann_file)}", flush=True)
        print(f"output_dir: {output_dir}", flush=True)
        print(f"world_size: {world_size}, batch_size_per_gpu: {args.batch_size}, subset_ratio: {args.subset_ratio}", flush=True)
        print(f"score_thresh: {args.score_thresh}, nms_thresh: {args.nms_thresh}, detections_per_img: {args.detections_per_img}", flush=True)
        print(f"obj_loss_weight: {args.obj_loss_weight}, cls_loss_weight: {args.cls_loss_weight}", flush=True)
        print(f"pre_nms_topk: {args.pre_nms_topk}, center_radius: {args.center_radius}", flush=True)

    dataset, sampler, loader = build_dataloader(args, rank, world_size)
    model = build_model(args, device)
    if is_distributed():
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    scaler = GradScaler(enabled=args.amp)

    start_epoch = 1
    if args.resume:
        loaded_epoch = load_checkpoint(resolve_path(args.resume), model.module if hasattr(model, "module") else model, optimizer, scheduler, scaler, device)
        start_epoch = loaded_epoch + 1
        if is_main_process():
            print(f"resumed from {args.resume}, start_epoch={start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        sampler.set_epoch(epoch)
        stats, elapsed = train_one_epoch(model, loader, optimizer, scaler, device, epoch, args)
        scheduler.step()
        if is_main_process():
            stats_text = "  ".join(f"{key}={value:.4f}" for key, value in sorted(stats.items()))
            print(f"epoch={epoch} done elapsed={elapsed:.1f}s {stats_text}", flush=True)
            save_checkpoint(output_dir / f"epoch_{epoch}.pth", model, optimizer, scheduler, scaler, epoch, args)
            save_checkpoint(output_dir / "latest.pth", model, optimizer, scheduler, scaler, epoch, args)
        if is_distributed():
            torch.distributed.barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
