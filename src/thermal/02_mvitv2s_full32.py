from __future__ import annotations

import gc
import importlib.util
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s
from tqdm.auto import tqdm

# Keep the notebook self-contained in a single ordinary Python cell.
for package in ("ultralytics", "timm"):
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

Image.MAX_IMAGE_PIXELS = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    thermal_train_root: Path = Path("/kaggle/input/datasets/zhuowamg/thermal/Thermal")
    ir_train_root: Path = Path("/kaggle/input/datasets/zhuowamg/irdata/IR")
    test_root: Path = Path("/kaggle/input/datasets/zhuowamg/cuhksmall/CUHK-X_dataset/Small-Model-Track/Testing/data/small_model_track_test/small_model_track_test")
    test_csv: Path = Path("/kaggle/input/datasets/zhuowamg/cuhksmall/CUHK-X_dataset/Small-Model-Track/Testing/test_file/test.csv")
    class_map_csv: Path = Path("/kaggle/input/datasets/zhuowamg/class-mapping1/class_mapping.csv")
    work_root: Path = Path("/kaggle/working")
    cache_root: Path = Path("/kaggle/temp/thermal_r2p18_cache")

    cache_frames: int = 32
    model_frames: int = 32
    img_size: int = 128
    in_channels: int = 3  # single-view v1: cropped Thermal RGB only
    cache_workers: int = 4

    person_det_frames: int = 8
    person_det_conf: float = 0.25
    person_crop_margin: float = 1.60
    person_crop_min_side: float = 0.45

    n_folds: int = 5
    selection_fold: int = 0
    seed: int = 2026

    max_selection_epochs: int = 30
    full_train_updates: int = 3740
    schedule_total_steps: int = 5610
    schedule_warmup_steps: int = 374
    batch_size: int = 2
    grad_accum: int = 6  # LayerNorm makes physical batch 2 safe; effective batch remains 12
    lr: float = 1e-4
    min_lr: float = 1e-6
    warmup_epochs: int = 2
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    label_smoothing: float = 0.05
    ema_decay: float = 0.995
    amp: bool = True
    num_workers: int = 2

    dropout: float = 0.30
    hflip_p: float = 0.5
    crop_scale: tuple[float, float] = (0.80, 1.0)
    calibration_p: float = 0.80
    brightness_delta: float = 0.08
    contrast_delta: float = 0.10
    mixup_p: float = 0.30
    mixup_alpha: float = 0.20
    boundary_users: tuple[str, ...] = ("user8", "user9", "user23", "user24")
    tta_hflip: bool = True

    thermal_target_mib: float = 30.0
    thermal_soft_ceiling_mib: float = 35.0
    other_models_mib: float = 61.0  # IR/Depth 60 + IMU 1; YOLO is measured separately.

    @property
    def cache_dir(self):
        # Cache contents depend on cache_frames, not on the number of frames
        # consumed by the model. Keep the old path so the 32-frame cache can be reused.
        return self.cache_root / f"Tcache{self.cache_frames}_Tmodel16_S{self.img_size}_IRcrop_segment_v1"

    @property
    def out_dir(self):
        return self.work_root / "thermal_full32_calibration_mixup_mvitv2s_boundary"


CFG = Config()
CFG.cache_dir.mkdir(parents=True, exist_ok=True)
CFG.out_dir.mkdir(parents=True, exist_ok=True)
for required in (CFG.thermal_train_root, CFG.ir_train_root, CFG.test_root, CFG.test_csv, CFG.class_map_csv):
    assert required.exists(), f"Missing input: {required}"
print("torch", torch.__version__, "| device", DEVICE)
print("cache", CFG.cache_dir)
print(f"experiment: full {CFG.model_frames}-frame input, physical batch={CFG.batch_size}, "
      f"gradient accumulation={CFG.grad_accum}, effective batch={CFG.batch_size * CFG.grad_accum}")
print(f"clip-consistent calibration augmentation: p={CFG.calibration_p:.2f}, "
      f"brightness=+/-{CFG.brightness_delta:.2f}, contrast=+/-{CFG.contrast_delta:.2f}")
print(f"video MixUp: p={CFG.mixup_p:.2f}, alpha={CFG.mixup_alpha:.2f}, "
      "lambda=max(lambda, 1-lambda)")
print(f"boundary validation users: {CFG.boundary_users}")
print("backbone: Torchvision MViTv2-S, Kinetics-400 pretrained, LayerNorm (no precise-BN)")
print(f"fixed full training: {CFG.full_train_updates} updates, "
      f"cosine schedule={CFG.schedule_total_steps}, warmup={CFG.schedule_warmup_steps}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


_LAST_NUMBER = re.compile(r"(\d+)(?=\.[^.]+$)")


def frame_files(folder: Path):
    if not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    def key(p):
        m = _LAST_NUMBER.search(p.name)
        return (0, int(m.group(1))) if m else (1, p.name)
    return [str(p) for p in sorted(files, key=key)]


@dataclass
class Clip:
    clip_id: str
    action_name: str
    action_id: int
    user: str
    trial: str
    thermal: list[str]
    ir: list[str]


class_map = pd.read_csv(CFG.class_map_csv, encoding="utf-8-sig")
class_map["action_id"] = class_map["action_id"].astype(int)
LABEL_VALUES = np.sort(class_map["action_id"].unique())
N_CLASSES = len(LABEL_VALUES)
ID_TO_INDEX = {int(v): i for i, v in enumerate(LABEL_VALUES)}
NAME_TO_ID = dict(zip(class_map["action_name"].astype(str), class_map["action_id"]))
assert N_CLASSES == 40, f"Expected 40 classes, got {N_CLASSES}"


def build_train_index(cfg):
    clips = []
    for trial_dir in tqdm(sorted(cfg.thermal_train_root.glob("*/*/*")), desc="index Thermal"):
        if not trial_dir.is_dir():
            continue
        action, user, trial = trial_dir.parts[-3:]
        if action not in NAME_TO_ID:
            continue
        thermal = frame_files(trial_dir)
        if not thermal:
            continue
        ir = frame_files(cfg.ir_train_root / action / user / trial)
        clips.append(Clip(f"{action}/{user}/{trial}", action, int(NAME_TO_ID[action]), user, trial, thermal, ir))
    if not clips:
        raise RuntimeError("No Thermal training clips found; check thermal_train_root")
    return clips


def build_test_index(cfg):
    clips = []
    for path in pd.read_csv(cfg.test_csv)["path"]:
        clip_id = str(path).strip("/").split("/")[-1]
        root = cfg.test_root / clip_id
        clips.append(Clip(clip_id, "", -1, "test", "", frame_files(root / "Thermal"), frame_files(root / "IR")))
    return clips


train_clips = build_train_index(CFG)
test_clips = build_test_index(CFG)
print(f"train={len(train_clips)} clips, users={len({c.user for c in train_clips})}, test={len(test_clips)}")
print(f"train clips without matched IR: {sum(not c.ir for c in train_clips)}")
print(f"test clips without IR/Thermal: IR={sum(not c.ir for c in test_clips)}, "
      f"Thermal={sum(not c.thermal for c in test_clips)}")

raw_lengths = np.asarray([len(c.thermal) for c in train_clips])
names = ["min", "p10", "p25", "median", "p75", "p90", "max"]
values = np.percentile(raw_lengths, [0, 10, 25, 50, 75, 90, 100])
print("\nRaw Thermal frame-count distribution:")
print(dict(zip(names, np.rint(values).astype(int).tolist())))
print("\nAll classes ordered by action_id:")
print(class_map.sort_values("action_id")[["action_id", "action_name"]].to_string(index=False))
direction_words = re.compile(r"sit|stand|pick|put|open|close|enter|leave|walk|lie|rise|get[_ ]?up|wear|take[_ ]?off|turn[_ ]?on|turn[_ ]?off", re.I)
directional = class_map[class_map.action_name.astype(str).str.contains(direction_words, regex=True)]
print("\nPotential direction-sensitive classes (heuristic; inspect manually):")
print(directional.sort_values("action_id")[["action_id", "action_name"]].to_string(index=False) if len(directional) else "none found by keyword heuristic")


def pick_indices(n_src, n_out):
    if n_src <= 0:
        return np.zeros(0, dtype=int)
    return np.linspace(0, n_src - 1, n_out).round().astype(int)


def ensure_yolo(cfg):
    from ultralytics import YOLO
    target = cfg.work_root / "yolo11n.pt"
    if target.exists():
        return target
    old_cwd = Path.cwd()
    os.chdir(cfg.work_root)
    try:
        model = YOLO("yolo11n.pt")
        source = Path(getattr(model, "ckpt_path", "yolo11n.pt"))
        if source.exists() and source.resolve() != target.resolve():
            target.write_bytes(source.read_bytes())
    finally:
        os.chdir(old_cwd)
    if not target.exists():
        for candidate in (old_cwd / "yolo11n.pt", Path("/root/yolo11n.pt")):
            if candidate.exists():
                target.write_bytes(candidate.read_bytes())
                break
    assert target.exists(), "YOLO download did not produce /kaggle/working/yolo11n.pt"
    return target


YOLO_PATH = ensure_yolo(CFG)
print(f"YOLO: {YOLO_PATH.stat().st_size / 1024**2:.2f} MiB")


def crop_window(boxes, width, height, cfg):
    x0, y0 = boxes[:, 0].min() / width, boxes[:, 1].min() / height
    x1, y1 = boxes[:, 2].max() / width, boxes[:, 3].max() / height
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side_px = max((x1 - x0) * width, (y1 - y0) * height) * cfg.person_crop_margin
    side_px = max(side_px, cfg.person_crop_min_side * max(width, height))
    hx, hy = side_px / width / 2, side_px / height / 2
    return max(cx - hx, 0.0), max(cy - hy, 0.0), min(cx + hx, 1.0), min(cy + hy, 1.0)


def detect_windows(clips, split, cfg):
    cache_path = cfg.cache_dir / f"{split}_crop_windows.parquet"
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if cached["clip_id"].tolist() == [c.clip_id for c in clips]:
            print(f"[{split}] reusing {cache_path}")
            return cached

    from ultralytics import YOLO
    detector = YOLO(str(YOLO_PATH))
    device = 0 if DEVICE.type == "cuda" else "cpu"
    rows = []
    for clip in tqdm(clips, desc=f"YOLO IR crops [{split}]"):
        probe = [clip.ir[i] for i in sorted(set(pick_indices(len(clip.ir), cfg.person_det_frames).tolist()))] if clip.ir else []
        boxes, width, height = [], None, None
        if probe:
            try:
                results = detector.predict(probe, classes=[0], conf=cfg.person_det_conf, verbose=False, device=device)
                for result in results:
                    height, width = result.orig_shape
                    if len(result.boxes):
                        boxes.append(result.boxes.xyxy[result.boxes.conf.argmax()].detach().cpu().numpy())
            except Exception:
                boxes = []
        if boxes:
            win = crop_window(np.asarray(boxes), width, height, cfg)
            rows.append({"clip_id": clip.clip_id, "has_crop": True, "x0": win[0], "y0": win[1], "x1": win[2], "y1": win[3]})
        else:
            rows.append({"clip_id": clip.clip_id, "has_crop": False, "x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0})
    out = pd.DataFrame(rows)
    out.to_parquet(cache_path, index=False)
    print(f"[{split}] YOLO crop success: {out['has_crop'].mean():.1%}")
    del detector
    torch.cuda.empty_cache()
    return out


train_windows = detect_windows(train_clips, "train", CFG)
test_windows = detect_windows(test_clips, "test", CFG)


def show_crop_alignment(clips, windows, split, n=4, seed=0):
    """Visual QA: IR source box, its mapped Thermal box, and the actual crop."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    candidates = [i for i, c in enumerate(clips) if c.thermal and c.ir]
    if not candidates:
        print(f"[{split}] no paired IR/Thermal clips available for crop QA")
        return
    picks = np.random.default_rng(seed).choice(candidates, min(n, len(candidates)), replace=False)
    fig, axes = plt.subplots(len(picks), 3, figsize=(10, 3 * len(picks)), squeeze=False)
    for row, idx in enumerate(picks):
        clip = clips[idx]
        win_row = windows.iloc[idx]
        win = (win_row.x0, win_row.y0, win_row.x1, win_row.y1)
        ir = Image.open(clip.ir[len(clip.ir) // 2]).convert("L")
        thermal = Image.open(clip.thermal[len(clip.thermal) // 2]).convert("RGB")
        for ax, image, title in ((axes[row, 0], ir, "IR + detected window"),
                                 (axes[row, 1], thermal, "Thermal + mapped window")):
            ax.imshow(image, cmap="gray" if image.mode == "L" else None)
            width, height = image.size
            ax.add_patch(Rectangle((win[0] * width, win[1] * height),
                                   (win[2] - win[0]) * width, (win[3] - win[1]) * height,
                                   fill=False, ec="yellow", lw=2))
            ax.set_title(title)
        width, height = thermal.size
        mapped = thermal.crop((round(win[0] * width), round(win[1] * height),
                               round(win[2] * width), round(win[3] * height)))
        axes[row, 2].imshow(mapped)
        axes[row, 2].set_title(f"actual Thermal crop\n{clip.clip_id}")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    print(f"[{split}] QA criterion: full person, hands, and nearby interaction objects should remain inside the Thermal crop.")


show_crop_alignment(train_clips, train_windows, "train", n=4, seed=0)
show_crop_alignment(test_clips, test_windows, "test", n=4, seed=1)


_MM = None
_CACHE_ARGS = None


def _worker_init(path, shape, size):
    global _MM, _CACHE_ARGS
    _MM = np.memmap(path, dtype=np.uint8, mode="r+", shape=shape)
    _CACHE_ARGS = {"size": size, "frames": shape[1]}


def _read_thermal(path, size, win):
    """Decode the single cropped Thermal RGB view used by v1."""
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if win is not None:
                width, height = image.size
                image = image.crop((round(win[0] * width), round(win[1] * height),
                                    round(win[2] * width), round(win[3] * height)))
            cropped = np.asarray(image.resize((size, size), Image.BILINEAR), dtype=np.uint8)
        return None if cropped.max() == 0 else cropped
    except Exception:
        return None


def _cache_one(job):
    row, clip, win = job
    buf = np.zeros(_MM.shape[1:], dtype=np.uint8)
    bad = _CACHE_ARGS["frames"] if not clip.thermal else 0
    for t, source_idx in enumerate(pick_indices(len(clip.thermal), _CACHE_ARGS["frames"])):
        cropped = _read_thermal(clip.thermal[source_idx], _CACHE_ARGS["size"], win)
        if cropped is None:
            bad += 1
        else:
            buf[t] = cropped.transpose(2, 0, 1)
    _MM[row] = buf
    return {"row": row, "clip_id": clip.clip_id, "action_name": clip.action_name,
            "action_id": clip.action_id, "user": clip.user, "trial": clip.trial,
            "bad_frames": bad, "has_crop": win is not None}


def build_cache(clips, windows, split, cfg):
    npy = cfg.cache_dir / f"{split}_thermal.npy"
    meta_path = cfg.cache_dir / f"{split}_meta.parquet"
    shape = (len(clips), cfg.cache_frames, cfg.in_channels, cfg.img_size, cfg.img_size)
    expected = int(np.prod(shape))
    if npy.exists() and meta_path.exists() and npy.stat().st_size == expected:
        meta = pd.read_parquet(meta_path)
        if meta["clip_id"].tolist() == [c.clip_id for c in clips]:
            print(f"[{split}] reusing frame cache")
            return meta
    print(f"[{split}] cache shape={shape}, size={expected / 1024**3:.2f} GiB")
    np.memmap(npy, dtype=np.uint8, mode="w+", shape=shape).flush()
    wins = [(r.x0, r.y0, r.x1, r.y1) if r.has_crop else None for r in windows.itertuples(index=False)]
    jobs = zip(range(len(clips)), clips, wins)
    records = []
    with Pool(cfg.cache_workers, initializer=_worker_init, initargs=(str(npy), shape, cfg.img_size)) as pool:
        for record in tqdm(pool.imap_unordered(_cache_one, jobs, chunksize=8), total=len(clips), desc=f"cache {split}"):
            records.append(record)
    meta = pd.DataFrame(records).sort_values("row").reset_index(drop=True)
    meta.to_parquet(meta_path, index=False)
    print(f"[{split}] unreadable frames={meta.bad_frames.sum()}, crop coverage={meta.has_crop.mean():.1%}")
    return meta


train_meta = build_cache(train_clips, train_windows, "train", CFG)
test_meta = build_cache(test_clips, test_windows, "test", CFG)
train_meta["label_idx"] = train_meta["action_id"].map(ID_TO_INDEX).astype(int)
test_meta["label_idx"] = -1


def assign_folds(meta, cfg):
    folds = np.full(len(meta), -1, dtype=int)
    splitter = StratifiedGroupKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed)
    for fold, (_, val_idx) in enumerate(splitter.split(meta, meta.label_idx, groups=meta.user)):
        folds[val_idx] = fold
    assert (folds >= 0).all()
    return folds


boundary_count = int(train_meta.user.isin(CFG.boundary_users).sum())
print(f"boundary split: train={len(train_meta)-boundary_count}, val={boundary_count}, "
      f"users={list(CFG.boundary_users)}")

KINETICS_MEAN = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
KINETICS_STD = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1)


class ThermalDataset(Dataset):
    def __init__(self, npy_path, meta, cfg, train):
        self.path, self.meta, self.cfg, self.train = str(npy_path), meta.reset_index(drop=True), cfg, train
        self.rows = self.meta.row.to_numpy()
        self.labels = self.meta.label_idx.to_numpy()
        per_row = cfg.cache_frames * cfg.in_channels * cfg.img_size * cfg.img_size
        n_rows, rem = divmod(os.path.getsize(self.path), per_row)
        assert rem == 0
        self.shape = (n_rows, cfg.cache_frames, cfg.in_channels, cfg.img_size, cfg.img_size)
        self.segment_edges = np.linspace(0, cfg.cache_frames, cfg.model_frames + 1).round().astype(int)
        assert np.all(np.diff(self.segment_edges) > 0), "cache_frames must provide at least one frame per segment"
        self._mm = None

    @property
    def mm(self):
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=np.uint8, mode="r", shape=self.shape)
        return self._mm

    def __len__(self):
        return len(self.meta)

    def temporal_indices(self):
        # This experiment deliberately exposes the complete ordered 32-frame
        # sequence to the 3D backbone in a single forward pass.
        indices = np.arange(self.cfg.cache_frames, dtype=np.int64)
        assert len(indices) == self.cfg.model_frames
        return indices

    def augment(self, clip):
        if random.random() < self.cfg.hflip_p:
            clip = torch.flip(clip, [-1])
        lo, hi = self.cfg.crop_scale
        scale = random.uniform(lo, hi)
        side = int(round(self.cfg.img_size * scale))
        top = random.randint(0, self.cfg.img_size - side)
        left = random.randint(0, self.cfg.img_size - side)
        clip = F.interpolate(clip[:, :, top:top + side, left:left + side],
                             size=(self.cfg.img_size, self.cfg.img_size), mode="bilinear", align_corners=False)
        if random.random() < self.cfg.calibration_p:
            # One scalar brightness and contrast pair is shared by every frame
            # and RGB channel. This preserves temporal consistency and does not
            # rotate the pseudo-colour hue relationships.
            contrast = 1.0 + random.uniform(-self.cfg.contrast_delta, self.cfg.contrast_delta)
            brightness = random.uniform(-self.cfg.brightness_delta, self.cfg.brightness_delta)
            clip = ((clip - 0.5) * contrast + 0.5 + brightness).clamp_(0.0, 1.0)
        return clip

    def __getitem__(self, idx):
        # Both training and validation consume the complete ordered 32-frame clip.
        frame_idx = self.temporal_indices()
        clip = torch.from_numpy(np.asarray(self.mm[self.rows[idx], frame_idx]).copy()).float().div_(255.0)
        if self.train:
            clip = self.augment(clip)
        clip = clip.sub_(KINETICS_MEAN).div_(KINETICS_STD)
        return clip, int(self.labels[idx]), idx


def build_model(pretrained, cfg):
    weights = MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None
    model = mvit_v2_s(weights=weights)

    # Torchvision stores the native 16x224 patch grid (8x56x56) as metadata
    # used by forward(). MViTv2 uses relative positional embeddings, so the
    # learned tables remain valid and are interpolated after updating this
    # metadata to our actual post-convolution grid (16x32x32 for 32x128 input).
    conv = model.conv_proj
    def conv_out(size, kernel, stride, padding, dilation):
        return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    temporal = conv_out(cfg.model_frames, conv.kernel_size[0], conv.stride[0],
                        conv.padding[0], conv.dilation[0])
    height = conv_out(cfg.img_size, conv.kernel_size[1], conv.stride[1],
                      conv.padding[1], conv.dilation[1])
    width = conv_out(cfg.img_size, conv.kernel_size[2], conv.stride[2],
                     conv.padding[2], conv.dilation[2])
    model.pos_encoding.temporal_size = temporal
    model.pos_encoding.spatial_size = (height, width)

    assert isinstance(model.head, nn.Sequential) and isinstance(model.head[-1], nn.Linear)
    in_features = model.head[-1].in_features
    model.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(in_features, N_CLASSES))
    return model


probe_model = build_model(False, CFG)
params = sum(p.numel() for p in probe_model.parameters())
print(f"MViTv2-S: {params / 1e6:.2f}M params, estimated INT8 storage {params / 1024**2:.2f} MiB")
print(f"MViTv2-S patch grid: {probe_model.pos_encoding.temporal_size}x"
      f"{probe_model.pos_encoding.spatial_size[0]}x{probe_model.pos_encoding.spatial_size[1]}")
probe_model.eval()
with torch.no_grad():
    probe_output = probe_model(torch.zeros(1, 3, CFG.model_frames, CFG.img_size, CFG.img_size))
assert tuple(probe_output.shape) == (1, N_CLASSES)
print(f"MViTv2-S 32x128 forward preflight: output={tuple(probe_output.shape)} OK")
del probe_output
del probe_model


def lr_at(step, total_steps, warmup_steps, cfg):
    if step < warmup_steps:
        return cfg.lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return cfg.min_lr + .5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, cfg, tta=False):
    model.eval()
    probs = np.zeros((len(loader.dataset), N_CLASSES), np.float32)
    loss_sum = seen = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    for clip, label, idx in loader:
        clip, label = clip.to(DEVICE, non_blocking=True), label.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", enabled=cfg.amp and DEVICE.type == "cuda"):
            logits = model(clip.permute(0, 2, 1, 3, 4))
            if tta:
                flipped = model(torch.flip(clip, [-1]).permute(0, 2, 1, 3, 4))
                logits = (logits + flipped) * .5
            loss = criterion(logits, label)
        loss_sum += loss.item() * len(label); seen += len(label)
        probs[idx.numpy()] = logits.float().softmax(1).cpu().numpy()
    accuracy = float((probs.argmax(1) == loader.dataset.labels).mean())
    return loss_sum / max(seen, 1), accuracy, probs


def common_loader_args(cfg):
    return dict(num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0)


def video_mixup(clip, label, cfg):
    """Mix complete 32-frame clips and their smoothed class targets."""
    targets = F.one_hot(label, N_CLASSES).float()
    targets = targets * (1.0 - cfg.label_smoothing) + cfg.label_smoothing / N_CLASSES
    if len(clip) > 1 and random.random() < cfg.mixup_p:
        lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
        lam = max(lam, 1.0 - lam)  # preserve one dominant action per mixture
        permutation = torch.randperm(len(clip), device=clip.device)
        clip = clip * lam + clip[permutation] * (1.0 - lam)
        targets = targets * lam + targets[permutation] * (1.0 - lam)
    return clip, targets


def soft_target_cross_entropy(logits, targets):
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def train_selection(meta, cfg):
    from timm.utils import ModelEmaV3
    seed_everything(cfg.seed + 1)
    boundary_mask = meta.user.isin(cfg.boundary_users)
    assert set(meta.loc[boundary_mask, "user"].unique()) == set(cfg.boundary_users)
    train_part = meta.loc[~boundary_mask]
    val_part = meta.loc[boundary_mask]
    npy = cfg.cache_dir / "train_thermal.npy"
    tr_loader = DataLoader(ThermalDataset(npy, train_part, cfg, True), cfg.batch_size, shuffle=True,
                           drop_last=True, **common_loader_args(cfg))
    va_loader = DataLoader(ThermalDataset(npy, val_part, cfg, False),
                           cfg.batch_size, shuffle=False, **common_loader_args(cfg))
    print(f"selection boundary: train={len(train_part)}, val={len(val_part)}, "
          f"val users={sorted(val_part.user.unique())}")
    model = build_model(True, cfg).to(DEVICE)
    ema = ModelEmaV3(model, decay=cfg.ema_decay, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and DEVICE.type == "cuda")
    updates_per_epoch = len(tr_loader) // cfg.grad_accum
    assert updates_per_epoch > 0
    total_steps = updates_per_epoch * cfg.max_selection_epochs
    warmup_steps = updates_per_epoch * cfg.warmup_epochs
    step = 0; best_acc = -1.; best_epoch = 1; best_probs = best_state = None
    for epoch in range(1, cfg.max_selection_epochs + 1):
        model.train(); loss_sum = seen = 0; start = time.time()
        optimizer.zero_grad(set_to_none=True)
        usable_microbatches = updates_per_epoch * cfg.grad_accum
        for micro_step, (clip, label, _) in enumerate(tr_loader):
            if micro_step >= usable_microbatches:
                break
            clip, label = clip.to(DEVICE, non_blocking=True), label.to(DEVICE, non_blocking=True)
            clip, soft_targets = video_mixup(clip, label, cfg)
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(clip.permute(0, 2, 1, 3, 4))
                raw_loss = soft_target_cross_entropy(logits, soft_targets)
                loss = raw_loss / cfg.grad_accum
            scaler.scale(loss).backward()
            loss_sum += raw_loss.item() * len(label); seen += len(label)
            if (micro_step + 1) % cfg.grad_accum == 0:
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(step, total_steps, warmup_steps, cfg)
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                ema.update(model, step=step); step += 1
        val_loss, val_acc, val_probs = evaluate(ema.module, va_loader, cfg, tta=cfg.tta_hflip)
        if val_acc > best_acc:
            best_acc, best_epoch, best_probs = val_acc, epoch, val_probs.copy()
            best_state = {k: v.detach().cpu().clone() for k, v in ema.module.state_dict().items()}
        print(f"boundary {epoch:02d}/{cfg.max_selection_epochs} lr={optimizer.param_groups[0]['lr']:.2e} "
              f"train={loss_sum/max(seen,1):.4f} val={val_loss:.4f}/{val_acc:.4f} "
              f"best={best_acc:.4f}@{best_epoch} time={time.time()-start:.0f}s")
    np.save(cfg.out_dir / "selection_boundary_probs.npy", best_probs)
    val_out = val_part[["clip_id", "user", "action_name", "action_id", "label_idx"]].reset_index(drop=True)
    val_out["prediction"] = LABEL_VALUES[best_probs.argmax(1)]
    val_out.to_parquet(cfg.out_dir / "selection_boundary_predictions.parquet", index=False)
    selection_steps_per_epoch = updates_per_epoch
    del model, ema, optimizer, tr_loader
    gc.collect(); torch.cuda.empty_cache()

    packed = quantize_state_dict(best_state)
    quantized_model = build_model(False, cfg)
    quantized_model.load_state_dict(dequantize_state_dict(packed))
    quantized_model = quantized_model.to(DEVICE).eval()
    _, int8_acc, _ = evaluate(quantized_model, va_loader, cfg, tta=cfg.tta_hflip)
    print(f"DIAGNOSTICS boundary: FP32 original+hflip={best_acc:.4f}, "
          f"INT8 original+hflip={int8_acc:.4f}, delta={int8_acc-best_acc:+.4f}")
    pd.DataFrame([{
        "split": "boundary",
        "val_users": ",".join(sorted(val_part.user.unique())),
        "best_epoch": best_epoch,
        "fp32_accuracy": best_acc,
        "int8_accuracy": int8_acc,
        "quantization_delta": int8_acc - best_acc,
        "optimizer_updates": best_epoch * selection_steps_per_epoch,
    }]).to_csv(cfg.out_dir / "diagnostics_boundary.csv", index=False)
    del best_state, packed, quantized_model, va_loader
    gc.collect(); torch.cuda.empty_cache()
    return best_epoch, best_acc, int8_acc, best_epoch * selection_steps_per_epoch, total_steps, warmup_steps


def train_full(meta, target_updates, schedule_total_steps, schedule_warmup_steps, cfg):
    from timm.utils import ModelEmaV3
    seed_everything(cfg.seed + 100)
    npy = cfg.cache_dir / "train_thermal.npy"
    loader = DataLoader(ThermalDataset(npy, meta, cfg, True), cfg.batch_size, shuffle=True,
                        drop_last=True, **common_loader_args(cfg))
    model = build_model(True, cfg).to(DEVICE)
    ema = ModelEmaV3(model, decay=cfg.ema_decay, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and DEVICE.type == "cuda")
    full_steps_per_epoch = len(loader) // cfg.grad_accum
    assert full_steps_per_epoch > 0
    epochs = math.ceil(target_updates / full_steps_per_epoch)
    step = 0
    for epoch in range(1, epochs + 1):
        model.train(); loss_sum = seen = 0; start = time.time()
        optimizer.zero_grad(set_to_none=True)
        remaining_updates = target_updates - step
        usable_microbatches = min(full_steps_per_epoch, remaining_updates) * cfg.grad_accum
        for micro_step, (clip, label, _) in enumerate(loader):
            if micro_step >= usable_microbatches:
                break
            if step >= target_updates:
                break
            clip, label = clip.to(DEVICE, non_blocking=True), label.to(DEVICE, non_blocking=True)
            clip, soft_targets = video_mixup(clip, label, cfg)
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                raw_loss = soft_target_cross_entropy(
                    model(clip.permute(0, 2, 1, 3, 4)), soft_targets)
                loss = raw_loss / cfg.grad_accum
            scaler.scale(loss).backward()
            loss_sum += raw_loss.item() * len(label); seen += len(label)
            if (micro_step + 1) % cfg.grad_accum == 0:
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(step, schedule_total_steps, schedule_warmup_steps, cfg)
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                ema.update(model, step=step); step += 1
        print(f"full {epoch:02d}/{epochs} updates={step}/{target_updates} "
              f"lr={optimizer.param_groups[0]['lr']:.2e} train={loss_sum/max(seen,1):.4f} "
              f"time={time.time()-start:.0f}s")
    state = {k: v.detach().cpu().clone() for k, v in ema.module.state_dict().items()}
    del model, ema, optimizer, loader
    gc.collect(); torch.cuda.empty_cache()
    return state


def quantize_state_dict(state):
    quantized = {}
    for key, value in tqdm(state.items(), desc="INT8 pack weights"):
        if value.is_floating_point() and value.ndim >= 2:
            weight = value.float()
            dims = tuple(range(1, weight.ndim))
            scale = (weight.abs().amax(dim=dims, keepdim=True) / 127).clamp_min(1e-12)
            quantized[key] = {"q": (weight / scale).round().clamp(-127, 127).to(torch.int8), "scale": scale.half()}
        else:
            quantized[key] = value.half() if value.is_floating_point() else value
    return quantized


def dequantize_state_dict(quantized):
    return {key: (value["q"].float() * value["scale"].float()
                  if isinstance(value, dict) else (value.float() if value.is_floating_point() else value))
            for key, value in quantized.items()}


@torch.no_grad()
def predict_test(model_path, meta, cfg):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_model(False, cfg)
    model.load_state_dict(dequantize_state_dict(checkpoint["model_int8"]))
    model = model.to(DEVICE).eval()
    loader = DataLoader(ThermalDataset(cfg.cache_dir / "test_thermal.npy", meta, cfg, False),
                        cfg.batch_size, shuffle=False, **common_loader_args(cfg))
    probs = np.zeros((len(meta), N_CLASSES), np.float32)
    for clip, _, idx in tqdm(loader, desc="Thermal full-32 test inference"):
        clip = clip.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", enabled=cfg.amp and DEVICE.type == "cuda"):
            logits = model(clip.permute(0, 2, 1, 3, 4))
            if cfg.tta_hflip:
                logits = (logits + model(torch.flip(clip, [-1]).permute(0, 2, 1, 3, 4))) * .5
        probs[idx.numpy()] = logits.float().softmax(1).cpu().numpy()
    del model, loader
    gc.collect(); torch.cuda.empty_cache()
    return probs


best_epoch, selection_acc, selection_int8_acc, selection_updates, _, _ = train_selection(train_meta, CFG)
target_updates = CFG.full_train_updates
schedule_total_steps = CFG.schedule_total_steps
schedule_warmup_steps = CFG.schedule_warmup_steps
full_microbatches_per_epoch = len(train_meta) // CFG.batch_size
full_steps_per_epoch = full_microbatches_per_epoch // CFG.grad_accum
equivalent_full_epochs = target_updates / max(full_steps_per_epoch, 1)
print(f"Boundary experiment complete: best_epoch={best_epoch}, "
      f"FP32_accuracy={selection_acc:.4f}, INT8_accuracy={selection_int8_acc:.4f}, "
      f"selection_best_updates={selection_updates}")
print(f"Full-data training target={target_updates} updates "
      f"(~{equivalent_full_epochs:.2f} full-data epochs)")

final_state = train_full(train_meta, target_updates, schedule_total_steps,
                         schedule_warmup_steps, CFG)
packed = quantize_state_dict(final_state)
FINAL_MODEL = CFG.out_dir / "thermal_mvitv2s_full32_int8.pt"
torch.save({
    "model_int8": packed,
    "architecture": "torchvision_mvit_v2_s_relative_position_32x128",
    "classes": LABEL_VALUES.tolist(),
    "cache_frames": CFG.cache_frames,
    "model_frames": CFG.model_frames,
    "temporal_input": "all_cached_frames_in_order",
    "image_size": CFG.img_size,
    "input_channels": CFG.in_channels,
    "selected_epoch": best_epoch,
    "full_train_updates": target_updates,
    "schedule_total_steps": schedule_total_steps,
    "schedule_warmup_steps": schedule_warmup_steps,
    "boundary_fp32_accuracy": selection_acc,
    "boundary_int8_accuracy": selection_int8_acc,
    "mixup_probability": CFG.mixup_p,
    "mixup_alpha": CFG.mixup_alpha,
    "boundary_users": CFG.boundary_users,
}, FINAL_MODEL)

thermal_mib = FINAL_MODEL.stat().st_size / 1024**2
yolo_mib = YOLO_PATH.stat().st_size / 1024**2
total_mib = thermal_mib + yolo_mib + CFG.other_models_mib
print(f"Thermal INT8={thermal_mib:.2f} MiB "
      f"(target {CFG.thermal_target_mib:.0f}, soft ceiling {CFG.thermal_soft_ceiling_mib:.0f})")
if thermal_mib > CFG.thermal_target_mib:
    print("WARNING: Thermal model exceeds the 30 MiB target; output is retained as requested.")
if thermal_mib > CFG.thermal_soft_ceiling_mib:
    print("WARNING: Thermal model exceeds the 35 MiB soft ceiling; output is still retained.")
print(f"Estimated complete ensemble={total_mib:.2f} MiB = other {CFG.other_models_mib:.2f} "
      f"+ YOLO {yolo_mib:.2f} + Thermal {thermal_mib:.2f}")
if total_mib > 100:
    print("WARNING: Estimated inference package exceeds 100 MiB; no files were deleted.")

test_probs = predict_test(FINAL_MODEL, test_meta, CFG)
np.save(CFG.out_dir / "thermal_test_probs.npy", test_probs)
predicted_ids = LABEL_VALUES[test_probs.argmax(1)]
test_csv = pd.read_csv(CFG.test_csv)
assert len(test_csv) == len(predicted_ids)
submission = test_csv[["path"]].copy()
submission["prediction"] = predicted_ids.astype(int)
SUBMISSION = CFG.work_root / "thermal_submission.csv"
submission.to_csv(SUBMISSION, index=False)
print(f"wrote {SUBMISSION} ({len(submission)} rows, "
      f"{submission.prediction.nunique()}/{N_CLASSES} classes)")
print(f"saved ensemble probabilities: {CFG.out_dir / 'thermal_test_probs.npy'}")
print(f"saved final model: {FINAL_MODEL}")
display(submission.head())
