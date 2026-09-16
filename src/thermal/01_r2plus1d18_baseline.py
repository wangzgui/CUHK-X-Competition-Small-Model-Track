"""Compact Thermal-only video stream with shared IR-derived person crops.

Kaggle usage:
    %run /kaggle/input/<your-code-dataset>/thermal_r2plus1d18.py

The final classifier checkpoint is weight-only INT8 and is hard-limited to
30 MiB. YOLO is shared with the existing IR/Depth stream and is not duplicated
inside the Thermal checkpoint.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
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
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
from tqdm.auto import tqdm

Image.MAX_IMAGE_PIXELS = None
N_CLASSES = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    # --------------------------------------------------------------- data paths
    thermal_root: Path = Path("/kaggle/input/datasets/zhuowamg/thermal/Thermal")
    class_map_csv: Path = Path("/kaggle/input/datasets/zhuowamg/class-mapping1/class_mapping.csv")
    test_root: Path = Path("/kaggle/input/datasets/zhuowamg/cuhksmall/CUHK-X_dataset/Small-Model-Track/Testing/data/small_model_track_test/small_model_track_test")
    test_csv: Path = Path("/kaggle/input/datasets/zhuowamg/cuhksmall/CUHK-X_dataset/Small-Model-Track/Testing/test_file/test.csv")
    # Training IR is used only to calculate the crop shared with IR/Depth.
    ir_train_root: Path = Path("/kaggle/input/CUHK-X_Small_Model_Track/Small-Model-Track/Training/data/HAR/data/IR")
    work_root: Path = Path("/kaggle/working")
    cache_root: Path = Path("/kaggle/temp/thermal_cache")

    # ---------------------------------------------------------- clip processing
    n_frames: int = 16
    img_size: int = 128
    cache_workers: int = 4
    person_crop: bool = True
    person_det_frames: int = 8
    person_det_conf: float = .25
    # Larger than the IR/Depth crop because IR and Thermal cameras may not be
    # perfectly registered even when their frames are synchronised.
    person_crop_margin: float = 1.65
    person_crop_min_side: float = .45
    norm_sample_clips: int = 384

    # --------------------------------------------------------- validation/final
    n_folds: int = 5
    # Start with (0,). Use (0,1,2,3,4) for a stable epoch estimate.
    cv_folds: tuple[int, ...] = (0,)
    seed: int = 42

    # --------------------------------------------------------------- training
    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-4
    min_lr: float = 1e-6
    warmup_epochs: int = 2
    weight_decay: float = 1e-4
    label_smoothing: float = .05
    dropout: float = .30
    ema_decay: float = .99
    grad_clip: float = 5.
    amp: bool = True
    num_workers: int = 2

    # Pseudo-colour encodes temperature: do not use colour jitter.
    hflip_p: float = .5
    crop_scale: tuple[float, float] = (.82, 1.)
    tta_hflip: bool = True

    # --------------------------------------------------------------- budgets
    thermal_limit_mib: float = 30.
    other_models_mib: float = 66.  # IR/Depth 60 + shared YOLO 5 + IMU 1
    total_limit_mib: float = 100.

    @property
    def cache_dir(self):
        return self.cache_root / f"T{self.n_frames}_S{self.img_size}_IRcrop_v1"

    @property
    def out_dir(self):
        return self.work_root / "thermal_output"

    @property
    def yolo_path(self):
        return self.work_root / "yolo11n.pt"


CFG = Config()


@dataclass
class Clip:
    clip_id: str
    action_name: str
    action_id: int
    user: str
    trial: str
    thermal: list[str]
    ir: list[str]


_NUMBER = re.compile(r"(\d+)")


def frame_files(folder: Path) -> list[str]:
    if not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    def key(path):
        nums = _NUMBER.findall(path.stem)
        return (0, int(nums[-1])) if nums else (1, path.name)
    return [str(p) for p in sorted(files, key=key)]


def build_train_index(cfg) -> list[Clip]:
    cmap = pd.read_csv(cfg.class_map_csv, encoding="utf-8-sig")
    name2id = dict(zip(cmap.action_name.astype(str), cmap.action_id.astype(int)))
    clips = []
    for trial_dir in tqdm(sorted(cfg.thermal_root.glob("*/*/*")), desc="index Thermal"):
        if not trial_dir.is_dir():
            continue
        action, user, trial = trial_dir.parts[-3:]
        thermal = frame_files(trial_dir)
        if thermal and action in name2id:
            clips.append(Clip(f"{action}/{user}/{trial}", action, name2id[action], user,
                              trial, thermal, frame_files(cfg.ir_train_root / action / user / trial)))
    if not clips:
        raise RuntimeError(f"No Thermal training clips found below {cfg.thermal_root}")
    return clips


def build_test_index(cfg) -> list[Clip]:
    clips = []
    for raw_path in pd.read_csv(cfg.test_csv)["path"]:
        clip_id = str(raw_path).strip("/").split("/")[-1]
        root = cfg.test_root / clip_id
        clips.append(Clip(clip_id, "", -1, "test", "", frame_files(root / "Thermal"),
                          frame_files(root / "IR")))
    return clips


def pick_indices(n_src, n_out):
    return (np.linspace(0, n_src - 1, n_out).round().astype(int)
            if n_src > 0 else np.zeros(0, dtype=int))


def ensure_shared_yolo(cfg):
    if cfg.yolo_path.exists():
        return
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    candidates = [Path(getattr(model, "ckpt_path", "")), Path.cwd() / "yolo11n.pt",
                  Path("/root/yolo11n.pt")]
    for path in candidates:
        if path.is_file():
            shutil.copy2(path, cfg.yolo_path)
            return
    raise FileNotFoundError("YOLO downloaded but yolo11n.pt could not be located")


def window_from_boxes(boxes, width, height, cfg):
    x0, y0 = boxes[:, 0].min() / width, boxes[:, 1].min() / height
    x1, y1 = boxes[:, 2].max() / width, boxes[:, 3].max() / height
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side_px = max((x1 - x0) * width, (y1 - y0) * height) * cfg.person_crop_margin
    side_px = max(side_px, cfg.person_crop_min_side * max(width, height))
    hx, hy = side_px / width / 2, side_px / height / 2
    return max(cx-hx, 0.), max(cy-hy, 0.), min(cx+hx, 1.), min(cy+hy, 1.)


def detect_windows(clips, split, cfg):
    crop_file = cfg.cache_dir / f"{split}_ir_windows.parquet"
    if crop_file.exists():
        saved = pd.read_parquet(crop_file)
        if saved.clip_id.tolist() == [c.clip_id for c in clips]:
            return [None if pd.isna(r.x0) else (r.x0, r.y0, r.x1, r.y1)
                    for r in saved.itertuples()]
    if not cfg.person_crop:
        return [None] * len(clips)
    ensure_shared_yolo(cfg)
    from ultralytics import YOLO
    model = YOLO(str(cfg.yolo_path)); device = 0 if torch.cuda.is_available() else "cpu"
    windows = []
    for clip in tqdm(clips, desc=f"IR crop windows [{split}]"):
        paths = [clip.ir[i] for i in sorted(set(pick_indices(len(clip.ir), cfg.person_det_frames)))]
        boxes, width, height = [], None, None
        if paths:
            for result in model.predict(paths, classes=[0], conf=cfg.person_det_conf,
                                        verbose=False, device=device):
                height, width = result.orig_shape
                if len(result.boxes):
                    boxes.append(result.boxes.xyxy[result.boxes.conf.argmax()].cpu().numpy())
        windows.append(window_from_boxes(np.asarray(boxes), width, height, cfg) if boxes else None)
    rows = []
    for clip, win in zip(clips, windows):
        rows.append(dict(clip_id=clip.clip_id, x0=np.nan if win is None else win[0],
                         y0=np.nan if win is None else win[1], x1=np.nan if win is None else win[2],
                         y1=np.nan if win is None else win[3]))
    pd.DataFrame(rows).to_parquet(crop_file, index=False)
    return windows


_MM = None
_WORKER = {}


def _init_worker(path, shape, size):
    global _MM, _WORKER
    _MM = np.memmap(path, np.uint8, "r+", shape=shape)
    _WORKER = dict(size=size, frames=shape[1])


def _read_thermal(path, size, window):
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if window is not None:
                width, height = image.size
                image = image.crop((round(window[0]*width), round(window[1]*height),
                                    round(window[2]*width), round(window[3]*height)))
            return np.asarray(image.resize((size, size), Image.BILINEAR), np.uint8)
    except Exception:
        return None


def _cache_one(job):
    row, clip, window = job
    buf = np.zeros(_MM.shape[1:], np.uint8)
    bad = 0
    for t, source_idx in enumerate(pick_indices(len(clip.thermal), _WORKER["frames"])):
        arr = _read_thermal(clip.thermal[source_idx], _WORKER["size"], window)
        if arr is None or arr.max() == 0:
            bad += 1
        else:
            buf[t] = arr.transpose(2, 0, 1)
    _MM[row] = buf
    return dict(row=row, clip_id=clip.clip_id, action_name=clip.action_name,
                action_id=clip.action_id, user=clip.user, trial=clip.trial,
                has_crop=window is not None, bad_frames=bad)


def build_cache(clips, split, cfg):
    npy = cfg.cache_dir / f"{split}_frames.npy"
    meta_file = cfg.cache_dir / f"{split}_meta.parquet"
    shape = (len(clips), cfg.n_frames, 3, cfg.img_size, cfg.img_size)
    if (npy.exists() and meta_file.exists() and
            npy.stat().st_size == int(np.prod(shape))):
        return pd.read_parquet(meta_file)
    windows = detect_windows(clips, split, cfg)
    np.memmap(npy, np.uint8, "w+", shape=shape).flush()
    rows = []
    jobs = zip(range(len(clips)), clips, windows)
    with Pool(cfg.cache_workers, initializer=_init_worker,
              initargs=(str(npy), shape, cfg.img_size)) as pool:
        for record in tqdm(pool.imap_unordered(_cache_one, jobs, chunksize=8),
                           total=len(clips), desc=f"cache Thermal [{split}]"):
            rows.append(record)
    meta = pd.DataFrame(rows).sort_values("row").reset_index(drop=True)
    meta.to_parquet(meta_file, index=False)
    print(f"{split}: {len(meta)} clips | no IR crop {(~meta.has_crop).sum()} | bad frames {meta.bad_frames.sum()}")
    return meta


def thermal_stats(cache_npy, n_rows, cfg):
    stats_file = cfg.cache_dir / "thermal_norm.json"
    if stats_file.exists():
        saved = json.loads(stats_file.read_text())
        return torch.tensor(saved["mean"]).view(1, 3, 1, 1), torch.tensor(saved["std"]).view(1, 3, 1, 1)
    shape = (n_rows, cfg.n_frames, 3, cfg.img_size, cfg.img_size)
    mm = np.memmap(cache_npy, np.uint8, "r", shape=shape)
    ids = np.random.default_rng(cfg.seed).choice(n_rows, min(n_rows, cfg.norm_sample_clips), replace=False)
    sums = np.zeros(3, np.float64); squares = np.zeros(3, np.float64); count = 0
    for start in range(0, len(ids), 16):
        x = np.asarray(mm[ids[start:start+16]], dtype=np.float32) / 255.
        sums += x.sum(axis=(0, 1, 3, 4)); squares += np.square(x).sum(axis=(0, 1, 3, 4))
        count += x.shape[0] * x.shape[1] * x.shape[3] * x.shape[4]
    mean = sums / count; std = np.sqrt(np.maximum(squares / count - mean**2, 1e-6))
    stats_file.write_text(json.dumps(dict(mean=mean.tolist(), std=std.tolist())))
    print("Thermal mean/std:", mean, std)
    return torch.tensor(mean, dtype=torch.float32).view(1,3,1,1), torch.tensor(std, dtype=torch.float32).view(1,3,1,1)


def assign_folds(meta, cfg):
    folds = np.full(len(meta), -1, int)
    splitter = StratifiedGroupKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed)
    for fold, (_, val) in enumerate(splitter.split(meta, meta.action_id, meta.user)):
        folds[val] = fold
    return folds


class ThermalDataset(Dataset):
    def __init__(self, npy, meta, cfg, mean, std, train):
        self.path, self.meta, self.cfg, self.train = str(npy), meta.reset_index(drop=True), cfg, train
        self.rows = self.meta.row.to_numpy(); self.labels = self.meta.action_id.to_numpy()
        self.shape = (os.path.getsize(self.path) // (cfg.n_frames*3*cfg.img_size**2),
                      cfg.n_frames, 3, cfg.img_size, cfg.img_size)
        self.mean, self.std, self._mm = mean, std, None
    def __len__(self): return len(self.meta)
    @property
    def mm(self):
        if self._mm is None:
            self._mm = np.memmap(self.path, np.uint8, "r", shape=self.shape)
        return self._mm
    def __getitem__(self, idx):
        x = torch.from_numpy(np.asarray(self.mm[self.rows[idx]]).copy()).float().div_(255.)
        if self.train:
            if random.random() < self.cfg.hflip_p:
                x = torch.flip(x, (-1,))
            lo, hi = self.cfg.crop_scale
            scale = random.uniform(lo, hi); side = max(8, round(self.cfg.img_size*scale))
            top = random.randint(0, self.cfg.img_size-side); left = random.randint(0, self.cfg.img_size-side)
            x = F.interpolate(x[:, :, top:top+side, left:left+side],
                              (self.cfg.img_size, self.cfg.img_size), mode="bilinear", align_corners=False)
        return x.sub(self.mean).div(self.std), int(self.labels[idx]), idx


class CompactR2Plus1D18(nn.Module):
    """Pretrained R(2+1)D-18 with the second layer4 block removed."""
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        net = r2plus1d_18(weights=weights)
        net.layer4 = nn.Sequential(net.layer4[0])
        width = net.fc.in_features; net.fc = nn.Identity()
        self.encoder = net
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(width, N_CLASSES))
    def forward(self, x):
        return self.head(self.encoder(x.permute(0, 2, 1, 3, 4)))


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def lr_at(step, total, warmup, cfg):
    if step < warmup: return cfg.lr * (step+1) / max(warmup, 1)
    p = (step-warmup) / max(total-warmup, 1)
    return cfg.min_lr + .5*(cfg.lr-cfg.min_lr)*(1+math.cos(math.pi*p))


@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval(); correct = total = 0
    for x, y, _ in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", enabled=cfg.amp): logits = model(x)
        correct += (logits.argmax(1) == y).sum().item(); total += len(y)
    return correct / max(total, 1)


def fit(meta, fold, epochs, cfg, mean, std, full=False):
    from timm.utils import ModelEmaV3
    seed_everything(cfg.seed + (100 if full else fold))
    train_meta = meta if full else meta[meta.fold != fold]
    val_meta = None if full else meta[meta.fold == fold]
    npy = cfg.cache_dir / "train_frames.npy"
    common = dict(num_workers=cfg.num_workers, pin_memory=True,
                  persistent_workers=cfg.num_workers > 0)
    train_loader = DataLoader(ThermalDataset(npy, train_meta, cfg, mean, std, True),
                              cfg.batch_size, shuffle=True, drop_last=True, **common)
    val_loader = None if full else DataLoader(ThermalDataset(npy, val_meta, cfg, mean, std, False),
                                               cfg.batch_size, shuffle=False, **common)
    model = CompactR2Plus1D18(cfg, pretrained=True).to(DEVICE)
    ema = ModelEmaV3(model, decay=cfg.ema_decay, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    total_steps = len(train_loader)*epochs; warmup = len(train_loader)*cfg.warmup_epochs
    best_acc, best_epoch, best_state, step = -1., epochs, None, 0
    for epoch in range(epochs):
        model.train(); loss_sum = seen = 0
        for x, y, _ in train_loader:
            for group in optimizer.param_groups: group["lr"] = lr_at(step, total_steps, warmup, cfg)
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=cfg.amp):
                loss = F.cross_entropy(model(x), y, label_smoothing=cfg.label_smoothing)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer); scaler.update(); ema.update(model, step=step)
            step += 1; loss_sum += loss.item()*len(y); seen += len(y)
        if full:
            print(f"full epoch {epoch+1:02d}/{epochs} train={loss_sum/max(seen,1):.4f}")
        else:
            acc = evaluate(ema.module, val_loader, cfg)
            print(f"fold {fold} epoch {epoch+1:02d}/{epochs} train={loss_sum/max(seen,1):.4f} val={acc:.4f}")
            if acc > best_acc:
                best_acc, best_epoch = acc, epoch+1
                best_state = {k:v.detach().cpu().clone() for k,v in ema.module.state_dict().items()}
    if full:
        return {k:v.detach().cpu().clone() for k,v in ema.module.state_dict().items()}, None, epochs
    del model, ema; torch.cuda.empty_cache()
    return best_state, best_acc, best_epoch


def quantize_state_dict(state):
    out = {}
    for key, value in state.items():
        if value.is_floating_point() and value.ndim >= 2:
            w = value.float(); dims = tuple(range(1, w.ndim))
            scale = (w.abs().amax(dim=dims, keepdim=True)/127).clamp_min(1e-12)
            out[key] = dict(q=(w/scale).round().clamp(-127,127).to(torch.int8), scale=scale.half())
        else:
            out[key] = value.half() if value.is_floating_point() else value
    return out


def dequantize_state_dict(state):
    return {k: (v["q"].float()*v["scale"].float() if isinstance(v, dict)
                else (v.float() if v.is_floating_point() else v)) for k,v in state.items()}


@torch.no_grad()
def predict(checkpoint, loader, cfg):
    model = CompactR2Plus1D18(cfg, pretrained=False).to(DEVICE)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(dequantize_state_dict(saved["model_int8"])); model.eval()
    probabilities = np.zeros((len(loader.dataset), N_CLASSES), np.float32)
    for x, _, idx in tqdm(loader, desc="Thermal inference"):
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", enabled=cfg.amp):
            logits = model(x)
            if cfg.tta_hflip: logits = (logits + model(torch.flip(x, (-1,)))) / 2
        probabilities[idx.numpy()] = logits.float().softmax(1).cpu().numpy()
    return probabilities


def main():
    cfg = CFG
    for path in (cfg.thermal_root, cfg.class_map_csv, cfg.test_root, cfg.test_csv):
        assert path.exists(), f"missing: {path}"
    if cfg.person_crop:
        assert cfg.ir_train_root.is_dir(), (
            f"shared IR crop source is missing: {cfg.ir_train_root}; "
            "set Config.ir_train_root to the attached training IR directory"
        )
    cfg.cache_dir.mkdir(parents=True, exist_ok=True); cfg.out_dir.mkdir(parents=True, exist_ok=True)
    train_clips, test_clips = build_train_index(cfg), build_test_index(cfg)
    train_meta = build_cache(train_clips, "train", cfg)
    test_meta = build_cache(test_clips, "test", cfg); test_meta["action_id"] = -1
    mean, std = thermal_stats(cfg.cache_dir/"train_frames.npy", len(train_meta), cfg)
    train_meta["fold"] = assign_folds(train_meta, cfg)

    probe = CompactR2Plus1D18(cfg, pretrained=False)
    params = sum(p.numel() for p in probe.parameters()); del probe
    print(f"Compact R(2+1)D params={params/1e6:.2f}M | estimated INT8={params/1024**2:.2f} MiB")

    scores, epochs = [], []
    for fold in cfg.cv_folds:
        _, score, best_epoch = fit(train_meta, fold, cfg.epochs, cfg, mean, std)
        scores.append(score); epochs.append(best_epoch)
    final_epochs = max(5, round(float(np.mean(epochs))))
    print(f"CV={np.mean(scores):.4f} | selected full-data epochs={final_epochs}")

    state, _, _ = fit(train_meta, 0, final_epochs, cfg, mean, std, full=True)
    checkpoint = cfg.out_dir / "thermal_int8.pt"
    torch.save(dict(model_int8=quantize_state_dict(state), epochs=final_epochs,
                    classes=N_CLASSES), checkpoint)
    thermal_mib = checkpoint.stat().st_size / 1024**2
    print(f"Thermal checkpoint={thermal_mib:.2f} MiB / {cfg.thermal_limit_mib:.2f} MiB")
    if thermal_mib > cfg.thermal_limit_mib:
        raise RuntimeError("Thermal checkpoint exceeds 30 MiB; inference stopped")
    total_mib = cfg.other_models_mib + thermal_mib
    print(f"Estimated complete ensemble={total_mib:.2f} MiB / {cfg.total_limit_mib:.2f} MiB")
    if total_mib > cfg.total_limit_mib:
        raise RuntimeError("Complete ensemble exceeds 100 MiB; inference stopped")

    loader = DataLoader(ThermalDataset(cfg.cache_dir/"test_frames.npy", test_meta, cfg,
                                       mean, std, False), cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)
    probs = predict(checkpoint, loader, cfg); labels = probs.argmax(1)
    test = pd.read_csv(cfg.test_csv)
    test["prediction"] = labels
    test[["path", "prediction"]].to_csv(cfg.work_root/"thermal_submission.csv", index=False)
    np.save(cfg.out_dir/"thermal_test_probs.npy", probs)
    print(f"wrote {cfg.work_root/'thermal_submission.csv'} and thermal_test_probs.npy")


if __name__ == "__main__":
    main()
