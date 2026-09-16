from __future__ import annotations

import gc
import hashlib
import html
import json
import importlib.util
import math
import os
import random
import re
import shutil
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
from PIL import Image, ImageDraw
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s
from tqdm.auto import tqdm

# Keep the notebook self-contained in a single ordinary Python cell.
for package in ("ultralytics", "timm"):
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
try:
    from timm.utils import ModelEmaV3
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "timm>=1.0.3"])
    from timm.utils import ModelEmaV3

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

    cache_frames: int = 64
    model_frames: int = 32
    img_size: int = 160
    in_channels: int = 3  # single-view v1: cropped Thermal RGB only
    cache_workers: int = 4

    person_det_frames: int = 8
    person_det_conf: float = 0.25
    person_crop_margin: float = 1.60
    person_crop_min_side: float = 0.45

    n_folds: int = 5
    selection_fold: int = 0
    seed: int = 2026

    # Thermal-only, MaskFeat-style domain pre-training. The target is a
    # normalized 9-bin HOG descriptor for each final MViTv2 token. Boundary
    # pre-training excludes every held-out boundary user; full pre-training
    # uses training Thermal only and never reads test clips or other modalities.
    maskfeat_updates: int = 1870
    maskfeat_lr: float = 2e-5
    maskfeat_min_lr: float = 2e-7
    maskfeat_warmup_steps: int = 187
    maskfeat_weight_decay: float = 0.05
    maskfeat_ratio: float = 0.40
    maskfeat_hog_bins: int = 9
    maskfeat_final_grid: tuple[int, int, int] = (16, 5, 5)

    # V19 adds a short LP-FT stage after MaskFeat. Only the 40-class head is
    # updated here; the original 3740-update V16 full fine-tune is not shortened.
    linear_probe_updates: int = 374
    linear_probe_lr: float = 1e-4
    linear_probe_weight_decay: float = 1e-4

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
    # Epoch selection remains SINGLE uniform view + hflip, as in the baseline.
    # After quantization, choose dual only with >=3 extra correct boundary samples.
    dual_min_extra_correct: int = 3
    qa_max_clips: int = 8
    champion_artifact_dir: Path = Path('/kaggle/working/thermal_full32_calibration_mixup_mvitv2s_boundary')

    thermal_target_mib: float = 30.0
    thermal_soft_ceiling_mib: float = 35.0
    other_models_mib: float = 61.0  # IR/Depth 60 + IMU 1; YOLO is measured separately.

    @property
    def cache_dir(self):
        return self.cache_root / f"V8_Tcache{self.cache_frames}_S{self.img_size}_IRcrop_uniform_v1"

    @property
    def out_dir(self):
        return self.work_root / "thermal_v19_mvitv2s_maskfeat_160_lpft_v16"


CFG = Config()
assert (CFG.cache_frames, CFG.model_frames, CFG.img_size, CFG.batch_size, CFG.grad_accum) == (64, 32, 160, 2, 6)
assert CFG.maskfeat_final_grid == (16, 5, 5) and CFG.maskfeat_hog_bins == 9
assert (CFG.linear_probe_updates, CFG.linear_probe_lr) == (374, 1e-4)
assert CFG.model_frames // CFG.maskfeat_final_grid[0] == 2
assert CFG.img_size // CFG.maskfeat_final_grid[1] == 32
assert CFG.img_size % CFG.maskfeat_final_grid[1] == 0
assert "S160" in CFG.cache_dir.name and "S128" not in CFG.cache_dir.name
# V16 MixUp mixes examples inside each physical minibatch. A silent fallback to
# batch=1 would therefore turn MixUp off and invalidate the spatial-resolution
# ablation. If batch=2 OOMs, stop and build a separately named memory variant.
assert CFG.batch_size >= 2, "V19 requires physical batch >=2 so V16 Video MixUp remains active"
assert CFG.tta_hflip and CFG.dual_min_extra_correct >= 1
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
print('V19: V16-160 plus head-only Linear Probe before the unchanged full fine-tune')
print(f'MaskFeat: updates={CFG.maskfeat_updates}, mask={CFG.maskfeat_ratio:.0%}, '
      f'lr={CFG.maskfeat_lr:.1e}, target_grid={CFG.maskfeat_final_grid}, HOG bins={CFG.maskfeat_hog_bins}')
print('Single hypothesis: a trained 40-class head reduces early full-finetune distortion of the MaskFeat representation.')
print(f'LP-FT: head-only updates={CFG.linear_probe_updates}, lr={CFG.linear_probe_lr:.1e}; '
      f'then the original {CFG.full_train_updates} full-model updates with a fresh optimizer and EMA.')
print('Cache guard: raw Thermal -> fixed IR union crop (margin=1.60) -> direct 160x160 resize; S128 cache is forbidden.')
print('Boundary SSL excludes held-out users; full SSL uses training Thermal only; test Thermal is never used.')
print('V16 fine-tuning remains: 64 cached frames -> ordered 32-bin random sampling; uniform LR; original INT8')
print('Checkpoint selection: single uniform32 + hflip. Boundary-only temporal diagnostics remain available.')
print('Official test export is locked to single+hflip: exactly 2 forwards/sample; no zoom or temporal ensemble.')
print(f"fixed full training: {CFG.full_train_updates} updates, "
      f"cosine schedule={CFG.schedule_total_steps}, warmup={CFG.schedule_warmup_steps}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def capture_rng_state():
    """Capture every RNG used by sampling, augmentation, MixUp, and DropPath."""
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state().clone(),
        cuda=[state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
    )


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


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
assert class_map['action_name'].is_unique, 'class_mapping.csv contains duplicate action_name values'
assert class_map['action_id'].is_unique, 'class_mapping.csv contains duplicate action_id values'
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
    manifest_path = cfg.cache_dir / f'{split}_cache_manifest.json'
    signature_data = dict(version='v16_raw_thermal_direct160_uniform64', shape=shape,
                          preprocessing='raw Thermal -> fixed IR union crop -> direct resize to configured size',
                          files=[c.thermal for c in clips],
                          windows=windows.to_dict(orient='records'))
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if (manifest.get('ready') and manifest.get('signature') == signature and
            npy.exists() and meta_path.exists() and npy.stat().st_size == expected):
        meta = pd.read_parquet(meta_path)
        if meta["clip_id"].tolist() == [c.clip_id for c in clips]:
            print(f"[{split}] reusing frame cache")
            return meta
    print(f"[{split}] cache shape={shape}, size={expected / 1024**3:.2f} GiB")
    # Account for an existing file that will be reused/rebuilt in this experiment directory.
    available = shutil.disk_usage(cfg.cache_dir).free + (npy.stat().st_size if npy.exists() else 0)
    if available < expected + 256 * 1024**2:
        raise RuntimeError(f'Insufficient cache disk space: need {expected/1024**3:.2f} GiB plus reserve')
    manifest_path.write_text(json.dumps(dict(ready=False, signature=signature)), encoding='utf-8')
    np.memmap(npy, dtype=np.uint8, mode="w+", shape=shape).flush()
    wins = [(r.x0, r.y0, r.x1, r.y1) if r.has_crop else None for r in windows.itertuples(index=False)]
    jobs = zip(range(len(clips)), clips, wins)
    records = []
    with Pool(cfg.cache_workers, initializer=_worker_init, initargs=(str(npy), shape, cfg.img_size)) as pool:
        for record in tqdm(pool.imap_unordered(_cache_one, jobs, chunksize=8), total=len(clips), desc=f"cache {split}"):
            records.append(record)
    meta = pd.DataFrame(records).sort_values("row").reset_index(drop=True)
    meta['source_count'] = [len(c.thermal) for c in clips]
    meta['thermal_available'] = (meta.bad_frames < cfg.cache_frames).astype(int)
    meta.to_parquet(meta_path, index=False)
    manifest_path.write_text(json.dumps(dict(ready=True, signature=signature, shape=shape)), encoding='utf-8')
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


def temporal_indices64(view, train=False):
    if train:
        # One frame from each ordered pair; never shuffle chronological order.
        return np.arange(0, 64, 2, dtype=np.int64) + np.random.randint(0, 2, size=32)
    if view == 'single':
        return np.linspace(0, 63, 32).round().astype(np.int64)
    if view == 'phase_a':
        return np.arange(0, 64, 2, dtype=np.int64)
    if view == 'phase_b':
        return np.arange(1, 64, 2, dtype=np.int64)
    raise ValueError(f'Unknown temporal view: {view}')


def sampling_audit(clips, cfg, split):
    rows, indices = [], []
    for clip in clips:
        n = len(clip.thermal)
        source = pick_indices(n, 64) if n else np.full(64, -1, dtype=int)
        indices.append(source)
        a, b, s = [source[temporal_indices64(v)] for v in ('phase_a', 'phase_b', 'single')]
        rows.append(dict(clip_id=clip.clip_id, source_count=n,
                         unique_cache_frames=len(np.unique(source)) if n else 0,
                         unique_single_frames=len(np.unique(s)) if n else 0,
                         unique_phase_a_frames=len(np.unique(a)) if n else 0,
                         unique_phase_b_frames=len(np.unique(b)) if n else 0,
                         different_ab_slots=int((a != b).sum()) if n else 0,
                         original32_matches_v8single=int((pick_indices(n, 32) == s).sum()) if n else 0))
    table = pd.DataFrame(rows)
    table.to_csv(cfg.out_dir / f'{split}_sampling_audit.csv', index=False)
    np.save(cfg.out_dir / f'{split}_raw_source_indices64.npy', np.asarray(indices))
    print(f'[{split}] sampling audit: median unique64={table.unique_cache_frames.median():.0f}, '
          f'median different A/B slots={table.different_ab_slots.median():.0f}/32, '
          f'clips with identical A/B={(table.different_ab_slots == 0).sum()}')
    print('64 cache slots are not necessarily 64 different source frames. No temporal trimming or reversal.')


sampling_audit(train_clips, CFG, 'train')
sampling_audit(test_clips, CFG, 'test')


class ThermalDataset(Dataset):
    def __init__(self, npy_path, meta, cfg, train, view='single'):
        assert (cfg.cache_frames, cfg.model_frames) == (64, 32)
        self.view = view
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
        return temporal_indices64(self.view, train=self.train)

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
        # Train: random one-per-bin. Validation/test: deterministic full-span schedule.
        frame_idx = self.temporal_indices()
        clip = torch.from_numpy(np.asarray(self.mm[self.rows[idx], frame_idx]).copy()).float().div_(255.0)
        if self.train:
            clip = self.augment(clip)
        clip = clip.sub_(KINETICS_MEAN).div_(KINETICS_STD)
        return clip, int(self.labels[idx]), idx


class MultiViewDataset(ThermalDataset):
    def __getitem__(self, idx):
        views = []
        for view in ('single', 'phase_a', 'phase_b'):
            frames = np.asarray(self.mm[self.rows[idx], temporal_indices64(view)]).copy()
            clip = torch.from_numpy(frames).float().div_(255.0)
            views.append(clip.sub_(KINETICS_MEAN).div_(KINETICS_STD))
        # Stored on CPU; each view goes to GPU separately to preserve microbatch size.
        return torch.stack(views), int(self.labels[idx]), idx


def build_model(pretrained, cfg):
    weights = MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None
    model = mvit_v2_s(weights=weights)

    # Torchvision stores the native 16x224 patch grid (8x56x56) as metadata
    # used by forward(). MViTv2 uses relative positional embeddings, so the
    # learned tables remain valid and are interpolated after updating this
    # metadata to our actual post-convolution grid (16x40x40 for 32x160 input).
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


class ThermalMaskFeat(nn.Module):
    """Mask final-grid regions at patch-token level and predict local Thermal HOG targets."""
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        patch_dim = backbone.conv_proj.out_channels
        feature_dim = backbone.head[-1].in_features
        self.mask_token = nn.Parameter(torch.zeros(1, 1, patch_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.predictor = nn.Linear(feature_dim, cfg.maskfeat_hog_bins)
        nn.init.trunc_normal_(self.predictor.weight, std=0.02)
        nn.init.zeros_(self.predictor.bias)

    def forward(self, video, final_mask):
        # video: normalized [B,C,T,H,W]; final_mask: [B,Tf,Hf,Wf].
        x = self.backbone.conv_proj(video)
        b, c, t, h, w = x.shape
        expected_initial = (self.backbone.pos_encoding.temporal_size,
                            *self.backbone.pos_encoding.spatial_size)
        assert (t, h, w) == expected_initial
        expanded_mask = F.interpolate(final_mask[:, None].float(), size=(t, h, w),
                                      mode='nearest')[:, 0].bool()
        x = x.flatten(2).transpose(1, 2)
        x = torch.where(expanded_mask.flatten(1).unsqueeze(-1),
                        self.mask_token.expand(b, x.shape[1], -1), x)
        x = self.backbone.pos_encoding(x)
        thw = expected_initial
        for block in self.backbone.blocks:
            x, thw = block(x, thw)
        x = self.backbone.norm(x)
        assert tuple(thw) == tuple(self.cfg.maskfeat_final_grid), (thw, self.cfg.maskfeat_final_grid)
        patch_tokens = x[:, 1:]
        assert patch_tokens.shape[1] == int(np.prod(thw))
        return self.predictor(patch_tokens)


@torch.no_grad()
def thermal_hog_targets(normalized_clip, cfg):
    """Return L2-normalized 9-bin unsigned HOG on the final 16x5x5 token grid."""
    mean = KINETICS_MEAN.to(normalized_clip.device)
    std = KINETICS_STD.to(normalized_clip.device)
    raw = (normalized_clip * std + mean).clamp(0, 1)  # [B,T,C,H,W]
    gray = 0.299 * raw[:, :, 0] + 0.587 * raw[:, :, 1] + 0.114 * raw[:, :, 2]
    gx = F.pad(gray[..., 2:] - gray[..., :-2], (1, 1, 0, 0))
    gy = F.pad(gray[..., 2:, :] - gray[..., :-2, :], (0, 0, 1, 1))
    magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)
    angle = torch.remainder(torch.atan2(gy, gx), math.pi)
    bins = torch.clamp((angle * cfg.maskfeat_hog_bins / math.pi).long(),
                       0, cfg.maskfeat_hog_bins - 1)
    hist = F.one_hot(bins, cfg.maskfeat_hog_bins).permute(0, 4, 1, 2, 3).float()
    hist.mul_(magnitude[:, None])
    tf, hf, wf = cfg.maskfeat_final_grid
    _, _, t, h, w = hist.shape
    assert t % tf == 0 and h % hf == 0 and w % wf == 0
    pooled = F.avg_pool3d(hist, kernel_size=(t // tf, h // hf, w // wf),
                         stride=(t // tf, h // hf, w // wf))
    targets = pooled.permute(0, 2, 3, 4, 1).reshape(len(raw), -1, cfg.maskfeat_hog_bins)
    return F.normalize(targets, dim=-1, eps=1e-6)


def final_grid_mask(batch_size, cfg, device):
    n_tokens = int(np.prod(cfg.maskfeat_final_grid))
    n_masked = max(1, min(n_tokens - 1, int(round(cfg.maskfeat_ratio * n_tokens))))
    chosen = torch.rand(batch_size, n_tokens, device=device).topk(n_masked, dim=1).indices
    mask = torch.zeros(batch_size, n_tokens, dtype=torch.bool, device=device)
    mask.scatter_(1, chosen, True)
    return mask.view(batch_size, *cfg.maskfeat_final_grid)


probe_model = build_model(False, CFG)
params = sum(p.numel() for p in probe_model.parameters())
print(f"MViTv2-S: {params / 1e6:.2f}M params, estimated INT8 storage {params / 1024**2:.2f} MiB")
print(f"MViTv2-S patch grid: {probe_model.pos_encoding.temporal_size}x"
      f"{probe_model.pos_encoding.spatial_size[0]}x{probe_model.pos_encoding.spatial_size[1]}")
assert (probe_model.pos_encoding.temporal_size, *probe_model.pos_encoding.spatial_size) == (16, 40, 40)
probe_model.eval()
with torch.no_grad():
    probe_output = probe_model(torch.zeros(1, 3, CFG.model_frames, CFG.img_size, CFG.img_size))
assert tuple(probe_output.shape) == (1, N_CLASSES)
print(f"MViTv2-S 32x160 forward preflight: output={tuple(probe_output.shape)} OK")
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
    if not np.isfinite(probs).all():
        raise FloatingPointError('Non-finite validation/test probabilities detected')
    accuracy = float((probs.argmax(1) == loader.dataset.labels).mean())
    return loss_sum / max(seen, 1), accuracy, probs


def common_loader_args(cfg):
    return dict(num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0)


def maskfeat_lr_at(step, cfg):
    if step < cfg.maskfeat_warmup_steps:
        return cfg.maskfeat_lr * (step + 1) / max(cfg.maskfeat_warmup_steps, 1)
    progress = (step - cfg.maskfeat_warmup_steps) / max(
        cfg.maskfeat_updates - cfg.maskfeat_warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return cfg.maskfeat_min_lr + .5 * (cfg.maskfeat_lr - cfg.maskfeat_min_lr) * (
        1 + math.cos(math.pi * progress))


def maskfeat_pretrain(meta, cfg, seed_value, tag):
    """Thermal-only domain adaptation. `meta` controls leakage: boundary users are excluded upstream."""
    seed_everything(seed_value)
    npy = cfg.cache_dir / 'train_thermal.npy'
    loader = DataLoader(ThermalDataset(npy, meta, cfg, True), cfg.batch_size,
                        shuffle=True, drop_last=True, **common_loader_args(cfg))
    backbone = build_model(True, cfg)
    ssl_model = ThermalMaskFeat(backbone, cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=cfg.maskfeat_lr,
                                  weight_decay=cfg.maskfeat_weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp and DEVICE.type == 'cuda')
    updates_per_epoch = len(loader) // cfg.grad_accum
    assert updates_per_epoch > 0
    epochs = math.ceil(cfg.maskfeat_updates / updates_per_epoch)
    step, history = 0, []
    for epoch in range(1, epochs + 1):
        ssl_model.train(); optimizer.zero_grad(set_to_none=True)
        loss_sum = masked_vectors = 0; start = time.time()
        remaining_updates = cfg.maskfeat_updates - step
        usable_microbatches = min(updates_per_epoch, remaining_updates) * cfg.grad_accum
        for micro_step, (clip, _, _) in enumerate(loader):
            if micro_step >= usable_microbatches or step >= cfg.maskfeat_updates:
                break
            clip = clip.to(DEVICE, non_blocking=True)
            mask = final_grid_mask(len(clip), cfg, DEVICE)
            targets = thermal_hog_targets(clip, cfg)
            flat_mask = mask.flatten(1)
            with torch.autocast('cuda', enabled=scaler.is_enabled()):
                predictions = ssl_model(clip.permute(0, 2, 1, 3, 4), mask)
                raw_loss = F.smooth_l1_loss(predictions[flat_mask], targets[flat_mask])
                loss = raw_loss / cfg.grad_accum
            assert torch.isfinite(raw_loss), f'non-finite MaskFeat loss at {tag} step {step}'
            if epoch == 1 and micro_step == 0:
                target_norm = targets[flat_mask].float().norm(dim=-1)
                print(f'MaskFeat {tag} QA: input={tuple(clip.shape)}, '
                      f'masked={int(flat_mask[0].sum())}/{flat_mask.shape[1]}, '
                      f'prediction={tuple(predictions.shape)}, target_norm_mean={target_norm.mean().item():.4f}, '
                      f'active_target_fraction={(target_norm > 0.1).float().mean().item():.3f}')
            scaler.scale(loss).backward()
            count = int(flat_mask.sum().item())
            loss_sum += raw_loss.item() * count; masked_vectors += count
            if (micro_step + 1) % cfg.grad_accum == 0:
                for group in optimizer.param_groups:
                    group['lr'] = maskfeat_lr_at(step, cfg)
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(ssl_model.parameters(), cfg.grad_clip)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                step += 1
        epoch_loss = loss_sum / max(masked_vectors, 1)
        history.append(dict(stage=tag, epoch=epoch, optimizer_updates=step,
                            lr=optimizer.param_groups[0]['lr'], maskfeat_loss=epoch_loss,
                            seconds=time.time()-start))
        pd.DataFrame(history).to_csv(cfg.out_dir / f'maskfeat_history_{tag}.csv', index=False)
        print(f'MaskFeat {tag} {epoch:02d}/{epochs} updates={step}/{cfg.maskfeat_updates} '
              f'lr={optimizer.param_groups[0]["lr"]:.2e} loss={epoch_loss:.6f} '
              f'time={time.time()-start:.0f}s')
    assert step == cfg.maskfeat_updates, (step, cfg.maskfeat_updates)
    state = {k: v.detach().cpu().clone() for k, v in ssl_model.backbone.state_dict().items()}
    del ssl_model, backbone, optimizer, loader
    gc.collect(); torch.cuda.empty_cache()
    return state


def build_v16_lpft_model(maskfeat_state, cfg, finetune_seed):
    """Build the V16-160 model with its seeded random head and adapted encoder."""
    seed_everything(finetune_seed)
    model = build_model(True, cfg)
    random_v16_head = {k: v.detach().clone() for k, v in model.head.state_dict().items()}
    model.load_state_dict(maskfeat_state, strict=True)
    model.head.load_state_dict(random_v16_head, strict=True)
    return model


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


def linear_probe_warmup(model, npy, meta, cfg, seed_value, tag):
    """Train only the random 40-class head while preserving MaskFeat features exactly."""
    seed_everything(seed_value)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    trainable_names = [name for name, _ in trainable]
    assert trainable_names == ['head.1.weight', 'head.1.bias'], trainable_names
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    expected_count = model.head[-1].in_features * N_CLASSES + N_CLASSES
    assert trainable_count == expected_count == 30_760

    frozen_versions = {name: parameter._version for name, parameter in model.named_parameters()
                       if not name.startswith('head.')}
    head_before = {name: parameter.detach().cpu().clone() for name, parameter in model.head.named_parameters()}
    loader = DataLoader(ThermalDataset(npy, meta, cfg, True), cfg.batch_size, shuffle=True,
                        drop_last=True, **common_loader_args(cfg))
    optimizer = torch.optim.AdamW((parameter for _, parameter in trainable),
                                  lr=cfg.linear_probe_lr,
                                  weight_decay=cfg.linear_probe_weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp and DEVICE.type == 'cuda')
    updates_per_epoch = len(loader) // cfg.grad_accum
    assert updates_per_epoch > 0
    epochs = math.ceil(cfg.linear_probe_updates / updates_per_epoch)
    step, history = 0, []
    for epoch in range(1, epochs + 1):
        # Frozen MViT features must be deterministic: disable DropPath while
        # keeping the classifier Dropout active.
        model.eval(); model.head.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = seen = 0; start = time.time()
        remaining_updates = cfg.linear_probe_updates - step
        usable_microbatches = min(updates_per_epoch, remaining_updates) * cfg.grad_accum
        for micro_step, (clip, label, _) in enumerate(loader):
            if micro_step >= usable_microbatches or step >= cfg.linear_probe_updates:
                break
            clip, label = clip.to(DEVICE, non_blocking=True), label.to(DEVICE, non_blocking=True)
            clip, soft_targets = video_mixup(clip, label, cfg)
            with torch.autocast('cuda', enabled=scaler.is_enabled()):
                logits = model(clip.permute(0, 2, 1, 3, 4))
                raw_loss = soft_target_cross_entropy(logits, soft_targets)
                loss = raw_loss / cfg.grad_accum
            assert torch.isfinite(raw_loss), f'non-finite LP loss at {tag} step {step}'
            scaler.scale(loss).backward()
            loss_sum += raw_loss.item() * len(label); seen += len(label)
            if (micro_step + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], cfg.grad_clip)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                step += 1
        epoch_loss = loss_sum / max(seen, 1)
        history.append(dict(stage=tag, epoch=epoch, optimizer_updates=step,
                            linear_probe_loss=epoch_loss, seconds=time.time()-start))
        pd.DataFrame(history).to_csv(cfg.out_dir / f'linear_probe_history_{tag}.csv', index=False)
        print(f'LP {tag} {epoch:02d}/{epochs} updates={step}/{cfg.linear_probe_updates} '
              f'loss={epoch_loss:.6f} time={time.time()-start:.0f}s')
    assert step == cfg.linear_probe_updates
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if not name.startswith('head.'))
    assert all(parameter._version == frozen_versions[name] for name, parameter in model.named_parameters()
               if name in frozen_versions), 'Frozen MaskFeat backbone changed during Linear Probe'
    head_delta = math.sqrt(sum(
        (parameter.detach().cpu() - head_before[name]).float().square().sum().item()
        for name, parameter in model.head.named_parameters()))
    assert head_delta > 0
    torch.save({name: value.detach().cpu() for name, value in model.head.state_dict().items()},
               cfg.out_dir / f'linear_probe_head_{tag}_TRAINING_ONLY.pt')
    print(f'LP {tag} QA: trainable={trainable_names}, params={trainable_count:,}, '
          f'head_delta_l2={head_delta:.6f}, frozen_backbone_exact=True')
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    del loader, optimizer, scaler, head_before
    gc.collect(); torch.cuda.empty_cache()
    return history


@torch.no_grad()
def infer_view_logits(model, meta, cfg, split, requested_views=('single', 'phase_a', 'phase_b')):
    model.eval()
    requested_views = tuple(requested_views)
    valid_views = {'single', 'phase_a', 'phase_b'}
    assert requested_views and set(requested_views) <= valid_views
    dataset = MultiViewDataset(cfg.cache_dir / f'{split}_thermal.npy', meta, cfg, False)
    loader = DataLoader(dataset, cfg.batch_size, shuffle=False, **common_loader_args(cfg))
    output_views = requested_views + (('dual',) if {'phase_a', 'phase_b'} <= set(requested_views) else ())
    records = {f'{view}_{kind}': np.zeros((len(meta), N_CLASSES), np.float32)
               for view in output_views
               for kind in ('original', 'flipped', 'hflip')}
    view_positions = {'single': 0, 'phase_a': 1, 'phase_b': 2}
    for views, _, idx in tqdm(loader, desc=f'{split}: {"/".join(requested_views)} + flip'):
        batch_logits = {}
        for view in requested_views:
            clip = views[:, view_positions[view]].to(DEVICE, non_blocking=True)
            with torch.autocast('cuda', enabled=cfg.amp and DEVICE.type == 'cuda'):
                original = model(clip.permute(0, 2, 1, 3, 4))
                flipped = model(torch.flip(clip, [-1]).permute(0, 2, 1, 3, 4))
                # Same add-then-half arithmetic as the champion evaluate().
                combined = (original + flipped) * .5
            for kind, logits in (('original', original), ('flipped', flipped), ('hflip', combined)):
                batch_logits[f'{view}_{kind}'] = logits
            del clip
        if 'dual' in output_views:
            for kind in ('original', 'flipped', 'hflip'):
                batch_logits[f'dual_{kind}'] = (batch_logits[f'phase_a_{kind}'] +
                                               batch_logits[f'phase_b_{kind}']) * .5
        for key, logits in batch_logits.items():
            records[key][idx.numpy()] = logits.float().cpu().numpy()
    del loader
    return records


def logits_to_probs(records):
    return {key: torch.from_numpy(value).softmax(1).numpy() for key, value in records.items()}


def enriched_meta(meta, windows):
    fields = windows[['clip_id', 'has_crop', 'x0', 'y0', 'x1', 'y1']].copy()
    return meta.drop(columns=['has_crop'], errors='ignore').merge(
        fields, on='clip_id', how='left', validate='one_to_one', sort=False)


def diagnostic_report(meta, probabilities, cfg, tag, provenance):
    """Descriptive error statistics, never automatic claims about visual causes."""
    folder = cfg.out_dir / 'diagnostics' / tag
    folder.mkdir(parents=True, exist_ok=True)
    meta = meta.reset_index(drop=True)
    assert not meta.clip_id.duplicated().any()
    truth = meta.action_id.to_numpy(dtype=int)
    label_idx = np.asarray([ID_TO_INDEX[int(x)] for x in truth])
    names = dict(zip(class_map.action_id.astype(int), class_map.action_name.astype(str)))
    summary, predictions = [], meta.copy()
    directional_pairs = {frozenset(('take_off_clothes', 'put_on_clothes')),
                         frozenset(('stand_up', 'sit_down'))}
    object_words = ('drink', 'medicine', 'temperature', 'phone', 'selfie', 'teeth',
                    'tableware', 'fruit', 'keyboard', 'write', 'pages', 'time', 'food')
    clean_name = lambda x: re.sub(r'^\d+_', '', names[int(x)]).lower()
    for key, probs in probabilities.items():
        assert probs.shape == (len(meta), N_CLASSES) and np.isfinite(probs).all()
        assert np.all(probs >= 0) and np.allclose(probs.sum(1), 1, atol=1e-4)
        np.save(folder / f'{key}_probs.npy', probs)
        pred_idx = probs.argmax(1)
        pred = LABEL_VALUES[pred_idx]
        correct = pred == truth
        predictions[f'{key}_prediction'] = pred
        predictions[f'{key}_confidence'] = probs.max(1)
        predictions[f'{key}_correct'] = correct
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        np.add.at(cm, (label_idx, pred_idx), 1)
        labels = [f'{x}:{names[int(x)]}' for x in LABEL_VALUES]
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(folder / f'{key}_confusion_counts.csv')
        pd.DataFrame(cm / np.maximum(cm.sum(1, keepdims=True), 1), index=labels,
                     columns=labels).to_csv(folder / f'{key}_confusion_row_normalized.csv')
        support = cm.sum(1)
        per_class = pd.DataFrame(dict(action_id=LABEL_VALUES, action_name=[names[int(x)] for x in LABEL_VALUES],
                                      support=support, correct=cm.diagonal(),
                                      recall=np.divide(cm.diagonal(), support, out=np.full(N_CLASSES, np.nan), where=support>0),
                                      precision=np.divide(cm.diagonal(), cm.sum(0), out=np.full(N_CLASSES, np.nan), where=cm.sum(0)>0)))
        per_class.to_csv(folder / f'{key}_per_class.csv', index=False)
        per_user = meta[['user']].assign(correct=correct).groupby('user').correct.agg(['count', 'sum', 'mean'])
        per_user.to_csv(folder / f'{key}_per_user.csv')
        pairs = []
        for i, j in zip(*np.where(cm > 0)):
            if i == j:
                continue
            a, b = int(LABEL_VALUES[i]), int(LABEL_VALUES[j])
            pairs.append(dict(true_id=a, predicted_id=b, true_name=names[a], predicted_name=names[b],
                              count=int(cm[i, j]),
                              direction_pair_candidate=frozenset((clean_name(a), clean_name(b))) in directional_pairs,
                              object_related_candidate=any(w in clean_name(a) or w in clean_name(b) for w in object_words)))
        pairs = pd.DataFrame(pairs, columns=['true_id', 'predicted_id', 'true_name', 'predicted_name',
                                             'count', 'direction_pair_candidate', 'object_related_candidate'])
        pairs.sort_values('count', ascending=False).to_csv(folder / f'{key}_confusion_pairs.csv', index=False)
        summary.append(dict(view=key, samples=len(meta), correct=int(correct.sum()), accuracy=float(correct.mean()),
                            macro_recall=float(per_class.recall.mean())))
    transitions = []
    for left, right, label in (('single_original', 'single_flipped', 'flip_alone'),
                               ('single_original', 'single_hflip', 'single_flip_ensemble'),
                               ('dual_original', 'dual_hflip', 'dual_flip_ensemble'),
                               ('single_hflip', 'dual_hflip', 'temporal_ensemble')):
        if left not in probabilities or right not in probabilities:
            continue
        before = predictions[f'{left}_correct'].to_numpy()
        after = predictions[f'{right}_correct'].to_numpy()
        status = np.select([~before & after, before & ~after, before & after],
                           ['rescued', 'harmed', 'both_correct'], default='both_wrong')
        effect = meta[['clip_id', 'user', 'action_id', 'action_name']].copy()
        effect['before_prediction'] = predictions[f'{left}_prediction']
        effect['after_prediction'] = predictions[f'{right}_prediction']
        effect['effect'] = status
        effect.to_csv(folder / f'{label}_effects.csv', index=False)
        transitions.append(dict(comparison=label, rescued=int((status == 'rescued').sum()),
                                harmed=int((status == 'harmed').sum()),
                                net_correct=int(after.sum()-before.sum())))
    scores = pd.DataFrame(summary)
    scores.to_csv(folder / 'view_scores.csv', index=False)
    pd.DataFrame(transitions, columns=['comparison', 'rescued', 'harmed', 'net_correct']).to_csv(
        folder / 'view_effect_summary.csv', index=False)
    predictions.to_parquet(folder / 'sample_predictions.parquet', index=False)
    np.save(folder / 'class_order.npy', LABEL_VALUES)
    (folder / 'provenance.json').write_text(json.dumps(dict(tag=tag, provenance=provenance,
        warning='Direction/object tags are label-name heuristics, not verified causes. '
                'Hand completeness and missed key moments require manual video review.'), indent=2), encoding='utf-8')
    print(f'[{tag}] view comparison:\n{scores.to_string(index=False)}')
    if transitions:
        print(pd.DataFrame(transitions).to_string(index=False))
    return scores, predictions, folder


def error_gallery(predictions, clips, cfg, folder, primary='single_hflip', champion=False):
    """Raw-frame GIFs + mapped crop + sampled-frame markers for HUMAN review."""
    lookup = {c.clip_id: c for c in clips}
    wrong = predictions.loc[~predictions[f'{primary}_correct']].sort_values(
        f'{primary}_confidence', ascending=False)
    chosen = []
    # Include harmful flips and temporal-view disagreements, then diverse confident errors.
    for other in ('single_original', 'dual_hflip'):
        column = f'{other}_correct'
        if column in wrong:
            chosen += wrong.loc[wrong[column]].head(2).clip_id.tolist()
    chosen += wrong.drop_duplicates('action_id').clip_id.tolist()
    chosen += wrong.clip_id.tolist()
    chosen = list(dict.fromkeys(chosen))[:cfg.qa_max_clips]
    assets = folder / 'qa_assets'
    assets.mkdir(exist_ok=True)
    entries, review = [], []
    for number, clip_id in enumerate(chosen):
        c = lookup.get(clip_id)
        row = predictions.loc[predictions.clip_id == clip_id].iloc[0]
        if c is None or not c.thermal:
            continue
        dense = pick_indices(len(c.thermal), 64)
        single = pick_indices(len(c.thermal), 32) if champion else dense[temporal_indices64('single')]
        a, b = dense[temporal_indices64('phase_a')], dense[temporal_indices64('phase_b')]
        # For champion QA include exact original32 source frames, not just approximations from64.
        timeline = sorted(set(dense.tolist() + single.tolist()))
        frames = []
        win = np.asarray([row.x0, row.y0, row.x1, row.y1], dtype=float)
        for source_idx in timeline:
            try:
                with Image.open(c.thermal[source_idx]) as image:
                    raw = image.convert('RGB')
                w, h = raw.size
                rect = (round(win[0]*w), round(win[1]*h), round(win[2]*w), round(win[3]*h))
                # Keep the gallery thumbnail compact; training/inference cache is cfg.img_size.
                crop = raw.crop(rect).resize((128, 128), Image.Resampling.BILINEAR)
                outlined = raw.copy()
                ImageDraw.Draw(outlined).rectangle(rect, outline='yellow', width=3)
                outlined = outlined.resize((320, 240), Image.Resampling.BILINEAR)
                canvas = Image.new('RGB', (456, 294), 'black')
                canvas.paste(outlined, (0, 0)); canvas.paste(crop, (324, 55))
                flags = ('S' if source_idx in single else '-')
                if not champion:
                    flags += (' A' if source_idx in a else ' -') + (' B' if source_idx in b else ' -')
                draw = ImageDraw.Draw(canvas)
                draw.text((4, 243), f'source frame {source_idx}/{len(c.thermal)-1} | selected: {flags}', fill='white')
                draw.text((4, 259), f'true={int(row.action_id)} pred={int(row[f"{primary}_prediction"])}', fill='white')
                draw.text((4, 275), f'Yellow: mapped IR crop | right: preview of {cfg.img_size}px input crop', fill='white')
                frames.append(canvas)
            except (OSError, ValueError) as exc:
                print(f'QA decode skipped {clip_id}, frame {source_idx}: {exc}')
        if not frames:
            continue
        name = f'error_{number:02d}'
        frames[0].save(assets / f'{name}.gif', save_all=True, append_images=frames[1:],
                       duration=120, loop=0)
        sheet = Image.new('RGB', (456*2, 294*2))
        for k, i in enumerate(pick_indices(len(frames), 4)):
            sheet.paste(frames[i], ((k%2)*456, (k//2)*294))
        sheet.save(assets / f'{name}.jpg', quality=90)
        entries.append(f'<h3>{html.escape(clip_id)}</h3><p>True: {html.escape(str(row.action_name))}; '
                       f'prediction ID: {int(row[f"{primary}_prediction"])}</p>'
                       f'<img src="qa_assets/{name}.gif"><br><a href="qa_assets/{name}.jpg">Contact sheet</a>')
        review.append(dict(clip_id=clip_id, true_id=int(row.action_id),
                           predicted_id=int(row[f'{primary}_prediction']),
                           full_person_visible='', hands_visible='', object_visible='',
                           key_action_between_sampled_frames='', reviewer_notes=''))
    pd.DataFrame(review, columns=['clip_id', 'true_id', 'predicted_id', 'full_person_visible',
                                  'hands_visible', 'object_visible', 'key_action_between_sampled_frames',
                                  'reviewer_notes']).to_csv(folder / 'manual_video_review.csv', index=False)
    page = '<!doctype html><meta charset="utf-8"><title>Thermal error review</title>'
    page += '<h1>Error review: ' + html.escape(folder.name) + '</h1>'
    page += ('<p>Uniformly sampled playback, NOT original timing. S=single-view input; A/B=temporal phases. '
             'Repeated raw frames are shown once. This is evidence for manual review, not an automatic '
             'judgment that hands are missing or a key action was skipped. Unsampled raw frames may still matter.</p>')
    page += ''.join(entries) if entries else '<p>No previewable errors available.</p>'
    (folder / 'error_gallery.html').write_text(page, encoding='utf-8')


def load_champion_boundary(meta, cfg):
    """Import only saved HELD-OUT predictions. Never evaluate the full-training model on its training users."""
    root = cfg.champion_artifact_dir
    status_path = cfg.out_dir / 'champion_diagnostic_status.json'
    prob_path, meta_path = root / 'selection_boundary_probs.npy', root / 'selection_boundary_predictions.parquet'
    if not prob_path.exists() or not meta_path.exists():
        status = dict(status='missing_champion_artifacts', searched=str(root),
                      missing=[str(p) for p in (prob_path, meta_path) if not p.exists()],
                      limitation='V19 diagnostics are NOT legacy-champion diagnostics. Original/flip champion logits '
                                 'cannot be recovered from averaged probabilities. Full-training weights '
                                 'must not be evaluated on their training subjects as held-out results.')
        status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
        print('CHAMPION DIAGNOSTICS: saved boundary artifacts absent; skipping, NOT substituting V19 or full-training predictions.')
        return None
    old = pd.read_parquet(meta_path)
    probs = np.load(prob_path, allow_pickle=False)
    assert not old.clip_id.duplicated().any(), 'Champion duplicate sample IDs'
    assert set(old.clip_id) == set(meta.clip_id), 'Champion must use exactly the same boundary clips'
    assert probs.shape == (len(old), N_CLASSES)
    # Original champion did not save a standalone class-order file. Its label_idx/action_id
    # pairs establish the order only if ALL classes occur. Otherwise fail closed.
    mapping = old[['label_idx', 'action_id']].drop_duplicates().sort_values('label_idx')
    assert np.array_equal(mapping.label_idx.to_numpy(), np.arange(N_CLASSES)), 'Cannot establish champion class order'
    assert np.array_equal(mapping.action_id.to_numpy(), LABEL_VALUES), 'Champion class order mismatch'
    order = old.set_index('clip_id').index.get_indexer(meta.clip_id)
    aligned = old.iloc[order].reset_index(drop=True)
    assert np.array_equal(aligned.action_id.to_numpy(), meta.action_id.to_numpy())
    assert np.array_equal(aligned.user.to_numpy(), meta.user.to_numpy())
    probs = probs[order]
    scores, predictions, folder = diagnostic_report(meta, {'single_hflip': probs}, cfg,
        'champion_saved_boundary', f'Saved boundary predictions from {root}; source model training provenance supplied by user')
    error_gallery(predictions, train_clips, cfg, folder, champion=True)
    status_path.write_text(json.dumps(dict(status='partial_saved_boundary_diagnostics', source=str(root),
        accuracy=float(scores.iloc[0].accuracy),
        limitation='Class/user/confusion/QA available; champion original-vs-flip unavailable without '
                   'separate logits or genuine boundary checkpoint. QA uses current unchanged IR-crop reconstruction; '
                   'confirm it matches the historical cache.'), indent=2), encoding='utf-8')
    return probs


def train_selection(meta, cfg):
    boundary_mask = meta.user.isin(cfg.boundary_users)
    assert set(meta.loc[boundary_mask, "user"].unique()) == set(cfg.boundary_users)
    train_part = meta.loc[~boundary_mask]
    val_part = meta.loc[boundary_mask]
    assert set(train_part.user).isdisjoint(set(val_part.user))
    assert set(val_part.user.unique()) == set(cfg.boundary_users)
    diagnostic_meta = enriched_meta(val_part, train_windows).reset_index(drop=True)
    champion_probs = load_champion_boundary(diagnostic_meta, cfg)
    # Leakage-safe: the self-supervised stage sees no held-out boundary clips.
    maskfeat_state = maskfeat_pretrain(train_part, cfg, cfg.seed + 11, 'boundary_train_users_only')
    npy = cfg.cache_dir / "train_thermal.npy"
    va_loader = DataLoader(ThermalDataset(npy, val_part, cfg, False),
                           cfg.batch_size, shuffle=False, **common_loader_args(cfg))
    print(f"selection boundary: train={len(train_part)}, val={len(val_part)}, "
          f"val users={sorted(val_part.user.unique())}")
    model = build_v16_lpft_model(maskfeat_state, cfg, cfg.seed + 1).to(DEVICE)
    del maskfeat_state
    # Save the exact post-initialization RNG stream that vanilla V16 would use.
    # LP runs on an isolated seed, then the saved stream is restored before the
    # full-finetune DataLoader, optimizer, EMA, augmentations, and DropPath start.
    baseline_rng = capture_rng_state()
    linear_probe_warmup(model, npy, train_part, cfg, cfg.seed + 1001, 'boundary_train_users_only')
    lp_val_loss, lp_val_acc, lp_val_probs = evaluate(model, va_loader, cfg, tta=cfg.tta_hflip)
    np.save(cfg.out_dir / 'linear_probe_boundary_probs.npy', lp_val_probs)
    print(f'LP boundary-only diagnostic: loss={lp_val_loss:.4f}, single+hflip={lp_val_acc:.4f}')
    restore_rng_state(baseline_rng)
    del baseline_rng, lp_val_probs
    tr_loader = DataLoader(ThermalDataset(npy, train_part, cfg, True), cfg.batch_size, shuffle=True,
                           drop_last=True, **common_loader_args(cfg))
    assert all(parameter.requires_grad for parameter in model.parameters())
    ema = ModelEmaV3(model, decay=cfg.ema_decay, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and DEVICE.type == "cuda")
    updates_per_epoch = len(tr_loader) // cfg.grad_accum
    assert updates_per_epoch > 0
    total_steps = updates_per_epoch * cfg.max_selection_epochs
    warmup_steps = updates_per_epoch * cfg.warmup_epochs
    step = 0; best_acc = -1.; best_epoch = 1; best_probs = best_state = None
    history = []
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
        history.append(dict(epoch=epoch, updates=step, linear_probe_updates=cfg.linear_probe_updates,
                            linear_probe_val_accuracy=lp_val_acc,
                            train_loss=loss_sum/max(seen, 1),
                            single_hflip_loss=val_loss, single_hflip_accuracy=val_acc))
        table = pd.DataFrame(history)
        table['single_hflip_ma3'] = table.single_hflip_accuracy.rolling(3).mean()
        table['single_hflip_ma5'] = table.single_hflip_accuracy.rolling(5).mean()
        table.to_csv(cfg.out_dir / 'boundary_epoch_history.csv', index=False)
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

    fp32_model = build_model(False, cfg).to(DEVICE)
    fp32_model.load_state_dict(best_state, strict=True)
    fp32_logits = infer_view_logits(fp32_model, val_part, cfg, 'train')
    fp32_probs = logits_to_probs(fp32_logits)
    fp32_scores, _, _ = diagnostic_report(diagnostic_meta, fp32_probs, cfg,
        'v19_boundary_fp32', f'V19 LP-FT EMA epoch {best_epoch}; held-out users {cfg.boundary_users}')
    np.savez_compressed(cfg.out_dir / 'boundary_fp32_view_logits.npz', **fp32_logits)
    del fp32_model, fp32_logits
    gc.collect(); torch.cuda.empty_cache()
    packed = quantize_state_dict(best_state)
    torch.save(dict(model_int8=packed, classes=LABEL_VALUES.tolist(), split='boundary',
                    train_users=sorted(train_part.user.unique()), val_users=list(cfg.boundary_users),
                    val_clip_ids=val_part.clip_id.tolist(), selected_epoch=best_epoch,
                    cache_frames=64, model_frames=32, image_size=160,
                    linear_probe_updates=cfg.linear_probe_updates,
                    linear_probe_lr=cfg.linear_probe_lr,
                    linear_probe_weight_decay=cfg.linear_probe_weight_decay,
                    architecture='torchvision_mvit_v2_s_relative_position_32x160'),
               cfg.out_dir / 'boundary_best_int8_DIAGNOSTIC_ONLY.pt')
    quantized_model = build_model(False, cfg)
    quantized_model.load_state_dict(dequantize_state_dict(packed))
    quantized_model = quantized_model.to(DEVICE).eval()
    int8_logits = infer_view_logits(quantized_model, val_part, cfg, 'train')
    int8_probs = logits_to_probs(int8_logits)
    np.savez_compressed(cfg.out_dir / 'boundary_int8_view_logits.npz', **int8_logits)
    scores, predictions, folder = diagnostic_report(diagnostic_meta, int8_probs, cfg,
        'v19_boundary_int8', f'V16-compatible INT8 quantization of V19 LP-FT boundary epoch {best_epoch}; no full-training subjects')
    error_gallery(predictions, train_clips, cfg, folder)
    indexed = scores.set_index('view')
    single_acc = float(indexed.loc['single_hflip', 'accuracy'])
    dual_acc = float(indexed.loc['dual_hflip', 'accuracy'])
    extra = int(indexed.loc['dual_hflip', 'correct'] - indexed.loc['single_hflip', 'correct'])
    # Keep inference identical to the verified V16-160 baseline. Dual is
    # diagnostic only; selecting it here would confound the LP-FT ablation.
    selected_view = 'single'
    int8_acc = single_acc
    print(f'DIAGNOSTICS V19: FP32 single+hflip={best_acc:.4f}, INT8 single+hflip={single_acc:.4f}, '
          f'INT8 dual+hflip={dual_acc:.4f}, extra_correct={extra}, selected_test_view={selected_view}')
    if champion_probs is not None:
        truth = diagnostic_meta.action_id.to_numpy()
        comparison = diagnostic_meta[['clip_id', 'user', 'action_id']].copy()
        comparison['champion_prediction'] = LABEL_VALUES[champion_probs.argmax(1)]
        for precision, probabilities in (('fp32', fp32_probs), ('int8', int8_probs)):
            for view in ('single', 'dual'):
                key = f'v19_{precision}_{view}'
                comparison[f'{key}_prediction'] = LABEL_VALUES[probabilities[f'{view}_hflip'].argmax(1)]
                old_ok = comparison.champion_prediction.to_numpy() == truth
                new_ok = comparison[f'{key}_prediction'].to_numpy() == truth
                comparison[f'{key}_effect'] = np.select([~old_ok & new_ok, old_ok & ~new_ok],
                                                       ['rescued', 'harmed'], default='unchanged_correctness')
        comparison.to_csv(cfg.out_dir / 'champion_vs_v19_boundary.csv', index=False)
    (cfg.out_dir / 'test_view_decision.json').write_text(json.dumps(dict(
        criterion='V19 isolates head-only Linear Probe before unchanged V16 full fine-tuning; deployment remains single+hflip',
        minimum_extra_correct=cfg.dual_min_extra_correct, extra_correct=extra,
        selected_view=selected_view, single_accuracy=single_acc, dual_accuracy=dual_acc), indent=2), encoding='utf-8')
    pd.DataFrame([{
        "split": "boundary",
        "val_users": ",".join(sorted(val_part.user.unique())),
        "best_epoch": best_epoch,
        "fp32_accuracy": best_acc,
        "int8_accuracy": int8_acc,
        "int8_single_accuracy": single_acc,
        "int8_dual_accuracy": dual_acc,
        "selected_test_view": selected_view,
        "quantization_delta_single": single_acc - best_acc,
        "optimizer_updates": best_epoch * selection_steps_per_epoch,
    }]).to_csv(cfg.out_dir / "diagnostics_boundary.csv", index=False)
    del best_state, packed, quantized_model, va_loader
    gc.collect(); torch.cuda.empty_cache()
    return best_epoch, best_acc, int8_acc, best_epoch * selection_steps_per_epoch, total_steps, warmup_steps, selected_view


def train_full(meta, target_updates, schedule_total_steps, schedule_warmup_steps, cfg):
    maskfeat_state = maskfeat_pretrain(meta, cfg, cfg.seed + 110, 'full_training_thermal_only')
    npy = cfg.cache_dir / "train_thermal.npy"
    model = build_v16_lpft_model(maskfeat_state, cfg, cfg.seed + 100).to(DEVICE)
    del maskfeat_state
    baseline_rng = capture_rng_state()
    linear_probe_warmup(model, npy, meta, cfg, cfg.seed + 1100, 'full_training_users')
    restore_rng_state(baseline_rng)
    del baseline_rng
    loader = DataLoader(ThermalDataset(npy, meta, cfg, True), cfg.batch_size, shuffle=True,
                        drop_last=True, **common_loader_args(cfg))
    assert all(parameter.requires_grad for parameter in model.parameters())
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
    assert step == target_updates, (step, target_updates)
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
    # V19 official inference is strictly the V16 single temporal view + hflip.
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    inference_start = time.perf_counter()
    records = infer_view_logits(model, meta, cfg, 'test', requested_views=('single',))
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start
    timing = dict(
        device=str(DEVICE), clips=len(meta), forwards_per_clip=2,
        total_seconds=inference_seconds,
        seconds_per_clip=inference_seconds / max(len(meta), 1),
        image_size=cfg.img_size, model_frames=cfg.model_frames,
    )
    (cfg.out_dir / 'test_inference_timing.json').write_text(
        json.dumps(timing, indent=2), encoding='utf-8')
    print(f"V19 test inference: {inference_seconds:.2f}s total, "
          f"{timing['seconds_per_clip']:.4f}s/clip, 2 forwards/clip on {DEVICE}")
    np.savez_compressed(cfg.out_dir / 'test_view_logits.npz', **records)
    probs = logits_to_probs(records)
    for name, values in probs.items():
        assert values.shape == (len(meta), N_CLASSES), (name, values.shape)
        if not np.isfinite(values).all() or not np.allclose(values.sum(1), 1.0, atol=1e-4):
            raise FloatingPointError(f'Invalid test probabilities for view {name}')
    del model
    gc.collect(); torch.cuda.empty_cache()
    return probs


best_epoch, selection_acc, selection_int8_acc, selection_updates, _, _, selected_view = train_selection(train_meta, CFG)
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
FINAL_MODEL = CFG.out_dir / "thermal_v19_mvitv2s_maskfeat_160_lpft_v16_int8.pt"
torch.save({
    "model_int8": packed,
    "architecture": "torchvision_mvit_v2_s_relative_position_32x160",
    "classes": LABEL_VALUES.tolist(),
    "cache_frames": CFG.cache_frames,
    "model_frames": CFG.model_frames,
    "temporal_input": "train: random one per ordered pair from uniform64; test: declared fixed schedule",
    "temporal_test_view": selected_view,
    "temporal_schedules": {v: temporal_indices64(v).tolist() for v in ('single', 'phase_a', 'phase_b')},
    "quantization": "original champion per-output-channel INT8 for all ndim>=2 floats",
    "pretraining": "Thermal-only MaskFeat-style masked final-token HOG prediction",
    "fine_tuning": "head-only Linear Probe followed by unchanged full-model V16-160 fine-tuning",
    "experiment_hypothesis": "a short head-only warm-up reduces early destructive gradients into the MaskFeat encoder",
    "cache_preprocessing": "raw Thermal -> fixed IR union crop margin 1.60 -> direct 160x160 resize",
    "pretraining_test_data_used": False,
    "maskfeat_updates": CFG.maskfeat_updates,
    "maskfeat_mask_ratio": CFG.maskfeat_ratio,
    "maskfeat_hog_bins": CFG.maskfeat_hog_bins,
    "maskfeat_final_grid": CFG.maskfeat_final_grid,
    "maskfeat_lr": CFG.maskfeat_lr,
    "linear_probe_updates": CFG.linear_probe_updates,
    "linear_probe_lr": CFG.linear_probe_lr,
    "linear_probe_weight_decay": CFG.linear_probe_weight_decay,
    "linear_probe_training_only": True,
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
print(f'Estimated ensemble={total_mib*1024**2/1e6:.2f} decimal MB. Other streams are estimates.')
if total_mib * 1024**2 > 100_000_000:
    print('WARNING: under 100 MiB does not establish compliance with decimal 100 MB.')
print('boundary_best_int8_DIAGNOSTIC_ONLY.pt is a training audit artifact, NOT an extra deployment model.')

test_views = predict_test(FINAL_MODEL, test_meta, CFG)
test_probs = test_views[f'{selected_view}_hflip']
np.save(CFG.out_dir / "thermal_test_probs.npy", test_probs)
predicted_ids = LABEL_VALUES[test_probs.argmax(1)]
test_csv = pd.read_csv(CFG.test_csv)
assert len(test_csv) == len(predicted_ids)
submission = test_csv[["path"]].copy()
submission["prediction"] = predicted_ids.astype(int)
assert submission['path'].tolist() == test_csv['path'].tolist()
assert submission['prediction'].notna().all() and submission['prediction'].isin(LABEL_VALUES).all()
assert list(submission.columns) == ['path', 'prediction']
SUBMISSION = CFG.work_root / "thermal_submission.csv"
submission.to_csv(SUBMISSION, index=False)
np.save(CFG.out_dir / 'thermal_test_probs_single.npy', test_views['single_hflip'])
submission.to_csv(CFG.out_dir / 'thermal_submission_single.csv', index=False)
test_export_meta = enriched_meta(test_meta, test_windows)
test_export_meta['path'] = test_csv['path'].to_numpy()
test_export_meta.to_parquet(CFG.out_dir / 'test_probability_metadata.parquet', index=False)
np.save(CFG.out_dir / 'class_order.npy', LABEL_VALUES)
(CFG.out_dir / 'run_config.json').write_text(json.dumps(vars(CFG), default=str, indent=2), encoding='utf-8')
# QA HTML uses relative image links: keep its complete directory together via this archive.
shutil.make_archive(str(CFG.out_dir / 'diagnostics_bundle'), 'zip', CFG.out_dir / 'diagnostics')
print(f'Selected main submission: {selected_view}+hflip; no spatial or temporal test ensemble')
print(f"wrote {SUBMISSION} ({len(submission)} rows, "
      f"{submission.prediction.nunique()}/{N_CLASSES} classes)")
print(f"saved ensemble probabilities: {CFG.out_dir / 'thermal_test_probs.npy'}")
print(f"saved final model: {FINAL_MODEL}")
print(submission.head().to_string(index=False))
