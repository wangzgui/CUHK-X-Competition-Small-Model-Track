from __future__ import annotations

# Paste this whole file into ONE Kaggle code cell. Internet must be enabled.
import gc, hashlib, json, math, os, random, re, shutil, subprocess, sys, time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s
from tqdm.auto import tqdm


def ensure_package(import_name: str, package_name: str | None = None):
    try:
        return __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name or import_name])
        return __import__(import_name)


if not torch.cuda.is_available() or torch.version.cuda is None:
    raise RuntimeError(
        "CUDA GPU is REQUIRED. This kernel is CPU-only. In Kaggle select a GPU accelerator, "
        "start a fresh session, and rerun. No preprocessing or training has been started."
    )
DEVICE = torch.device("cuda:0")
ensure_package("ultralytics")
ensure_package("timm")
from timm.utils import ModelEmaV3
from ultralytics import YOLO

Image.MAX_IMAGE_PIXELS = None
EPS = 1e-12
N_CLASSES = 40


@dataclass
class Config:
    # Existing Visual Fold-0 artifacts.
    visual_val_probs: Path = Path("/kaggle/input/datasets/zhuowamg/ir-depth-color/outputs/val_probs_fold0.npy")
    visual_val_meta: Path = Path("/kaggle/input/datasets/zhuowamg/ir-depth-color/outputs/val_pred_fold0.parquet")

    # New MViTv2-S Thermal deployment artifacts and cached Visual test probabilities.
    v11_weight_path: Path = Path("/kaggle/input/datasets/zhuowamg/v11model/thermal_v11_mvitv2s_maskfeat_pretrain_v8_finetune_int8.pt")
    v11_probs_root: Path = Path("/kaggle/input/datasets/zhuowamg/v11thermal")
    thermal_test_probs: str = "thermal_test_probs (1).npy"
    thermal_test_probs_single: str = "thermal_test_probs_single (1).npy"
    thermal_test_probs_dual: str = "thermal_test_probs_dual (1).npy"
    visual_test_probs_path: Path = Path("/kaggle/input/datasets/zhuowamg/newensembledata/visual_test_probs.npy")

    # Raw data used only to train one Thermal model aligned to Visual Fold-0.
    thermal_train_root: Path = Path("/kaggle/input/datasets/zhuowamg/thermal/Thermal")
    ir_train_root: Path = Path("/kaggle/input/datasets/zhuowamg/irdata/IR")
    test_root: Path = Path("/kaggle/input/datasets/zhuowamg/cuhksmall/CUHK-X_dataset/Small-Model-Track/Testing/data/small_model_track_test/small_model_track_test")
    test_csv: Path = Path("/kaggle/input/datasets/zhuowamg/cuhksmall/CUHK-X_dataset/Small-Model-Track/Testing/test_file/test.csv")
    class_map_csv: Path = Path("/kaggle/input/datasets/zhuowamg/class-mapping1/class_mapping.csv")

    # Optional hard fallback: artifacts that generated the proven 0.74626 submission.
    old_root: Path = Path("/kaggle/input/datasets/zhuowamg/data-ooo")
    old_oof_name: str = "fixed_blend_oof_probs.npy"
    old_test_name: str = "fixed_blend_test_probs.npy"
    old_meta_name: str = "newthermal_aligned_meta.parquet"

    work_root: Path = Path("/kaggle/working")
    cache_root: Path = Path("/kaggle/temp/v11_vt_aligned_v1")
    output_root: Path = Path("/kaggle/working/v11_visual_thermal_fusion")

    cache_frames: int = 64
    model_frames: int = 32
    image_size: int = 128
    batch_size: int = 2
    grad_accum: int = 6
    num_workers: int = 2
    cache_workers: int = 4
    seed: int = 2026
    maskfeat_updates: int = 1870
    maskfeat_lr: float = 2e-5
    maskfeat_min_lr: float = 2e-7
    maskfeat_warmup_steps: int = 187
    maskfeat_weight_decay: float = 0.05
    maskfeat_ratio: float = 0.40
    maskfeat_hog_bins: int = 9
    maskfeat_final_grid: tuple[int, int, int] = (16, 4, 4)
    target_updates: int = 3740
    schedule_total_steps: int = 5610
    schedule_warmup_steps: int = 374
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    label_smoothing: float = 0.05
    ema_decay: float = 0.995
    amp: bool = True
    dropout: float = 0.30
    hflip_p: float = 0.50
    crop_scale: tuple[float, float] = (0.80, 1.0)
    calibration_p: float = 0.80
    brightness_delta: float = 0.08
    contrast_delta: float = 0.10
    mixup_p: float = 0.30
    mixup_alpha: float = 0.20
    validation_users: tuple[str, ...] = ("user9", "user16", "user18")

    person_det_frames: int = 8
    person_det_conf: float = 0.25
    person_crop_margin: float = 1.60
    person_crop_min_side: float = 0.45

    visual_temperatures: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    thermal_temperatures: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    # Official-informed primary search cannot collapse to a zero-Thermal solution.
    thermal_weights: tuple[float, ...] = tuple(np.round(np.arange(0.15, 0.451, 0.025), 3))
    anchored_weights: tuple[float, ...] = tuple(np.round(np.arange(0.25, 0.351, 0.025), 3))
    max_reasonable_stage_epoch_seconds: float = 1200.0

    @property
    def cache_dir(self):
        return self.cache_root / "T64_S128_IRcrop_m160_min045"


CFG = Config()
CFG.cache_dir.mkdir(parents=True, exist_ok=True)
CFG.output_root.mkdir(parents=True, exist_ok=True)
assert (CFG.cache_frames, CFG.model_frames, CFG.batch_size, CFG.grad_accum) == (64, 32, 2, 6)


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_weight(path: Path):
    candidates = [path]
    # Keep a small compatibility fallback for Kaggle dataset renaming.
    if path.name.endswith(".pt"):
        candidates.append(path.with_name(path.name.replace(" (1).pt", ".pt")))
    else:
        candidates.extend([path, path.with_suffix(".pt")])
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError("Thermal weight not found; tried:\n" + "\n".join(map(str, candidates)))


THERMAL_WEIGHT = resolve_weight(CFG.v11_weight_path)
required = [CFG.visual_val_probs, CFG.visual_val_meta, CFG.thermal_train_root, CFG.ir_train_root,
            CFG.test_root, CFG.test_csv, CFG.class_map_csv,
            CFG.v11_probs_root / CFG.thermal_test_probs,
            CFG.v11_probs_root / CFG.thermal_test_probs_single,
            CFG.v11_probs_root / CFG.thermal_test_probs_dual,
            CFG.visual_test_probs_path]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError("Missing required input(s):\n" + "\n".join(missing))
print("torch", torch.__version__, "CUDA", torch.version.cuda, "device", DEVICE,
      "GPU count", torch.cuda.device_count(), "GPU0", torch.cuda.get_device_name(0))
if torch.cuda.device_count() > 1:
    print("INFO: exact-recipe training uses cuda:0 only; the second T4 is intentionally unused.")
print("new Thermal weight", THERMAL_WEIGHT, "SHA256", sha256(THERMAL_WEIGHT))


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


deployment_ckpt = safe_torch_load(THERMAL_WEIGHT)
if deployment_ckpt.get("architecture") != "torchvision_mvit_v2_s_relative_position_32x128":
    raise ValueError(f"Unexpected Thermal architecture: {deployment_ckpt.get('architecture')}")
if deployment_ckpt.get("classes") != list(range(N_CLASSES)):
    raise ValueError("Thermal checkpoint class order is not 0..39")
if int(deployment_ckpt.get("full_train_updates", -1)) != CFG.target_updates:
    raise ValueError("Thermal checkpoint was not trained for the expected 3740 updates")
if int(deployment_ckpt.get("maskfeat_updates", -1)) != CFG.maskfeat_updates:
    raise ValueError("Thermal checkpoint does not contain the expected 1870-update MaskFeat pretraining")
if bool(deployment_ckpt.get("pretraining_test_data_used", True)):
    raise ValueError("Thermal checkpoint metadata does not prove that test data was excluded from pretraining")
del deployment_ckpt

class_map = pd.read_csv(CFG.class_map_csv, encoding="utf-8-sig")
class_map["action_id"] = class_map["action_id"].astype(int)
LABEL_VALUES = np.sort(class_map.action_id.unique())
if not np.array_equal(LABEL_VALUES, np.arange(N_CLASSES)):
    raise ValueError(f"Expected action IDs 0..39, got {LABEL_VALUES.tolist()}")
NAME_TO_ID = dict(zip(class_map.action_name.astype(str), class_map.action_id.astype(int)))


_LAST_NUMBER = re.compile(r"(\d+)(?=\.[^.]+$)")


def frame_files(folder: Path):
    if not folder.is_dir(): return []
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


def build_train_index():
    clips = []
    for trial_dir in tqdm(sorted(CFG.thermal_train_root.glob("*/*/*")), desc="index Thermal train"):
        if not trial_dir.is_dir(): continue
        action, user, trial = trial_dir.parts[-3:]
        if action not in NAME_TO_ID: continue
        thermal = frame_files(trial_dir)
        if not thermal: continue
        ir = frame_files(CFG.ir_train_root / action / user / trial)
        clips.append(Clip(f"{action}/{user}/{trial}", action, NAME_TO_ID[action], user, trial, thermal, ir))
    if not clips: raise RuntimeError("No Thermal training clips found")
    return clips


def pick_indices(n_src: int, n_out: int):
    return np.zeros(0, dtype=np.int64) if n_src <= 0 else np.linspace(0, n_src - 1, n_out).round().astype(np.int64)


def ensure_yolo():
    target = CFG.work_root / "yolo11n.pt"
    if target.exists(): return target
    old = Path.cwd()
    try:
        os.chdir(CFG.work_root)
        detector = YOLO("yolo11n.pt")
        source = Path(getattr(detector, "ckpt_path", "yolo11n.pt"))
        if source.exists() and source.resolve() != target.resolve(): shutil.copy2(source, target)
    finally:
        os.chdir(old)
    if not target.exists(): raise FileNotFoundError("YOLO download did not create /kaggle/working/yolo11n.pt")
    return target


def crop_window(boxes, width, height):
    x0, y0 = boxes[:, 0].min()/width, boxes[:, 1].min()/height
    x1, y1 = boxes[:, 2].max()/width, boxes[:, 3].max()/height
    cx, cy = (x0+x1)/2, (y0+y1)/2
    side = max((x1-x0)*width, (y1-y0)*height) * CFG.person_crop_margin
    side = max(side, CFG.person_crop_min_side * max(width, height))
    hx, hy = side/width/2, side/height/2
    return max(cx-hx, 0.), max(cy-hy, 0.), min(cx+hx, 1.), min(cy+hy, 1.)


def detect_windows(clips):
    cache = CFG.cache_dir / "train_crop_windows.parquet"
    if cache.exists():
        table = pd.read_parquet(cache)
        if table.clip_id.tolist() == [c.clip_id for c in clips]:
            print("reusing Thermal crop windows"); return table
    detector = YOLO(str(ensure_yolo()))
    rows = []
    for clip in tqdm(clips, desc="YOLO IR crops for Thermal"):
        probe = [clip.ir[i] for i in sorted(set(pick_indices(len(clip.ir), CFG.person_det_frames).tolist()))] if clip.ir else []
        boxes, width, height = [], None, None
        try:
            results = detector.predict(probe, classes=[0], conf=CFG.person_det_conf, verbose=False,
                                       device=0 if DEVICE.type == "cuda" else "cpu") if probe else []
            for result in results:
                height, width = result.orig_shape
                if len(result.boxes): boxes.append(result.boxes.xyxy[result.boxes.conf.argmax()].detach().cpu().numpy())
        except Exception as exc:
            print("YOLO warning", clip.clip_id, repr(exc))
        if boxes and width and height:
            w = crop_window(np.asarray(boxes), width, height)
            rows.append(dict(clip_id=clip.clip_id, has_crop=True, x0=w[0], y0=w[1], x1=w[2], y1=w[3]))
        else:
            rows.append(dict(clip_id=clip.clip_id, has_crop=False, x0=0., y0=0., x1=1., y1=1.))
    table = pd.DataFrame(rows); table.to_parquet(cache, index=False)
    print("Thermal crop success", f"{table.has_crop.mean():.1%}")
    del detector; gc.collect(); torch.cuda.empty_cache()
    return table


_MM = None
_CACHE_SHAPE = None


def _worker_init(path, shape):
    global _MM, _CACHE_SHAPE
    _CACHE_SHAPE = shape
    _MM = np.memmap(path, dtype=np.uint8, mode="r+", shape=shape)


def _read_thermal(path, win):
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if win is not None:
                width, height = image.size
                image = image.crop((round(win[0]*width), round(win[1]*height), round(win[2]*width), round(win[3]*height)))
            resampling = getattr(Image, "Resampling", Image).BILINEAR
            arr = np.asarray(image.resize((CFG.image_size, CFG.image_size), resampling), dtype=np.uint8)
        return None if arr.max() == 0 else arr
    except Exception:
        return None


def _cache_one(job):
    row, clip, win = job
    buf = np.zeros(_CACHE_SHAPE[1:], dtype=np.uint8); bad = 0
    for t, src in enumerate(pick_indices(len(clip.thermal), CFG.cache_frames)):
        arr = _read_thermal(clip.thermal[src], win)
        if arr is None: bad += 1
        else: buf[t] = arr.transpose(2, 0, 1)
    _MM[row] = buf
    return dict(row=row, clip_id=clip.clip_id, action_name=clip.action_name, action_id=clip.action_id,
                user=clip.user, trial=clip.trial, bad_frames=bad, has_crop=win is not None)


def build_cache(clips, windows):
    npy = CFG.cache_dir / "train_thermal.npy"; meta_path = CFG.cache_dir / "train_meta.parquet"
    shape = (len(clips), CFG.cache_frames, 3, CFG.image_size, CFG.image_size)
    expected = int(np.prod(shape))
    if npy.exists() and meta_path.exists() and npy.stat().st_size == expected:
        meta = pd.read_parquet(meta_path)
        if meta.clip_id.tolist() == [c.clip_id for c in clips]:
            print("reusing 64-frame Thermal cache"); return meta
    available = shutil.disk_usage(CFG.cache_dir).free + (npy.stat().st_size if npy.exists() else 0)
    if available < expected + 256*1024**2:
        raise RuntimeError(f"Insufficient temporary disk: need {expected/1024**3:.2f} GiB plus reserve")
    print("building cache", shape, f"{expected/1024**3:.2f} GiB")
    np.memmap(npy, dtype=np.uint8, mode="w+", shape=shape).flush()
    wins = [(r.x0,r.y0,r.x1,r.y1) if r.has_crop else None for r in windows.itertuples(index=False)]
    records = []
    with Pool(CFG.cache_workers, initializer=_worker_init, initargs=(str(npy), shape)) as pool:
        for record in tqdm(pool.imap_unordered(_cache_one, zip(range(len(clips)), clips, wins), chunksize=8),
                           total=len(clips), desc="cache Thermal train"):
            records.append(record)
    meta = pd.DataFrame(records).sort_values("row").reset_index(drop=True)
    meta.to_parquet(meta_path, index=False)
    return meta


KINETICS_MEAN = torch.tensor([0.45, 0.45, 0.45]).view(1,3,1,1)
KINETICS_STD = torch.tensor([0.225, 0.225, 0.225]).view(1,3,1,1)


def temporal_indices(train=False):
    if train: return np.arange(0,64,2,dtype=np.int64) + np.random.randint(0,2,size=32)
    return np.linspace(0,63,32).round().astype(np.int64)


class ThermalDataset(Dataset):
    def __init__(self, npy_path, meta, train):
        self.path = str(npy_path); self.meta = meta.reset_index(drop=True); self.train = train
        self.rows = self.meta.row.to_numpy(); self.labels = self.meta.action_id.to_numpy(np.int64)
        per_row = CFG.cache_frames * 3 * CFG.image_size * CFG.image_size
        n_rows, rem = divmod(os.path.getsize(self.path), per_row)
        if rem: raise ValueError("Thermal cache has invalid byte size")
        self.shape = (n_rows, CFG.cache_frames, 3, CFG.image_size, CFG.image_size); self._mm = None
    @property
    def mm(self):
        if self._mm is None: self._mm = np.memmap(self.path, dtype=np.uint8, mode="r", shape=self.shape)
        return self._mm
    def __len__(self): return len(self.meta)
    def __getitem__(self, idx):
        frames = np.asarray(self.mm[self.rows[idx], temporal_indices(self.train)]).copy()
        clip = torch.from_numpy(frames).float().div_(255.)
        if self.train:
            if random.random() < CFG.hflip_p: clip = torch.flip(clip, [-1])
            scale = random.uniform(*CFG.crop_scale); side = int(round(CFG.image_size*scale))
            top = random.randint(0, CFG.image_size-side); left = random.randint(0, CFG.image_size-side)
            clip = F.interpolate(clip[:,:,top:top+side,left:left+side], size=(CFG.image_size,CFG.image_size),
                                 mode="bilinear", align_corners=False)
            if random.random() < CFG.calibration_p:
                contrast = 1. + random.uniform(-CFG.contrast_delta, CFG.contrast_delta)
                brightness = random.uniform(-CFG.brightness_delta, CFG.brightness_delta)
                clip = ((clip-.5)*contrast+.5+brightness).clamp_(0.,1.)
        return clip.sub_(KINETICS_MEAN).div_(KINETICS_STD), int(self.labels[idx]), idx


def build_model(pretrained: bool):
    model = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None)
    conv = model.conv_proj
    def out_size(size,kernel,stride,padding,dilation):
        return (size+2*padding-dilation*(kernel-1)-1)//stride+1
    model.pos_encoding.temporal_size = out_size(CFG.model_frames, conv.kernel_size[0], conv.stride[0], conv.padding[0], conv.dilation[0])
    model.pos_encoding.spatial_size = (
        out_size(CFG.image_size, conv.kernel_size[1], conv.stride[1], conv.padding[1], conv.dilation[1]),
        out_size(CFG.image_size, conv.kernel_size[2], conv.stride[2], conv.padding[2], conv.dilation[2]))
    in_features = model.head[-1].in_features
    model.head = nn.Sequential(nn.Dropout(CFG.dropout), nn.Linear(in_features, N_CLASSES))
    return model


class ThermalMaskFeat(nn.Module):
    """Exact V11 masked final-token HOG prediction head."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        patch_dim = backbone.conv_proj.out_channels
        feature_dim = backbone.head[-1].in_features
        self.mask_token = nn.Parameter(torch.zeros(1, 1, patch_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.predictor = nn.Linear(feature_dim, CFG.maskfeat_hog_bins)
        nn.init.trunc_normal_(self.predictor.weight, std=0.02)
        nn.init.zeros_(self.predictor.bias)

    def forward(self, video, final_mask):
        x = self.backbone.conv_proj(video)
        batch, _, t, h, w = x.shape
        initial_grid = (self.backbone.pos_encoding.temporal_size, *self.backbone.pos_encoding.spatial_size)
        if (t, h, w) != initial_grid:
            raise RuntimeError(f"Unexpected MViT patch grid {(t,h,w)} != {initial_grid}")
        expanded = F.interpolate(final_mask[:, None].float(), size=(t, h, w), mode="nearest")[:, 0].bool()
        x = x.flatten(2).transpose(1, 2)
        x = torch.where(expanded.flatten(1).unsqueeze(-1), self.mask_token.expand(batch, x.shape[1], -1), x)
        x = self.backbone.pos_encoding(x)
        thw = initial_grid
        for block in self.backbone.blocks:
            x, thw = block(x, thw)
        x = self.backbone.norm(x)
        if tuple(thw) != tuple(CFG.maskfeat_final_grid):
            raise RuntimeError(f"Unexpected final grid {thw}")
        return self.predictor(x[:, 1:])


@torch.no_grad()
def thermal_hog_targets(normalized_clip):
    mean = KINETICS_MEAN.to(normalized_clip.device); std = KINETICS_STD.to(normalized_clip.device)
    raw = (normalized_clip * std + mean).clamp(0, 1)
    gray = 0.299*raw[:, :, 0] + 0.587*raw[:, :, 1] + 0.114*raw[:, :, 2]
    gx = F.pad(gray[..., 2:] - gray[..., :-2], (1, 1, 0, 0))
    gy = F.pad(gray[..., 2:, :] - gray[..., :-2, :], (0, 0, 1, 1))
    magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)
    angle = torch.remainder(torch.atan2(gy, gx), math.pi)
    bins = torch.clamp((angle*CFG.maskfeat_hog_bins/math.pi).long(), 0, CFG.maskfeat_hog_bins-1)
    hist = F.one_hot(bins, CFG.maskfeat_hog_bins).permute(0, 4, 1, 2, 3).float()
    hist.mul_(magnitude[:, None])
    tf, hf, wf = CFG.maskfeat_final_grid
    _, _, t, h, w = hist.shape
    if t % tf or h % hf or w % wf:
        raise RuntimeError("HOG grid is not divisible by the final MViT grid")
    pooled = F.avg_pool3d(hist, kernel_size=(t//tf, h//hf, w//wf), stride=(t//tf, h//hf, w//wf))
    targets = pooled.permute(0, 2, 3, 4, 1).reshape(len(raw), -1, CFG.maskfeat_hog_bins)
    return F.normalize(targets, dim=-1, eps=1e-6)


def final_grid_mask(batch_size):
    n_tokens = int(np.prod(CFG.maskfeat_final_grid))
    n_masked = max(1, min(n_tokens-1, int(round(CFG.maskfeat_ratio*n_tokens))))
    chosen = torch.rand(batch_size, n_tokens, device=DEVICE).topk(n_masked, dim=1).indices
    mask = torch.zeros(batch_size, n_tokens, dtype=torch.bool, device=DEVICE)
    mask.scatter_(1, chosen, True)
    return mask.view(batch_size, *CFG.maskfeat_final_grid)


def maskfeat_lr_at(step):
    if step < CFG.maskfeat_warmup_steps:
        return CFG.maskfeat_lr*(step+1)/max(CFG.maskfeat_warmup_steps, 1)
    progress = min(max((step-CFG.maskfeat_warmup_steps)/max(CFG.maskfeat_updates-CFG.maskfeat_warmup_steps,1),0.),1.)
    return CFG.maskfeat_min_lr + .5*(CFG.maskfeat_lr-CFG.maskfeat_min_lr)*(1+math.cos(math.pi*progress))


def maskfeat_pretrain(train_meta):
    """Leakage-safe V11 adaptation: aligned validation users are absent upstream."""
    if set(train_meta.user).intersection(CFG.validation_users):
        raise ValueError("MaskFeat leakage: aligned validation user entered SSL pretraining")
    seed_everything(CFG.seed+110)
    loader = DataLoader(ThermalDataset(CFG.cache_dir/"train_thermal.npy", train_meta, True),
                        CFG.batch_size, shuffle=True, drop_last=True, **loader_args())
    ssl_model = ThermalMaskFeat(build_model(True)).to(DEVICE)
    optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=CFG.maskfeat_lr,
                                  weight_decay=CFG.maskfeat_weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    updates_per_epoch = len(loader)//CFG.grad_accum
    epochs = math.ceil(CFG.maskfeat_updates/updates_per_epoch)
    step=0; history=[]
    for epoch in range(1, epochs+1):
        ssl_model.train(); optimizer.zero_grad(set_to_none=True); start=time.time(); loss_sum=vectors=0
        usable=min(updates_per_epoch, CFG.maskfeat_updates-step)*CFG.grad_accum
        for micro,(clip,_,_) in enumerate(loader):
            if micro>=usable or step>=CFG.maskfeat_updates: break
            clip=clip.to(DEVICE,non_blocking=True); mask=final_grid_mask(len(clip)); targets=thermal_hog_targets(clip)
            flat=mask.flatten(1)
            with torch.autocast("cuda", enabled=True):
                pred=ssl_model(clip.permute(0,2,1,3,4),mask)
                raw=F.smooth_l1_loss(pred[flat],targets[flat]); loss=raw/CFG.grad_accum
            if not torch.isfinite(raw): raise FloatingPointError(f"Non-finite MaskFeat loss at update {step}")
            if epoch==1 and micro==0:
                print("MaskFeat QA", "input",tuple(clip.shape),"prediction",tuple(pred.shape),
                      "masked",int(flat[0].sum()),"target_norm",float(targets[flat].float().norm(dim=-1).mean()))
            scaler.scale(loss).backward(); count=int(flat.sum()); loss_sum+=raw.item()*count; vectors+=count
            if (micro+1)%CFG.grad_accum==0:
                for group in optimizer.param_groups: group["lr"]=maskfeat_lr_at(step)
                scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(ssl_model.parameters(),CFG.grad_clip)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); step+=1
        elapsed=time.time()-start
        history.append(dict(epoch=epoch,updates=step,lr=optimizer.param_groups[0]["lr"],
                            maskfeat_loss=loss_sum/max(vectors,1),seconds=elapsed))
        pd.DataFrame(history).to_csv(CFG.output_root/"maskfeat_aligned_history.csv",index=False)
        print(f"MaskFeat aligned {epoch:02d}/{epochs} updates={step}/{CFG.maskfeat_updates} "
              f"loss={loss_sum/max(vectors,1):.6f} time={elapsed:.0f}s")
        if elapsed > CFG.max_reasonable_stage_epoch_seconds:
            raise RuntimeError(f"MaskFeat epoch took {elapsed:.0f}s; GPU acceleration is not functioning normally")
    if step != CFG.maskfeat_updates: raise RuntimeError("MaskFeat update count mismatch")
    state={k:v.detach().cpu().clone() for k,v in ssl_model.backbone.state_dict().items()}
    del ssl_model,optimizer,loader; gc.collect(); torch.cuda.empty_cache()
    return state


def build_finetune_model(maskfeat_state):
    seed_everything(CFG.seed+100)
    model=build_model(True)
    original_head={k:v.detach().clone() for k,v in model.head.state_dict().items()}
    model.load_state_dict(maskfeat_state,strict=True)
    model.head.load_state_dict(original_head,strict=True)
    return model


def lr_at(step):
    if step < CFG.schedule_warmup_steps: return CFG.lr*(step+1)/CFG.schedule_warmup_steps
    progress = (step-CFG.schedule_warmup_steps)/max(CFG.schedule_total_steps-CFG.schedule_warmup_steps,1)
    return CFG.min_lr + .5*(CFG.lr-CFG.min_lr)*(1+math.cos(math.pi*progress))


def video_mixup(clip, label):
    targets = F.one_hot(label, N_CLASSES).float()
    targets = targets*(1-CFG.label_smoothing) + CFG.label_smoothing/N_CLASSES
    if len(clip)>1 and random.random()<CFG.mixup_p:
        lam = max(float(np.random.beta(CFG.mixup_alpha,CFG.mixup_alpha)), .5)
        perm = torch.randperm(len(clip), device=clip.device)
        clip = clip*lam + clip[perm]*(1-lam); targets = targets*lam + targets[perm]*(1-lam)
    return clip, targets


def soft_ce(logits, targets): return -(targets*F.log_softmax(logits,dim=1)).sum(1).mean()


def loader_args(): return dict(num_workers=CFG.num_workers, pin_memory=True, persistent_workers=CFG.num_workers>0)


def train_aligned(meta):
    seed_everything(CFG.seed+100)
    train_meta = meta.loc[~meta.user.isin(CFG.validation_users)].reset_index(drop=True)
    val_meta = meta.loc[meta.user.isin(CFG.validation_users)].reset_index(drop=True)
    if set(val_meta.user.unique()) != set(CFG.validation_users): raise ValueError("Aligned validation users are incomplete")
    if set(train_meta.user).intersection(set(val_meta.user)):
        raise ValueError("Subject leakage between aligned training and validation")
    maskfeat_state = maskfeat_pretrain(train_meta)
    seed_everything(CFG.seed+100)
    npy = CFG.cache_dir / "train_thermal.npy"
    train_loader = DataLoader(ThermalDataset(npy,train_meta,True), CFG.batch_size, shuffle=True, drop_last=True, **loader_args())
    updates_per_epoch = len(train_loader)//CFG.grad_accum
    epochs = math.ceil(CFG.target_updates/updates_per_epoch)
    print(f"aligned Thermal train={len(train_meta)} val={len(val_meta)} updates/epoch={updates_per_epoch} epochs={epochs}")
    model = build_finetune_model(maskfeat_state).to(DEVICE)
    del maskfeat_state; gc.collect(); torch.cuda.empty_cache()
    ema = ModelEmaV3(model,decay=CFG.ema_decay,device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),lr=CFG.lr,weight_decay=CFG.weight_decay)
    scaler = torch.amp.GradScaler("cuda",enabled=CFG.amp and DEVICE.type=="cuda")
    step=0; history=[]
    for epoch in range(1,epochs+1):
        model.train(); optimizer.zero_grad(set_to_none=True); start=time.time(); loss_sum=seen=0
        usable = min(updates_per_epoch, CFG.target_updates-step)*CFG.grad_accum
        for micro,(clip,label,_) in enumerate(train_loader):
            if micro>=usable or step>=CFG.target_updates: break
            clip=clip.to(DEVICE,non_blocking=True); label=label.to(DEVICE,non_blocking=True)
            clip,targets=video_mixup(clip,label)
            with torch.autocast("cuda",enabled=scaler.is_enabled()):
                raw=soft_ce(model(clip.permute(0,2,1,3,4)),targets); loss=raw/CFG.grad_accum
            scaler.scale(loss).backward(); loss_sum+=raw.item()*len(label); seen+=len(label)
            if (micro+1)%CFG.grad_accum==0:
                for group in optimizer.param_groups: group["lr"]=lr_at(step)
                scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(),CFG.grad_clip)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                ema.update(model,step=step); step+=1
        elapsed=time.time()-start
        history.append(dict(epoch=epoch,updates=step,lr=optimizer.param_groups[0]["lr"],
                            train_loss=loss_sum/max(seen,1),seconds=elapsed))
        pd.DataFrame(history).to_csv(CFG.output_root/"aligned_training_history.csv",index=False)
        print(f"aligned {epoch:02d}/{epochs} updates={step}/{CFG.target_updates} lr={optimizer.param_groups[0]['lr']:.2e} "
              f"loss={loss_sum/max(seen,1):.4f} time={elapsed:.0f}s")
        if elapsed > CFG.max_reasonable_stage_epoch_seconds:
            raise RuntimeError(f"Supervised epoch took {elapsed:.0f}s; GPU acceleration is not functioning normally")
    if step != CFG.target_updates: raise RuntimeError(f"Expected {CFG.target_updates} updates, got {step}")
    state={k:v.detach().cpu().clone() for k,v in ema.module.state_dict().items()}
    del model,ema,optimizer,train_loader; gc.collect(); torch.cuda.empty_cache()
    return state,val_meta


def quantize_state_dict(state):
    packed={}
    for key,value in tqdm(state.items(),desc="INT8 pack aligned model"):
        if value.is_floating_point() and value.ndim>=2:
            weight=value.float(); dims=tuple(range(1,weight.ndim))
            scale=(weight.abs().amax(dim=dims,keepdim=True)/127).clamp_min(1e-12)
            packed[key]={"q":(weight/scale).round().clamp(-127,127).to(torch.int8),"scale":scale.half()}
        else: packed[key]=value.half() if value.is_floating_point() else value
    return packed


def dequantize_state_dict(packed):
    return {k:(v["q"].float()*v["scale"].float() if isinstance(v,dict) else (v.float() if v.is_floating_point() else v))
            for k,v in packed.items()}


@torch.no_grad()
def evaluate_aligned(model,val_meta):
    dataset=ThermalDataset(CFG.cache_dir/"train_thermal.npy",val_meta,False)
    loader=DataLoader(dataset,CFG.batch_size,shuffle=False,**loader_args())
    probs=np.zeros((len(dataset),N_CLASSES),np.float32)
    model.eval()
    for clip,_,idx in tqdm(loader,desc="aligned Thermal INT8 validation"):
        clip=clip.to(DEVICE,non_blocking=True)
        with torch.autocast("cuda",enabled=CFG.amp and DEVICE.type=="cuda"):
            logits=model(clip.permute(0,2,1,3,4))
            logits=(logits+model(torch.flip(clip,[-1]).permute(0,2,1,3,4)))*.5
        probs[idx.numpy()]=logits.float().softmax(1).cpu().numpy()
    accuracy=float((probs.argmax(1)==dataset.labels).mean())
    del loader,dataset; return accuracy,probs


# CUDA model preflight happens before YOLO and the 8.47-GiB cache build.
probe_model=build_model(False).to(DEVICE).eval()
with torch.no_grad(), torch.autocast("cuda",enabled=True):
    probe_output=probe_model(torch.zeros(1,3,CFG.model_frames,CFG.image_size,CFG.image_size,device=DEVICE))
if tuple(probe_output.shape)!=(1,N_CLASSES) or not torch.isfinite(probe_output).all():
    raise RuntimeError("CUDA MViTv2-S preflight failed")
print("CUDA MViTv2-S preflight OK",tuple(probe_output.shape))
del probe_output,probe_model; gc.collect(); torch.cuda.empty_cache()

train_clips=build_train_index()
expected_cache_bytes=len(train_clips)*CFG.cache_frames*3*CFG.image_size*CFG.image_size
if shutil.disk_usage(CFG.cache_dir).free < expected_cache_bytes+256*1024**2:
    raise RuntimeError(f"Insufficient /kaggle/temp space: need at least {(expected_cache_bytes+256*1024**2)/1024**3:.2f} GiB")
train_windows=detect_windows(train_clips)
thermal_meta=build_cache(train_clips,train_windows)
aligned_state,thermal_val_meta=train_aligned(thermal_meta)
packed=quantize_state_dict(aligned_state)
torch.save(dict(model_int8=packed,architecture="torchvision_mvit_v2_s_relative_position_32x128",
                classes=list(range(N_CLASSES)),train_updates=CFG.target_updates,
                maskfeat_updates=CFG.maskfeat_updates,pretraining_test_data_used=False,
                validation_users=list(CFG.validation_users),diagnostic_only=True),
           CFG.output_root/"aligned_thermal_int8_DIAGNOSTIC_ONLY.pt")
del aligned_state; gc.collect()
aligned_model=build_model(False)
aligned_model.load_state_dict(dequantize_state_dict(packed),strict=True)
aligned_model=aligned_model.to(DEVICE)
thermal_acc,thermal_val_probs=evaluate_aligned(aligned_model,thermal_val_meta)
del aligned_model,packed; gc.collect(); torch.cuda.empty_cache()
print("aligned Thermal INT8 accuracy",f"{thermal_acc:.4f}")


def normalize_probs(p,name):
    p=np.asarray(p,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=N_CLASSES: raise ValueError(f"{name} shape must be (N,40), got {p.shape}")
    if not np.isfinite(p).all() or (p<0).any(): raise ValueError(f"{name} contains invalid values")
    mass=p.sum(1,keepdims=True)
    if (mass<=0).any(): raise ValueError(f"{name} contains zero-mass rows")
    return p/mass


visual_meta=pd.read_parquet(CFG.visual_val_meta).reset_index(drop=True)
visual_val=normalize_probs(np.load(CFG.visual_val_probs),"visual_val")
if len(visual_meta)!=len(visual_val): raise ValueError("Visual validation metadata/probability length mismatch")
needed={"clip_id","user","action_id"}
if not needed.issubset(visual_meta.columns): raise ValueError(f"Visual metadata missing {needed-set(visual_meta.columns)}")
if set(visual_meta.user.astype(str).unique())!=set(CFG.validation_users): raise ValueError("Unexpected Visual Fold-0 users")

tm=thermal_val_meta.reset_index(drop=True).copy(); tm["trow"]=np.arange(len(tm))
vm=visual_meta.copy(); vm["vrow"]=np.arange(len(vm))
aligned=vm.merge(tm[["clip_id","user","action_id","trow"]],on="clip_id",suffixes=("_v","_t"),validate="one_to_one")
if len(aligned)!=541: raise ValueError(f"Expected 541 aligned rows, found {len(aligned)}")
if not np.array_equal(aligned.action_id_v.to_numpy(),aligned.action_id_t.to_numpy()): raise ValueError("Aligned labels conflict")
y=aligned.action_id_v.to_numpy(np.int64); groups=aligned.user_v.astype(str).to_numpy()
pv=visual_val[aligned.vrow.to_numpy()]; pt=normalize_probs(thermal_val_probs[aligned.trow.to_numpy()],"thermal_val")


def temperature_scale(p,t):
    q=np.clip(normalize_probs(p,"temperature_input"),EPS,1.)**(1./t)
    return q/q.sum(1,keepdims=True)


def geometric_fuse(pv_,pt_,params):
    tv,tt,alpha=params; qv=temperature_scale(pv_,tv); qt=temperature_scale(pt_,tt)
    z=(1-alpha)*np.log(np.clip(qv,EPS,1.))+alpha*np.log(np.clip(qt,EPS,1.)); z-=z.max(1,keepdims=True)
    return normalize_probs(np.exp(z),"fused")


def nll(p,target): return float(-np.log(np.clip(p[np.arange(len(target)),target],EPS,1.)).mean())


PRIMARY_GRID=[(tv,tt,a) for tv in CFG.visual_temperatures for tt in CFG.thermal_temperatures for a in CFG.thermal_weights]
ANCHORED_GRID=[(tv,tt,a) for tv in CFG.visual_temperatures for tt in CFG.thermal_temperatures for a in CFG.anchored_weights]


def fit_params(ids, grid):
    best_key=best=None
    for params in grid:
        p=geometric_fuse(pv[ids],pt[ids],params)
        key=(int((p.argmax(1)==y[ids]).sum()),-nll(p,y[ids]),-params[2])
        if best_key is None or key>best_key: best_key,best=key,params
    return best


test_csv=pd.read_csv(CFG.test_csv)
visual_test=normalize_probs(np.load(CFG.visual_test_probs_path),"visual_test")
thermal_test=normalize_probs(np.load(CFG.v11_probs_root/CFG.thermal_test_probs),"thermal_test")
thermal_single=normalize_probs(np.load(CFG.v11_probs_root/CFG.thermal_test_probs_single),"thermal_single")
thermal_dual=normalize_probs(np.load(CFG.v11_probs_root/CFG.thermal_test_probs_dual),"thermal_dual")
for name,p in (("visual",visual_test),("thermal",thermal_test),("single",thermal_single),("dual",thermal_dual)):
    if len(p)!=len(test_csv): raise ValueError(f"{name} test rows {len(p)} != test.csv rows {len(test_csv)}")
main_single_diff=float(np.max(np.abs(thermal_test-thermal_single)))
if main_single_diff>1e-5: raise ValueError(f"thermal_test_probs is not the selected single-view file; max diff={main_single_diff}")
print("Thermal main-vs-single max difference",main_single_diff,"; dual retained as diagnostic only")


def test_clip_id(path): return str(path).strip("/").split("/")[-1]
thermal_available=np.array([bool(frame_files(CFG.test_root/test_clip_id(path)/"Thermal")) for path in test_csv.path])
print("Thermal-missing test clips",int((~thermal_available).sum()))
if int((~thermal_available).sum())!=10: print("WARNING: expected 10 missing Thermal clips from the source notebook")

def outer_user_cv(grid, method):
    oof=np.zeros_like(pv); test_folds=[]; rows=[]
    for held in sorted(np.unique(groups)):
        tr=np.flatnonzero(groups!=held); va=np.flatnonzero(groups==held); params=fit_params(tr,grid)
        vp=geometric_fuse(pv[va],pt[va],params); tp=geometric_fuse(visual_test,thermal_test,params)
        tp[~thermal_available]=visual_test[~thermal_available]
        oof[va]=vp; test_folds.append(tp)
        rows.append(dict(method=method,held_user=held,n=len(va),correct=int((vp.argmax(1)==y[va]).sum()),
                         accuracy=float((vp.argmax(1)==y[va]).mean()),nll=nll(vp,y[va]),
                         visual_temperature=params[0],thermal_temperature=params[1],thermal_weight=params[2]))
    test_prob=normalize_probs(np.mean(test_folds,axis=0),method+"_test")
    test_prob[~thermal_available]=visual_test[~thermal_available]
    return oof,test_prob,rows


primary_oof,primary_test,primary_rows=outer_user_cv(PRIMARY_GRID,"v11_loou")
anchored_oof,anchored_test,anchored_rows=outer_user_cv(ANCHORED_GRID,"v11_anchored")
fold_report=pd.DataFrame(primary_rows+anchored_rows)

visual_correct=int((pv.argmax(1)==y).sum()); thermal_correct=int((pt.argmax(1)==y).sum())
primary_correct=int((primary_oof.argmax(1)==y).sum())
anchored_correct=int((anchored_oof.argmax(1)==y).sum())
oracle_correct=int(((pv.argmax(1)==y)|(pt.argmax(1)==y)).sum())
primary_rescued=int(((pv.argmax(1)!=y)&(primary_oof.argmax(1)==y)).sum())
primary_harmed=int(((pv.argmax(1)==y)&(primary_oof.argmax(1)!=y)).sum())
anchored_rescued=int(((pv.argmax(1)!=y)&(anchored_oof.argmax(1)==y)).sum())
anchored_harmed=int(((pv.argmax(1)==y)&(anchored_oof.argmax(1)!=y)).sum())
print("\nV11 THERMAL OUTER USER-CV")
print(f"aligned={len(y)} visual={visual_correct}/{len(y)}={visual_correct/len(y):.4f} "
      f"thermal={thermal_correct}/{len(y)}={thermal_correct/len(y):.4f} "
      f"oracle={oracle_correct}/{len(y)}={oracle_correct/len(y):.4f}")
print(f"V11 LOOU={primary_correct}/{len(y)}={primary_correct/len(y):.4f} "
      f"rescued={primary_rescued} harmed={primary_harmed}")
print(f"V11 anchored={anchored_correct}/{len(y)}={anchored_correct/len(y):.4f} "
      f"rescued={anchored_rescued} harmed={anchored_harmed}")
print(fold_report.to_string(index=False))

# The old 0.74626 artifacts are a reportable baseline, never an automatic replacement.
old_paths=[CFG.old_root/CFG.old_oof_name,CFG.old_root/CFG.old_test_name,CFG.old_root/CFG.old_meta_name]
if not all(p.exists() for p in old_paths):
    raise FileNotFoundError("The confirmed old 0.74626 baseline artifacts are required:\n"+"\n".join(map(str,old_paths)))
old_oof_raw=normalize_probs(np.load(old_paths[0]),"old_oof"); old_test=normalize_probs(np.load(old_paths[1]),"old_test")
old_meta=pd.read_parquet(old_paths[2]).reset_index(drop=True)
if len(old_meta)!=len(old_oof_raw) or "clip_id" not in old_meta: raise ValueError("Old fusion meta/OOF mismatch")
old_map={str(cid):i for i,cid in enumerate(old_meta.clip_id)}
if len(old_map)!=len(old_meta) or any(str(cid) not in old_map for cid in aligned.clip_id):
    raise ValueError("Old and V11 aligned sample IDs do not match")
old_oof=old_oof_raw[[old_map[str(cid)] for cid in aligned.clip_id]]
if len(old_test)!=len(test_csv): raise ValueError("Old fixed blend test probability length mismatch")
old_correct=int((old_oof.argmax(1)==y).sum())
comparison=dict(old_correct=old_correct,old_accuracy=old_correct/len(y),old_nll=nll(old_oof,y),
                v11_loou_correct=primary_correct,v11_loou_accuracy=primary_correct/len(y),v11_loou_nll=nll(primary_oof,y),
                v11_anchored_correct=anchored_correct,v11_anchored_accuracy=anchored_correct/len(y),v11_anchored_nll=nll(anchored_oof,y),
                oracle_correct=oracle_correct,oracle_accuracy=oracle_correct/len(y),old_oracle_reference=0.7966728280961183)
print("\nCOMPARISON ONLY — NO AUTOMATIC FALLBACK")
print(json.dumps(comparison,indent=2))


def write_submission(prob,path):
    out=test_csv[["path"]].copy(); out["prediction"]=LABEL_VALUES[prob.argmax(1)].astype(int)
    if len(out)!=405 or out.prediction.isna().any() or not out.prediction.between(0,39).all():
        raise ValueError("Submission validation failed")
    out.to_csv(path,index=False); return out


primary_submission=write_submission(primary_test,CFG.work_root/"submission.csv")
anchored_submission=write_submission(anchored_test,CFG.work_root/"submission_v11_anchored.csv")
old_submission=write_submission(old_test,CFG.work_root/"submission_old_074626.csv")
np.save(CFG.output_root/"v11_loou_oof_probs.npy",primary_oof.astype(np.float32))
np.save(CFG.output_root/"v11_loou_test_probs.npy",primary_test.astype(np.float32))
np.save(CFG.output_root/"v11_anchored_oof_probs.npy",anchored_oof.astype(np.float32))
np.save(CFG.output_root/"v11_anchored_test_probs.npy",anchored_test.astype(np.float32))
np.save(CFG.output_root/"aligned_thermal_probs.npy",pt.astype(np.float32))
aligned_out=pd.DataFrame(dict(clip_id=aligned.clip_id.astype(str),user=groups,action_id=y))
aligned_out.to_parquet(CFG.output_root/"aligned_meta.parquet",index=False)
fold_report.to_csv(CFG.output_root/"outer_user_cv.csv",index=False)
summary=dict(primary_submission=str(CFG.work_root/"submission.csv"),
             anchored_submission=str(CFG.work_root/"submission_v11_anchored.csv"),
             old_baseline_submission=str(CFG.work_root/"submission_old_074626.csv"),aligned_rows=len(y),
             visual_accuracy=visual_correct/len(y),thermal_accuracy=thermal_correct/len(y),
             oracle_accuracy=oracle_correct/len(y),primary_accuracy=primary_correct/len(y),
             anchored_accuracy=anchored_correct/len(y),primary_rescued=primary_rescued,
             primary_harmed=primary_harmed,anchored_rescued=anchored_rescued,
             anchored_harmed=anchored_harmed,thermal_missing_test=int((~thermal_available).sum()),
             comparison=comparison,
             final_submission_sha256=sha256(CFG.work_root/"submission.csv"))
(CFG.output_root/"fusion_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print("\nFINAL")
print(json.dumps(summary,indent=2))
print("wrote /kaggle/working/submission.csv (V11 LOOU primary)")
print("wrote /kaggle/working/submission_v11_anchored.csv")
print("wrote /kaggle/working/submission_old_074626.csv")
print(primary_submission.head())
