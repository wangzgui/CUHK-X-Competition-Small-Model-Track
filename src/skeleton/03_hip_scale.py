"""Cross-subject skeleton action-recognition baseline.

Designed for the CUHK-X skeleton folder layout used in the supplied notebook.
Run this file in a Kaggle notebook after changing PATHS if needed. It uses a
five-fold subject-disjoint selection run, then trains two compact final models
on all training subjects and writes /kaggle/working/submission.csv.
"""
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# Change only this block when attaching a differently named Kaggle dataset.
TRAIN_ROOT = Path("/kaggle/input/datasets/zhuowamg/skeleton/Skeleton")
CLASS_MAPPING_CSV = Path("/kaggle/input/datasets/zhuowamg/class-mapping1/class_mapping.csv")
TEST_ROOT = Path("/kaggle/input/datasets/zhuowamg/cuhksmall/CUHK-X_dataset/Small-Model-Track/Testing/data/small_model_track_test/small_model_track_test")
TEST_CSV = Path("/kaggle/input/datasets/zhuowamg/cuhksmall/CUHK-X_dataset/Small-Model-Track/Testing/test_file/test.csv")
OUT_DIR = Path("/kaggle/working/skeleton_hip_scale")
SUBMISSION_PATH = Path("/kaggle/working/submission_hip_scale.csv")
AUDIT_ONLY = False

CFG = dict(seed=2026, frames=48, batch_size=64, epochs=45, warmup_epochs=4,
           lr=2e-3, min_lr=1e-5, weight_decay=2e-4, folds=5, workers=2,
           amp=True, channels=9, use_temporal_attention=False,
           use_multiscale_tcn=True, final_models=2, max_weight_mb=7.0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# COCO-17: every non-root joint has a kinematic parent.  Hip midpoint is the
# coordinate root (not an observed joint), avoiding viewpoint translation.
PARENT = np.array([5, 0, 0, 1, 2, 11, 12, 5, 6, 7, 8, -1, -1, 11, 12, 13, 14])
LEFT_RIGHT = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])
EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), (5, 7), (7, 9),
         (6, 8), (8, 10), (5, 11), (6, 12), (11, 13), (13, 15),
         (12, 14), (14, 16), (11, 12)]
# Anatomical distance from the hip centre. Used to split graph messages by
# direction; it is not an additional model input feature.
BODY_DEPTH = np.array([2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2])


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def frame_number(path):
    m = re.search(r"(\d+)", path.name)
    return int(m.group(1)) if m else -1


def read_frame(path):
    """Return (17, 3) = x, y, detector confidence; invalid frames are zero."""
    try:
        with open(path, "r") as f:
            item = json.load(f)[0]["keypoints"]
        a = np.asarray(item, dtype=np.float32)
        if a.shape[0] < 17:
            raise ValueError("fewer than 17 keypoints")
        return a[:17, :3]
    except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError):
        return np.zeros((17, 3), np.float32)


def resample(a, n):
    """Linear temporal resampling, preserving all clips at a common length."""
    if len(a) == 0: return np.zeros((n,) + a.shape[1:], np.float32)
    if len(a) == 1: return np.repeat(a, n, axis=0)
    x = np.linspace(0, len(a) - 1, n)
    lo, hi = np.floor(x).astype(int), np.ceil(x).astype(int)
    return ((1 - (x - lo))[:, None, None] * a[lo] +
            (x - lo)[:, None, None] * a[hi]).astype(np.float32)


def make_features(keypoints, frames=48):
    """Baseline seven channels plus two torso-relative angle channels, (C,T,V).

    Translation is removed per frame using the midpoint of hips. A single
    robust hip-width scale is used for the whole clip, matching this dataset's
    normalised coordinate units and avoiding artificial scale velocity.
    """
    a = resample(keypoints, frames)
    xy, score = a[..., :2], np.nan_to_num(a[..., 2:3], nan=0.0)
    valid = (score > 0.05).astype(np.float32)
    hip = (xy[:, 11] + xy[:, 12]) * .5
    shoulder = (xy[:, 5] + xy[:, 6]) * .5

    # The supplied keypoints are in approximately 0..1 coordinates. The old
    # pixel thresholds (>8 and >=20) therefore reduced every clip by the same
    # constant and never removed subject/camera scale. Both hips are visible in
    # about 99% of audited frames, making their clip median a stable body unit.
    hip_valid = (score[:, 11, 0] > .05) & (score[:, 12, 0] > .05)
    hip_width = np.linalg.norm(xy[:, 11] - xy[:, 12], axis=-1)
    usable = hip_width[hip_valid & np.isfinite(hip_width) & (hip_width > 1e-4)]
    if len(usable):
        clip_scale = float(np.median(usable))
    else:
        # Extremely rare fallback: use a valid shoulder width, then a neutral
        # unit. This affects only clips with no reliable hip width at all.
        shoulder_valid = (score[:, 5, 0] > .05) & (score[:, 6, 0] > .05)
        shoulder_width = np.linalg.norm(xy[:, 5] - xy[:, 6], axis=-1)
        usable = shoulder_width[shoulder_valid & np.isfinite(shoulder_width) &
                                (shoulder_width > 1e-4)]
        clip_scale = float(np.median(usable)) if len(usable) else 1.0
    clip_scale = max(clip_scale, 1e-3)
    pos = (xy - hip[:, None, :]) / clip_scale
    pos *= valid
    bone = np.zeros_like(pos)
    for j, p in enumerate(PARENT):
        if p >= 0: bone[:, j] = pos[:, j] - pos[:, p]
    vel = np.zeros_like(pos); vel[1:] = pos[1:] - pos[:-1]

    # One deliberately isolated experimental feature: each bone's orientation
    # relative to the torso. sin/cos avoids an artificial pi-boundary. It is
    # invariant to global camera rotation and has no value for the two roots.
    torso = shoulder - hip
    torso_angle = np.arctan2(torso[:, 1], torso[:, 0])[:, None]
    relative_angle = np.arctan2(bone[..., 1], bone[..., 0]) - torso_angle
    angle = np.stack([np.sin(relative_angle), np.cos(relative_angle)], axis=-1)
    angle[:, PARENT < 0] = 0.0
    # pos(2), confidence(1), bone vector(2), velocity(2), angle sin/cos(2).
    feat = np.concatenate([pos, score.clip(0, 1), bone, vel, angle], axis=-1)
    return np.nan_to_num(feat, nan=0., posinf=0., neginf=0.).transpose(2, 0, 1).astype(np.float32)


def all_jsons(pred_dir):
    return sorted(pred_dir.glob("Color_*.json"), key=frame_number)


def load_feature(pred_dir):
    return make_features(np.stack([read_frame(p) for p in all_jsons(pred_dir)]), CFG["frames"]) if pred_dir.exists() and all_jsons(pred_dir) else np.zeros((CFG["channels"], CFG["frames"], 17), np.float32)


def build_train_index():
    classes = pd.read_csv(CLASS_MAPPING_CSV, encoding="utf-8-sig")
    name_to_label = dict(zip(classes.action_name.astype(str), classes.action_id))
    rows = []
    for action_dir in tqdm(sorted(TRAIN_ROOT.iterdir()), desc="Index train clips"):
        if not action_dir.is_dir() or action_dir.name not in name_to_label: continue
        for user_dir in action_dir.iterdir():
            if not user_dir.is_dir(): continue
            for trial_dir in user_dir.iterdir():
                pred = trial_dir / "predictions"
                if pred.exists() and all_jsons(pred):
                    rows.append(dict(pred_dir=str(pred), user=str(user_dir.name),
                                     action_name=str(action_dir.name),
                                     action_id=name_to_label[action_dir.name]))
    df = pd.DataFrame(rows)
    if df.empty: raise RuntimeError("No training skeleton clips found; check TRAIN_ROOT")
    return df


def cache_train(index):
    # Include both frame and channel count: feature-experiment caches cannot be reused.
    cache = OUT_DIR / f"train_c{CFG['channels']}_f{CFG['frames']}.npy"; OUT_DIR.mkdir(parents=True, exist_ok=True)
    shape = (len(index), CFG["channels"], CFG["frames"], 17)
    if cache.exists() and cache.stat().st_size == int(np.prod(shape)) * 4:
        return np.load(cache, mmap_mode="r")
    x = np.empty(shape, np.float32)
    for i, d in enumerate(tqdm(index.pred_dir, desc="Extract train features")):
        x[i] = load_feature(Path(d))
    np.save(cache, x)
    return np.load(cache, mmap_mode="r")


class SkeletonDataset(Dataset):
    def __init__(self, x, y=None, augment=False): self.x, self.y, self.augment = x, y, augment
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        x = self.x[i].copy()
        if self.augment:
            # Physically valid mirror: negate horizontal values AND exchange left/right joints.
            if random.random() < .5:
                x = x[:, :, LEFT_RIGHT]; x[[0, 3, 5]] *= -1
                x[7] *= -1  # relative-angle sine reverses under reflection
            if random.random() < .35:
                shift = random.randint(-4, 4); x = np.roll(x, shift, axis=1)
            # Change action speed without changing its spatial geometry. This
            # is interpolation, not a circular time shift, and is deliberately
            # mild so it does not turn a short gesture into a different class.
            if random.random() < .30:
                t = x.shape[1]; grid = np.arange(t, dtype=np.float32)
                factor = random.uniform(.88, 1.12)
                source = np.clip((grid - (t - 1) / 2) * factor + (t - 1) / 2, 0, t - 1)
                x = np.stack([[np.interp(source, grid, x[c, :, v]) for v in range(x.shape[2])]
                              for c in range(x.shape[0])]).transpose(0, 2, 1).astype(np.float32)
                # Interpolation of sin/cos slightly changes their norm.
                norm = np.maximum(np.hypot(x[7], x[8]), 1e-6)
                x[7] /= norm; x[8] /= norm
        return (torch.from_numpy(x), -1 if self.y is None else int(self.y[i]))


def spatial_adjacency():
    """Three ST-GCN subsets: self, centripetal, and centrifugal edges."""
    a = np.zeros((3, 17, 17), dtype=np.float32)
    for i in range(17):
        a[0, i, i] = 1.0
    for i, j in EDGES:
        # Matrix convention below is source -> destination (v -> w).
        if BODY_DEPTH[i] < BODY_DEPTH[j]:
            a[1, i, j] = 1.0; a[2, j, i] = 1.0
        elif BODY_DEPTH[j] < BODY_DEPTH[i]:
            a[1, j, i] = 1.0; a[2, i, j] = 1.0
        else:  # bilateral links retain both directions in the third subset.
            a[2, i, j] = 1.0; a[2, j, i] = 1.0
    for k in range(3):
        a[k] /= np.maximum(a[k].sum(axis=0, keepdims=True), 1.0)
    return a


class GraphConv(nn.Module):
    def __init__(self, cin, cout, adjacency):
        super().__init__()
        self.k = adjacency.shape[0]
        self.conv = nn.Conv2d(cin, cout * self.k, 1)
        self.register_buffer("A", torch.tensor(adjacency, dtype=torch.float32))
        self.edge_residual = nn.Parameter(torch.zeros_like(self.A))
    def forward(self, x):
        n, _, t, v = x.shape
        x = self.conv(x).view(n, self.k, -1, t, v)
        a = self.A + .1 * torch.tanh(self.edge_residual)
        return torch.einsum("nkctv,kvw->nctw", x, a)


class MultiScaleTemporalConv(nn.Module):
    """Fuse short (3-frame) and long (7-frame receptive field) temporal cues."""
    def __init__(self, channels, stride=1, drop=.12):
        super().__init__()
        self.pre = nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        # dilation=3 with kernel 3 has a 7-frame receptive field, matching the
        # old kernel-7 TCN while retaining a separate local-motion pathway.
        self.short = nn.Conv2d(channels, channels, (3, 1), stride=(stride, 1), padding=(1, 0))
        self.long = nn.Conv2d(channels, channels, (3, 1), stride=(stride, 1), padding=(3, 0), dilation=(3, 1))
        self.fuse = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.post = nn.Sequential(nn.BatchNorm2d(channels), nn.Dropout(drop))

    def forward(self, x):
        x = self.pre(x)
        return self.post(self.fuse(torch.cat([self.short(x), self.long(x)], dim=1)))


class Block(nn.Module):
    def __init__(self, cin, cout, adjacency, stride=1, drop=.12):
        super().__init__()
        self.gcn = GraphConv(cin, cout, adjacency)
        self.tcn = (MultiScaleTemporalConv(cout, stride, drop)
                    if CFG["use_multiscale_tcn"] else
                    nn.Sequential(nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                  nn.Conv2d(cout, cout, (7, 1), stride=(stride, 1), padding=(3, 0)),
                                  nn.BatchNorm2d(cout), nn.Dropout(drop)))
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(nn.Conv2d(cin, cout, 1, stride=(stride, 1)), nn.BatchNorm2d(cout))
    def forward(self, x): return F.relu(self.tcn(self.gcn(x)) + self.skip(x), inplace=True)


class GlobalTemporalAttention(nn.Module):
    """Low-rank, two-head global attention over the temporal axis only."""
    def __init__(self, channels, max_frames, attention_dim=64, heads=2, drop=.10):
        super().__init__()
        self.input_norm = nn.LayerNorm(channels)
        self.in_proj = nn.Linear(channels, attention_dim, bias=False)
        self.position = nn.Parameter(torch.zeros(1, max_frames, attention_dim))
        self.norm = nn.LayerNorm(attention_dim)
        self.attn = nn.MultiheadAttention(attention_dim, heads, dropout=drop, batch_first=True)
        self.out_proj = nn.Linear(attention_dim, channels, bias=False)
        self.drop = nn.Dropout(drop)
        self.gain = nn.Parameter(torch.tensor(.5))
        nn.init.trunc_normal_(self.position, std=.02)

    def forward(self, x):
        # Pool only for attention tokens; the attended global context is then
        # broadcast back to every joint, preserving the original graph layout.
        t = x.size(2)
        tokens = x.mean(dim=-1).transpose(1, 2)
        tokens = self.in_proj(self.input_norm(tokens)) + self.position[:, :t]
        qkv = self.norm(tokens)
        attended, _ = self.attn(qkv, qkv, qkv, need_weights=False)
        context = self.out_proj(self.drop(attended)).transpose(1, 2).unsqueeze(-1)
        return x + self.gain * context


class SmallSTGCN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(CFG["channels"] * 17)
        # Learns separate self, body-inward and body-outward messages instead
        # of averaging every neighbour with a single shared 1x1 projection.
        a = spatial_adjacency()
        self.net = nn.Sequential(
            Block(CFG["channels"], 64, a, drop=.05), Block(64, 64, a),
            Block(64, 96, a, 2), Block(96, 96, a),
            Block(96, 160, a, 2), Block(160, 160, a),
        )
        # Attention was tested without a stable CV gain. Keep it switchable for
        # an ablation, but disable it in the topology-only experiment.
        self.temporal_attention = (GlobalTemporalAttention(160, CFG["frames"])
                                   if CFG["use_temporal_attention"] else nn.Identity())
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(.25), nn.Linear(160, classes))
    def forward(self, x):
        n, c, t, v = x.shape
        x = self.bn(x.permute(0, 3, 1, 2).reshape(n, v*c, t)).reshape(n, v, c, t).permute(0, 2, 3, 1)
        x = self.net(x)
        return self.head(self.temporal_attention(x))


@torch.no_grad()
def predict(model, loader):
    model.eval(); out = []
    for x, _ in loader:
        out.append(torch.softmax(model(x.to(DEVICE, non_blocking=True)), 1).cpu().numpy())
    return np.concatenate(out)


def compact_state_dict(state_dict):
    """FP16 checkpoint storage; load_state_dict safely casts it back to FP32."""
    return {k: (v.half() if v.is_floating_point() else v) for k, v in state_dict.items()}


def save_compact(state_dict, n_classes, path, **metadata):
    torch.save({"state_dict": compact_state_dict(state_dict), "classes": n_classes,
                "feature_channels": CFG["channels"], **metadata}, path)
    total = sum(p.stat().st_size for p in OUT_DIR.glob("final_model_*.pt")) / 1024**2
    if total > CFG["max_weight_mb"]:
        raise RuntimeError(f"Final weights are {total:.2f} MB, over {CFG['max_weight_mb']} MB")
    return path.stat().st_size / 1024**2


def scheduled_lr(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return CFG["lr"] * (step + 1) / max(warmup_steps, 1)
    p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return CFG["min_lr"] + .5 * (CFG["lr"] - CFG["min_lr"]) * (1 + math.cos(math.pi * p))


def train_one_fold(train_x, train_y, val_x, val_y, n_classes, fold):
    train_loader = DataLoader(SkeletonDataset(train_x, train_y, True), CFG["batch_size"], shuffle=True, num_workers=CFG["workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(SkeletonDataset(val_x, val_y), CFG["batch_size"] * 2, num_workers=CFG["workers"], pin_memory=True)
    model = SmallSTGCN(n_classes).to(DEVICE); opt = torch.optim.AdamW(model.parameters(), CFG["lr"], weight_decay=CFG["weight_decay"])
    scaler = GradScaler(enabled=CFG["amp"] and DEVICE.type == "cuda")
    best, best_sd, best_epoch = -1., None, 0
    total_steps = len(train_loader) * CFG["epochs"]
    warmup_steps = len(train_loader) * CFG["warmup_epochs"]
    step = 0
    for epoch in range(CFG["epochs"]):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            lr = scheduled_lr(step, total_steps, warmup_steps)
            for g in opt.param_groups: g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()): loss = F.cross_entropy(model(x), y, label_smoothing=.05)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 3.0); scaler.step(opt); scaler.update()
            step += 1
        p = predict(model, val_loader); acc = (p.argmax(1) == val_y).mean()
        if acc > best:
            best, best_epoch = acc, epoch + 1
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"fold {fold} epoch {epoch+1:02d}: val_acc={acc:.4f}, best={best:.4f}")
    # Recover validation probabilities from the best epoch. These predictions
    # are only for quality/error analysis and are not deployment checkpoints.
    model.load_state_dict(best_sd)
    return best, best_epoch, predict(model, val_loader)


def train_full(x, y, n_classes, epochs, model_id):
    """Train one final model on every available subject, with a unique seed."""
    loader = DataLoader(SkeletonDataset(x, y, True), CFG["batch_size"], shuffle=True,
                        num_workers=CFG["workers"], pin_memory=True, drop_last=True)
    model = SmallSTGCN(n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), CFG["lr"], weight_decay=CFG["weight_decay"])
    scaler = GradScaler(enabled=CFG["amp"] and DEVICE.type == "cuda")
    total_steps, warmup_steps, step = len(loader) * epochs, len(loader) * CFG["warmup_epochs"], 0
    for epoch in range(epochs):
        model.train(); loss_sum = 0.
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            lr = scheduled_lr(step, total_steps, warmup_steps)
            for group in opt.param_groups: group["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()): loss = F.cross_entropy(model(xb), yb, label_smoothing=.05)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 3.0); scaler.step(opt); scaler.update()
            step += 1; loss_sum += loss.item() * len(yb)
        print(f"final model {model_id} epoch {epoch+1:02d}/{epochs}: train_loss={loss_sum / len(x):.4f}")
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    size = save_compact(sd, n_classes, OUT_DIR / f"final_model_{model_id}.pt", epochs=epochs)
    print(f"Saved final_model_{model_id}.pt ({size:.2f} MB)")
    return sd


def test_features(test):
    features = []
    for p in tqdm(test["path"], desc="Extract test features"):
        # The supplied notebook uses the final component as the clip directory.
        clip = TEST_ROOT / Path(str(p).rstrip("/")).name
        features.append(load_feature(clip / "Skeleton" / "predictions"))
    return np.stack(features)


def main():
    seed_everything(CFG["seed"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Avoid accidentally counting stale weights from a previous run.
    for old in OUT_DIR.glob("final_model_*.pt"):
        old.unlink()
    index = build_train_index(); x = cache_train(index)
    # Never assume action_id values are 0..39 or contiguous.
    label_values = np.sort(index.action_id.unique()); to_idx = {v:i for i, v in enumerate(label_values)}
    y = index.action_id.map(to_idx).to_numpy(); groups = index.user.to_numpy()
    # Subject-disjoint five-fold CV is for selecting a stable full-data epoch
    # count. They are deliberately not used as the final ensemble because that
    # would make each final model discard half of the available subjects.
    splitter = StratifiedGroupKFold(CFG["folds"], shuffle=True, random_state=CFG["seed"])
    scores, best_epochs = [], []
    oof_prob = np.zeros((len(index), len(label_values)), dtype=np.float32)
    oof_fold = np.full(len(index), -1, dtype=np.int16)
    for fold, (tr, va) in enumerate(splitter.split(x, y, groups)):
        seed_everything(CFG["seed"] + fold)
        score, epoch, val_prob = train_one_fold(
            x[tr], y[tr], x[va], y[va], len(label_values), fold)
        oof_prob[va] = val_prob
        oof_fold[va] = fold
        scores.append(score); best_epochs.append(epoch)
    final_epochs = max(20, int(round(np.mean(best_epochs))))
    print(f"5-fold CV accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}; final epochs={final_epochs}")

    # Save every training clip's held-out prediction so skeleton quality can be
    # compared with correct/incorrect decisions without using test labels.
    analysis = OUT_DIR / "analysis"; analysis.mkdir(parents=True, exist_ok=True)
    pred_idx = oof_prob.argmax(1)
    oof = index.copy(); oof.insert(0, "row_id", np.arange(len(index)))
    oof["fold"] = oof_fold
    oof["true_class"] = y
    oof["pred_class"] = pred_idx
    oof["pred_action_id"] = label_values[pred_idx]
    oof["confidence"] = oof_prob.max(1)
    oof["correct"] = pred_idx == y
    oof_path = analysis / "oof_hip_scale.csv"
    oof.to_csv(oof_path, index=False)
    print(f"Wrote baseline OOF ({oof.correct.mean():.4f}) to {oof_path}")
    if AUDIT_ONLY:
        print("AUDIT_ONLY=True: skipping final-model training and test prediction.")
        return

    test = pd.read_csv(TEST_CSV); tx = test_features(test)
    test_loader = DataLoader(SkeletonDataset(tx), CFG["batch_size"] * 2, num_workers=CFG["workers"], pin_memory=True)
    prob = np.zeros((len(tx), len(label_values)), np.float32)
    for model_id in range(CFG["final_models"]):
        seed_everything(CFG["seed"] + 100 + model_id)
        sd = train_full(x, y, len(label_values), final_epochs, model_id)
        model = SmallSTGCN(len(label_values)).to(DEVICE); model.load_state_dict(sd)
        prob += predict(model, test_loader) / CFG["final_models"]
    pred = label_values[prob.argmax(1)]
    # Keep test.csv's identifier column and use the original notebook's label header.
    submission = test[["path"]].copy(); submission["prediction"] = pred
    submission.to_csv(SUBMISSION_PATH, index=False)
    total_size = sum(p.stat().st_size for p in OUT_DIR.glob("final_model_*.pt")) / 1024**2
    print(f"Final two-weight size: {total_size:.2f} MB / {CFG['max_weight_mb']:.2f} MB")
    print(f"Wrote {SUBMISSION_PATH} with columns {submission.columns.tolist()}")


if __name__ == "__main__": main()
