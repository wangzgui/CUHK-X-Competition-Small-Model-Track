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
OUT_DIR = Path("/kaggle/working/v20_imu_skeleton_conflict_arbitration/skeleton")
SUBMISSION_PATH = Path("/kaggle/working/submission_skeleton_reference.csv")

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
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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

    Translation is removed per frame using the midpoint of hips.  Scale uses
    torso length, so different camera distances/body sizes do not dominate.
    """
    a = resample(keypoints, frames)
    xy, score = a[..., :2], np.nan_to_num(a[..., 2:3], nan=0.0)
    valid = (score > 0.05).astype(np.float32)
    hip = (xy[:, 11] + xy[:, 12]) * .5
    shoulder = (xy[:, 5] + xy[:, 6]) * .5
    scale = np.linalg.norm(shoulder - hip, axis=-1, keepdims=True)
    # Fall back to shoulder width if a detector misses either hip.
    fallback = np.linalg.norm(xy[:, 5] - xy[:, 6], axis=-1, keepdims=True)
    scale = np.where(scale > 8.0, scale, fallback)
    scale = np.clip(scale, 20.0, None)[..., None]
    pos = (xy - hip[:, None, :]) / scale
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
        for user_dir in sorted(action_dir.iterdir()):
            if not user_dir.is_dir(): continue
            for trial_dir in sorted(user_dir.iterdir()):
                pred = trial_dir / "predictions"
                if pred.exists() and all_jsons(pred):
                    trial = str(trial_dir.name)
                    action_name = str(action_dir.name)
                    user = str(user_dir.name)
                    rows.append(dict(
                        pred_dir=str(pred), action_name=action_name, user=user,
                        trial=trial, clip_id=f"{action_name}/{user}/{trial}",
                        action_id=name_to_label[action_name],
                    ))
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
    best, best_sd, best_epoch, best_prob = -1., None, 0, None
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
            best_prob = p.astype(np.float32, copy=True)
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"fold {fold} epoch {epoch+1:02d}: val_acc={acc:.4f}, best={best:.4f}")
    # Selection checkpoints are unnecessary after this run and are not saved:
    # only the two final weights count toward the deployment budget.
    if best_prob is None:
        raise RuntimeError(f"Fold {fold} never produced validation probabilities")
    return best, best_epoch, best_prob


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
    skeleton_oof = np.zeros((len(index), len(label_values)), np.float32)
    skeleton_fold = np.full(len(index), -1, np.int64)
    for fold, (tr, va) in enumerate(splitter.split(x, y, groups)):
        seed_everything(CFG["seed"] + fold)
        score, epoch, fold_prob = train_one_fold(
            x[tr], y[tr], x[va], y[va], len(label_values), fold
        )
        skeleton_oof[va] = fold_prob
        skeleton_fold[va] = fold
        scores.append(score); best_epochs.append(epoch)
    if (skeleton_fold < 0).any() or not np.isfinite(skeleton_oof).all():
        raise RuntimeError("Skeleton OOF export is incomplete")
    skeleton_meta = index[["clip_id", "action_name", "user", "trial", "action_id"]].copy()
    skeleton_meta.insert(0, "row_id", np.arange(len(skeleton_meta)))
    skeleton_meta["fold"] = skeleton_fold
    skeleton_meta["prediction"] = label_values[skeleton_oof.argmax(1)]
    skeleton_meta["correct"] = skeleton_oof.argmax(1) == y
    skeleton_meta["confidence"] = skeleton_oof.max(1)
    skeleton_meta.to_csv(OUT_DIR / "skeleton_oof_meta.csv", index=False)
    np.save(OUT_DIR / "skeleton_oof_probs.npy", skeleton_oof)
    np.save(OUT_DIR / "skeleton_label_values.npy", label_values)
    final_epochs = max(20, int(round(np.mean(best_epochs))))
    print(f"5-fold CV accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}; final epochs={final_epochs}")

    test = pd.read_csv(TEST_CSV); tx = test_features(test)
    test_loader = DataLoader(SkeletonDataset(tx), CFG["batch_size"] * 2, num_workers=CFG["workers"], pin_memory=True)
    prob = np.zeros((len(tx), len(label_values)), np.float32)
    for model_id in range(CFG["final_models"]):
        seed_everything(CFG["seed"] + 100 + model_id)
        sd = train_full(x, y, len(label_values), final_epochs, model_id)
        model = SmallSTGCN(len(label_values)).to(DEVICE); model.load_state_dict(sd)
        prob += predict(model, test_loader) / CFG["final_models"]
    if not np.isfinite(prob).all() or not np.allclose(prob.sum(1), 1.0, atol=1e-4):
        raise RuntimeError("Invalid Skeleton test probabilities")
    np.save(OUT_DIR / "skeleton_test_probs.npy", prob.astype(np.float32))
    np.save(OUT_DIR / "skeleton_test_labels.npy", label_values)
    test[["path"]].to_csv(OUT_DIR / "skeleton_test_meta.csv", index=False)
    pred = label_values[prob.argmax(1)]
    # Keep test.csv's identifier column and use the original notebook's label header.
    submission = test[["path"]].copy(); submission["prediction"] = pred
    submission.to_csv(SUBMISSION_PATH, index=False)
    total_size = sum(p.stat().st_size for p in OUT_DIR.glob("final_model_*.pt")) / 1024**2
    print(f"Final two-weight size: {total_size:.2f} MB / {CFG['max_weight_mb']:.2f} MB")
    print(f"Wrote {SUBMISSION_PATH} with columns {submission.columns.tolist()}")
# =============================================================================
# V20: IMU + Skeleton as third-party witnesses in Visual/Thermal disagreements
# =============================================================================

from scipy.stats import beta as beta_distribution

V20_ROOT = Path("/kaggle/working/v20_imu_skeleton_conflict_arbitration")
V20_INPUT = Path("/kaggle/input/datasets/zhuowamg/v20yy11")
THERMAL_TEST_ORIGINAL = Path(
    "/kaggle/input/datasets/zhuowamg/v20lll2/thermal_v16_test_probs_original.npy"
)
V12_ROOT = Path("/kaggle/input/data-ooo/v12_visual_thermal_v16_fusion")
VISUAL_OOF_PROBS = Path(
    "/kaggle/input/datasets/zhuowamg/ir-depth-color/outputs/val_probs_fold0.npy"
)
VISUAL_OOF_META = Path(
    "/kaggle/input/datasets/zhuowamg/ir-depth-color/outputs/val_pred_fold0.parquet"
)
VISUAL_TEST_PROBS = Path(
    "/kaggle/input/datasets/zhuowamg/newensembledata/visual_test_probs.npy"
)
IMU_ROOT = Path(
    "/kaggle/input/datasets/zhuowamg/imudataout/imu_transformer_no_interp"
)

BASE_OOF = V20_INPUT / "v17_original_reference_oof_probs.npy"
BASE_TEST = V20_INPUT / "v17_original_reference_test_probs.npy"
THERMAL_OOF_LOGITS = V20_INPUT / "thermal_v16_aligned_logits_original_flipped.npz"
ALIGNED_META = V12_ROOT / "aligned_meta.parquet"

EPS = 1e-12
EXPECTED_OOF = 541
EXPECTED_TEST = 405
EXPECTED_CLASSES = 40
HELD_USERS = ("user16", "user18", "user9")


def v20_require(paths):
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("V20 required files are missing:\n" + "\n".join(missing))


def v20_normalize(prob, name, rows=None):
    prob = np.asarray(prob, dtype=np.float64)
    if prob.ndim != 2 or prob.shape[1] != EXPECTED_CLASSES:
        raise ValueError(f"{name}: expected (*,40), received {prob.shape}")
    if rows is not None and len(prob) != rows:
        raise ValueError(f"{name}: expected {rows} rows, received {len(prob)}")
    if not np.isfinite(prob).all() or (prob < 0).any():
        raise ValueError(f"{name}: invalid probability values")
    prob = np.clip(prob, EPS, None)
    prob /= prob.sum(1, keepdims=True)
    return prob


def v20_softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(1, keepdims=True)


def v20_reorder(prob, source_labels, canonical_labels, name):
    source_labels = np.asarray(source_labels).reshape(-1)
    canonical_labels = np.asarray(canonical_labels).reshape(-1)
    if set(source_labels.tolist()) != set(canonical_labels.tolist()):
        raise ValueError(f"{name}: class set differs from canonical labels")
    order = [int(np.flatnonzero(source_labels == label)[0]) for label in canonical_labels]
    return v20_normalize(np.asarray(prob)[:, order], name)


def v20_top_rank(prob, candidate):
    order = np.argsort(-prob, axis=1)
    inverse = np.empty_like(order)
    inverse[np.arange(len(prob))[:, None], order] = np.arange(prob.shape[1])[None, :]
    return inverse[np.arange(len(prob)), candidate] + 1


def v20_margin(prob):
    top2 = np.partition(prob, -2, axis=1)[:, -2:]
    return top2.max(1) - top2.min(1)


def v20_conflict_components(base, visual, thermal, imu, skeleton, available=None):
    bp = base.argmax(1)
    vp = visual.argmax(1)
    tp = thermal.argmax(1)
    ip = imu.argmax(1)
    sp = skeleton.argmax(1)
    if available is None:
        available = np.ones(len(base), dtype=bool)
    conflict = available & (vp != tp) & ((bp == vp) | (bp == tp))
    alt = np.where(bp == vp, tp, np.where(bp == tp, vp, bp)).astype(np.int64)
    alt_rank_base = v20_top_rank(base, alt)
    alt_rank_imu = v20_top_rank(imu, alt)
    alt_rank_skeleton = v20_top_rank(skeleton, alt)
    return dict(
        base_pred=bp, visual_pred=vp, thermal_pred=tp, imu_pred=ip,
        skeleton_pred=sp, conflict=conflict, alt=alt,
        alt_rank_base=alt_rank_base, alt_rank_imu=alt_rank_imu,
        alt_rank_skeleton=alt_rank_skeleton, margin=v20_margin(base),
    )


POLICIES = {
    # Both independent weak modalities must name the exact alternative class.
    "strict": dict(kind="both_top1", margin_quantile=0.75, base_rank=3,
                   min_decisive=2, min_rescues=2, min_posterior_mean=0.60,
                   min_net=1),
    # At least one names it Top-1 and the other must contain it in Top-2.
    "moderate": dict(kind="one_top1_other_top2", margin_quantile=0.50, base_rank=3,
                     min_decisive=4, min_rescues=3, min_posterior_mean=0.65,
                     min_net=2),
}


def v20_evidence_mask(parts, spec, margin_threshold):
    alt = parts["alt"]
    if spec["kind"] == "both_top1":
        helper = (parts["imu_pred"] == alt) & (parts["skeleton_pred"] == alt)
    elif spec["kind"] == "one_top1_other_top2":
        helper = (
            ((parts["imu_pred"] == alt) & (parts["alt_rank_skeleton"] <= 2))
            | ((parts["skeleton_pred"] == alt) & (parts["alt_rank_imu"] <= 2))
        )
    else:
        raise ValueError(spec["kind"])
    return (
        parts["conflict"] & helper
        & (parts["alt_rank_base"] <= spec["base_rank"])
        & (parts["margin"] <= margin_threshold)
    )


def v20_fit_policy(parts, y, groups, train_mask, spec):
    conflict_margins = parts["margin"][train_mask & parts["conflict"]]
    if len(conflict_margins) == 0:
        return dict(accepted=False, margin_threshold=-1.0, reason="no_conflicts")
    threshold = float(np.quantile(conflict_margins, spec["margin_quantile"]))
    eligible = train_mask & v20_evidence_mask(parts, spec, threshold)
    base_ok = parts["base_pred"] == y
    alt_ok = parts["alt"] == y
    decisive = eligible & (base_ok ^ alt_ok)
    rescues = int((decisive & alt_ok).sum())
    harms = int((decisive & base_ok).sum())
    support = rescues + harms
    posterior_mean = float((rescues + 1) / (support + 2))
    posterior_lcb90 = float(beta_distribution.ppf(0.10, rescues + 1, harms + 1))
    user_nets = {}
    for user in sorted(np.unique(groups[train_mask])):
        m = decisive & (groups == user)
        user_nets[str(user)] = int((m & alt_ok).sum() - (m & base_ok).sum())
    accepted = (
        support >= spec["min_decisive"]
        and rescues >= spec["min_rescues"]
        and rescues - harms >= spec["min_net"]
        and posterior_mean >= spec["min_posterior_mean"]
        and all(net >= 0 for net in user_nets.values())
    )
    return dict(
        accepted=bool(accepted), margin_threshold=threshold,
        eligible_train=int(eligible.sum()), decisive_support=support,
        rescues=rescues, harms=harms, net_gain=rescues - harms,
        posterior_mean=posterior_mean, posterior_lcb90=posterior_lcb90,
        user_nets=user_nets,
    )


def v20_swap_probabilities(prob, switch, alt):
    result = prob.copy()
    rows = np.flatnonzero(switch)
    old = result.argmax(1)
    if len(rows):
        hold = result[rows, old[rows]].copy()
        result[rows, old[rows]] = result[rows, alt[rows]]
        result[rows, alt[rows]] = hold
    return v20_normalize(result, "swapped probabilities")


def v20_score(name, prob, reference, y, groups):
    pred = prob.argmax(1)
    ref = reference.argmax(1)
    changed = pred != ref
    rescued = changed & (pred == y) & (ref != y)
    harmed = changed & (pred != y) & (ref == y)
    row = dict(
        method=name, correct=int((pred == y).sum()), accuracy=float((pred == y).mean()),
        changed=int(changed.sum()), rescued=int(rescued.sum()), harmed=int(harmed.sum()),
        net_gain=int(rescued.sum() - harmed.sum()),
    )
    for user in HELD_USERS:
        m = groups == user
        row[f"gain_{user}"] = int((pred[m] == y[m]).sum() - (ref[m] == y[m]).sum())
    return row


def run_v20():
    V20_ROOT.mkdir(parents=True, exist_ok=True)
    required = [
        BASE_OOF, BASE_TEST, THERMAL_OOF_LOGITS, THERMAL_TEST_ORIGINAL,
        ALIGNED_META, VISUAL_OOF_PROBS, VISUAL_OOF_META, VISUAL_TEST_PROBS,
        IMU_ROOT / "imu_oof_probs.npy", IMU_ROOT / "imu_oof_meta.csv",
        IMU_ROOT / "imu_label_values.npy", IMU_ROOT / "imu_test_probs.npy",
        OUT_DIR / "skeleton_oof_probs.npy", OUT_DIR / "skeleton_oof_meta.csv",
        OUT_DIR / "skeleton_label_values.npy", OUT_DIR / "skeleton_test_probs.npy",
        OUT_DIR / "skeleton_test_meta.csv", TEST_CSV, CLASS_MAPPING_CSV,
    ]
    v20_require(required)

    class_map = pd.read_csv(CLASS_MAPPING_CSV, encoding="utf-8-sig")
    labels = np.sort(class_map.action_id.astype(int).unique())
    if labels.shape != (EXPECTED_CLASSES,):
        raise ValueError(f"Expected 40 canonical labels, received {labels}")
    label_to_col = {int(label): i for i, label in enumerate(labels)}

    meta = pd.read_parquet(ALIGNED_META).reset_index(drop=True)
    if len(meta) != EXPECTED_OOF or not {"clip_id", "user", "action_id"}.issubset(meta.columns):
        raise ValueError(f"Unexpected aligned metadata: {meta.shape}, {meta.columns.tolist()}")
    if set(meta.user.astype(str)) != set(HELD_USERS):
        raise ValueError(f"Expected held users {HELD_USERS}, got {sorted(meta.user.unique())}")
    y = meta.action_id.astype(int).map(label_to_col).to_numpy(np.int64)
    groups = meta.user.astype(str).to_numpy()

    base_oof = v20_normalize(np.load(BASE_OOF), "V14-original OOF", EXPECTED_OOF)
    base_test = v20_normalize(np.load(BASE_TEST), "V14-original test", EXPECTED_TEST)
    if int((base_oof.argmax(1) == y).sum()) != 411:
        raise RuntimeError("V14-original OOF failed the expected 411/541 reproduction")

    visual_meta = pd.read_parquet(VISUAL_OOF_META).reset_index(drop=True)
    visual_index = {str(v): i for i, v in enumerate(visual_meta.clip_id)}
    if any(str(v) not in visual_index for v in meta.clip_id):
        raise ValueError("Visual OOF is missing aligned clip IDs")
    visual_all = v20_normalize(np.load(VISUAL_OOF_PROBS), "Visual OOF all")
    visual_oof = visual_all[[visual_index[str(v)] for v in meta.clip_id]]
    visual_test = v20_normalize(np.load(VISUAL_TEST_PROBS), "Visual test", EXPECTED_TEST)

    thermal_logits = np.load(THERMAL_OOF_LOGITS)
    if "original" not in thermal_logits.files:
        raise ValueError("Thermal OOF archive has no 'original' logits")
    thermal_oof = v20_normalize(v20_softmax(thermal_logits["original"]), "Thermal original OOF", EXPECTED_OOF)
    thermal_test = v20_normalize(np.load(THERMAL_TEST_ORIGINAL), "Thermal original test", EXPECTED_TEST)

    imu_meta = pd.read_csv(IMU_ROOT / "imu_oof_meta.csv")
    imu_meta["clip_id"] = (
        imu_meta.action_name.astype(str) + "/" + imu_meta.user.astype(str)
        + "/" + imu_meta.trial.astype(str)
    )
    if imu_meta.clip_id.duplicated().any():
        raise ValueError("Duplicate IMU OOF clip IDs")
    imu_index = {str(v): i for i, v in enumerate(imu_meta.clip_id)}
    if any(str(v) not in imu_index for v in meta.clip_id):
        raise ValueError("IMU OOF is missing aligned clip IDs")
    imu_labels = np.load(IMU_ROOT / "imu_label_values.npy")
    imu_all = v20_reorder(np.load(IMU_ROOT / "imu_oof_probs.npy"), imu_labels, labels, "IMU OOF")
    imu_oof = imu_all[[imu_index[str(v)] for v in meta.clip_id]]
    imu_test = v20_reorder(np.load(IMU_ROOT / "imu_test_probs.npy"), imu_labels, labels, "IMU test")

    skeleton_meta = pd.read_csv(OUT_DIR / "skeleton_oof_meta.csv")
    if skeleton_meta.clip_id.duplicated().any():
        raise ValueError("Duplicate Skeleton OOF clip IDs")
    skeleton_index = {str(v): i for i, v in enumerate(skeleton_meta.clip_id)}
    if any(str(v) not in skeleton_index for v in meta.clip_id):
        missing = [str(v) for v in meta.clip_id if str(v) not in skeleton_index][:10]
        raise ValueError(f"Skeleton OOF is missing aligned clip IDs: {missing}")
    skeleton_labels = np.load(OUT_DIR / "skeleton_label_values.npy")
    skeleton_all = v20_reorder(np.load(OUT_DIR / "skeleton_oof_probs.npy"), skeleton_labels, labels, "Skeleton OOF")
    skeleton_oof = skeleton_all[[skeleton_index[str(v)] for v in meta.clip_id]]

    test_csv = pd.read_csv(TEST_CSV).reset_index(drop=True)
    skeleton_test_meta = pd.read_csv(OUT_DIR / "skeleton_test_meta.csv")
    if test_csv.path.astype(str).tolist() != skeleton_test_meta.path.astype(str).tolist():
        raise ValueError("Skeleton test order differs from test.csv")
    skeleton_test = v20_reorder(
        np.load(OUT_DIR / "skeleton_test_probs.npy"), skeleton_labels, labels, "Skeleton test"
    )
    if any(len(p) != EXPECTED_TEST for p in (base_test, visual_test, thermal_test, imu_test, skeleton_test)):
        raise ValueError("Test probability row count mismatch")

    # In the frozen V12/V13/V14 stack, unavailable Thermal rows fall back exactly
    # to Visual. This recovers the ten-row mask without needing an extra CSV.
    thermal_available_test = np.max(np.abs(base_test - visual_test), axis=1) > 1e-10
    missing_thermal = int((~thermal_available_test).sum())
    if missing_thermal != 10:
        raise RuntimeError(f"Expected 10 Thermal-unavailable test rows, inferred {missing_thermal}")

    oof_parts = v20_conflict_components(base_oof, visual_oof, thermal_oof, imu_oof, skeleton_oof)
    test_parts = v20_conflict_components(
        base_test, visual_test, thermal_test, imu_test, skeleton_test, thermal_available_test
    )

    base_ok = oof_parts["base_pred"] == y
    oracle_any = base_ok | (imu_oof.argmax(1) == y) | (skeleton_oof.argmax(1) == y)
    both_alt = (
        oof_parts["conflict"]
        & (oof_parts["imu_pred"] == oof_parts["alt"])
        & (oof_parts["skeleton_pred"] == oof_parts["alt"])
    )
    oracle_summary = dict(
        aligned_rows=EXPECTED_OOF,
        base_correct=int(base_ok.sum()),
        visual_correct=int((visual_oof.argmax(1) == y).sum()),
        thermal_correct=int((thermal_oof.argmax(1) == y).sum()),
        imu_correct=int((imu_oof.argmax(1) == y).sum()),
        skeleton_correct=int((skeleton_oof.argmax(1) == y).sum()),
        union_oracle_correct=int(oracle_any.sum()),
        union_incremental_rescues=int((oracle_any & ~base_ok).sum()),
        visual_thermal_conflicts=int(oof_parts["conflict"].sum()),
        both_helpers_name_alternative=int(both_alt.sum()),
        both_helpers_rescues=int((both_alt & ~base_ok & (oof_parts["alt"] == y)).sum()),
        both_helpers_harms=int((both_alt & base_ok & (oof_parts["alt"] != y)).sum()),
    )

    policy_rows = []
    score_rows = [v20_score("V14_original_079106_reference", base_oof, base_oof, y, groups)]
    candidate_oof = {}
    candidate_test = {}
    changed_tables = []

    for name, spec in POLICIES.items():
        oof_switch = np.zeros(EXPECTED_OOF, dtype=bool)
        test_votes = np.zeros((EXPECTED_TEST, EXPECTED_CLASSES), dtype=np.int16)
        for held_user in HELD_USERS:
            train_mask = groups != held_user
            held_mask = groups == held_user
            fitted = v20_fit_policy(oof_parts, y, groups, train_mask, spec)
            row = {"candidate": name, "held_user": held_user, **fitted}
            row["user_nets"] = json.dumps(row.get("user_nets", {}), sort_keys=True)
            if fitted["accepted"]:
                held_eligible = held_mask & v20_evidence_mask(
                    oof_parts, spec, fitted["margin_threshold"]
                )
                oof_switch[held_eligible] = True
                test_eligible = v20_evidence_mask(
                    test_parts, spec, fitted["margin_threshold"]
                )
                rows = np.flatnonzero(test_eligible)
                test_votes[rows, test_parts["alt"][rows]] += 1
                row["held_switches"] = int(held_eligible.sum())
                row["test_proposals"] = int(test_eligible.sum())
            else:
                row["held_switches"] = 0
                row["test_proposals"] = 0
            policy_rows.append(row)

        oof_prob = v20_swap_probabilities(base_oof, oof_switch, oof_parts["alt"])
        vote_count = test_votes.max(1)
        vote_label = test_votes.argmax(1)
        test_switch = vote_count >= 2
        test_prob = v20_swap_probabilities(base_test, test_switch, vote_label)
        candidate_oof[name] = oof_prob
        candidate_test[name] = test_prob
        score_rows.append(v20_score(f"V20_{name}", oof_prob, base_oof, y, groups))

        for row_id in np.flatnonzero(test_switch):
            changed_tables.append(dict(
                candidate=name, row=int(row_id), path=str(test_csv.path.iloc[row_id]),
                old_prediction=int(labels[base_test.argmax(1)[row_id]]),
                new_prediction=int(labels[test_prob.argmax(1)[row_id]]),
                visual_prediction=int(labels[test_parts["visual_pred"][row_id]]),
                thermal_prediction=int(labels[test_parts["thermal_pred"][row_id]]),
                imu_prediction=int(labels[test_parts["imu_pred"][row_id]]),
                skeleton_prediction=int(labels[test_parts["skeleton_pred"][row_id]]),
                outer_votes=int(vote_count[row_id]),
                base_margin=float(test_parts["margin"][row_id]),
            ))

    scores = pd.DataFrame(score_rows)
    policies = pd.DataFrame(policy_rows)
    changes = pd.DataFrame(changed_tables)
    scores.to_csv(V20_ROOT / "v20_oof_scores.csv", index=False)
    policies.to_csv(V20_ROOT / "v20_outer_user_policies.csv", index=False)
    changes.to_csv(V20_ROOT / "v20_test_changed_rows.csv", index=False)
    meta[["clip_id", "user", "action_id"]].to_parquet(
        V20_ROOT / "v20_aligned_meta.parquet", index=False
    )

    def save_submission(prob, filename):
        submission = test_csv[["path"]].copy()
        submission["prediction"] = labels[prob.argmax(1)].astype(int)
        submission.to_csv(Path("/kaggle/working") / filename, index=False)

    save_submission(base_test, "submission_v14_original_079106_reference.csv")
    for name in POLICIES:
        np.save(V20_ROOT / f"v20_{name}_oof_probs.npy", candidate_oof[name].astype(np.float32))
        np.save(V20_ROOT / f"v20_{name}_test_probs.npy", candidate_test[name].astype(np.float32))
        save_submission(candidate_test[name], f"v20_submission_{name}.csv")

    strict_row = scores.loc[scores.method == "V20_strict"].iloc[0]
    moderate_row = scores.loc[scores.method == "V20_moderate"].iloc[0]
    strict_safe = (
        strict_row.net_gain > 0
        and min(strict_row[f"gain_{u}"] for u in HELD_USERS) >= 0
        and strict_row.changed > 0
    )
    moderate_safe = (
        moderate_row.net_gain > strict_row.net_gain
        and min(moderate_row[f"gain_{u}"] for u in HELD_USERS) >= 0
        and moderate_row.changed > 0
    )
    recommendation = (
        "moderate" if moderate_safe else "strict" if strict_safe
        else "NO_SUBMISSION_KEEP_V14_ORIGINAL"
    )
    summary = {
        "method": "LOOU IMU+Skeleton third-party arbitration restricted to Visual/Thermal conflicts",
        "no_test_label_fitting": True,
        "base": "V14 Stable25 propagated with V16 Thermal original (public 0.79106)",
        "oracle": oracle_summary,
        "scores": scores.to_dict("records"),
        "test_changes": {
            name: int((candidate_test[name].argmax(1) != base_test.argmax(1)).sum())
            for name in POLICIES
        },
        "recommendation": recommendation,
        "safety_rule": "Do not submit a candidate with non-positive cross-fitted net gain or a negative held-user gain.",
        "outputs": {
            "reference": "/kaggle/working/submission_v14_original_079106_reference.csv",
            "strict": "/kaggle/working/v20_submission_strict.csv",
            "moderate": "/kaggle/working/v20_submission_moderate.csv",
        },
    }
    (V20_ROOT / "v20_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nV20 MODALITY / ORACLE SUMMARY")
    print(json.dumps(oracle_summary, indent=2))
    print("\nV20 CROSS-FITTED SCORES")
    print(scores.to_string(index=False))
    print("\nV20 OUTER-USER POLICIES")
    print(policies.to_string(index=False))
    print("\nV20 TEST CHANGES")
    print(changes.to_string(index=False) if len(changes) else "No test labels changed.")
    print("\nV20 RECOMMENDATION:", recommendation)
    print("Saved standalone submissions (no ZIP):")
    print("/kaggle/working/v20_submission_strict.csv")
    print("/kaggle/working/v20_submission_moderate.csv")
    print("/kaggle/working/submission_v14_original_079106_reference.csv")


if __name__ == "__main__":
    main()
    run_v20()
