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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm


# Change only this block when attaching a differently named Kaggle dataset.
TRAIN_ROOT = Path("/kaggle/input/datasets/zhuowamg/skeleton/Skeleton")
CLASS_MAPPING_CSV = Path("/kaggle/input/datasets/zhuowamg/class-mapping1/class_mapping.csv")
TEST_ROOT = Path("/kaggle/input/datasets/zhuowamg/cuhksmall/CUHK-X_dataset/Small-Model-Track/Testing/data/small_model_track_test/small_model_track_test")
TEST_CSV = Path("/kaggle/input/datasets/zhuowamg/cuhksmall/CUHK-X_dataset/Small-Model-Track/Testing/test_file/test.csv")
OUT_DIR = Path("/kaggle/working/skeleton_subject_adversarial")
SUBMISSION_PATH = Path("/kaggle/working/submission_subject_adversarial.csv")

# Single-variable phase-aware-head ablation on the verified 48-frame input.
CFG = dict(seed=2026, frames=48, batch_size=64, epochs=45, warmup_epochs=4,
           lr=2e-3, min_lr=1e-5, weight_decay=2e-4, folds=5, workers=2,
           amp=True, use_height_motion=False, use_temporal_attention=False,
           use_confidence_smoothing=False, confidence_smooth_alpha=.35,
           # Keep the verified baseline sampler unless explicitly running the
           # isolated peak/context ablation.
           use_peak_context_sampling=False, peak_frames=24,
           use_multiscale_tcn=True, run_single_scale_ablation=False,
           use_diverse_ensemble=False, use_soft_class_balanced_sampler=False,
           sampler_power=.5, use_ssl_pretraining=False, ssl_epochs=10,
           ssl_lr=1e-3, ssl_time_mask_ratio=.20, ssl_joint_mask_ratio=.15,
           # Best verified training backbone: one 9-channel encoder.  The
           # two-branch variant improved OOF but reduced the external score.
           use_two_branch_model=False,
           use_phase_pooling_head=False,
           # Training-only domain generalisation. The subject classifier is
           # discarded before final checkpoint export and inference.
           use_subject_adversarial=True, subject_adv_max_lambda=.08,
           # Inference-only, label-preserving augmentation.  It neither adds
           # parameters nor changes the two final checkpoint sizes.
           # Keep the first diagnostic run reproducibly on the established
           # plain-inference baseline.  Each fold still prints the mirror-TTA
           # score from the same checkpoint; flip this to True only if its
           # OOF result improves consistently.
           use_flip_tta=False,
           final_models=2, max_weight_mb=7.0)
CFG["channels"] = 11 if CFG["use_height_motion"] else 9
CFG["feature_version"] = (
    "height_motion_v1" if CFG["use_height_motion"] else
    "confidence_smooth_v1" if CFG["use_confidence_smoothing"] else
    "base9_peak_context_v1" if CFG["use_peak_context_sampling"] else "base9_v1"
)
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


def motion_energy(a):
    """Confidence-masked framewise joint motion for temporal selection."""
    xy = np.nan_to_num(a[..., :2], nan=0.0, posinf=0.0, neginf=0.0)
    score = np.nan_to_num(a[..., 2], nan=0.0)
    valid = score > .05
    delta = xy[1:] - xy[:-1]
    # Detector jumps through missing joints must not be mistaken for action
    # peaks. A displacement counts only if that joint is visible on both frames.
    pair_valid = valid[1:] & valid[:-1]
    energy = np.zeros(len(a), dtype=np.float32)
    energy[1:] = (np.linalg.norm(delta, axis=-1) * pair_valid).sum(axis=1)
    return energy


def temporal_sample(a, n):
    """Return n frames and their original-frame timestamps.

    For long clips, retain the 24 largest confidence-masked motion peaks plus
    24 uniformly distributed *different* frames for context.  Sorting indices
    restores chronological order before feature extraction.  For clips at most
    n frames long, the exact old linear interpolation path is retained.
    """
    total = len(a)
    source_time = np.linspace(0, max(total - 1, 0), n, dtype=np.float32)
    if not CFG["use_peak_context_sampling"] or total <= n:
        return resample(a, n), source_time

    peaks = min(int(CFG["peak_frames"]), n - 1)
    energy = motion_energy(a)
    # A genuinely static or entirely untracked clip has no meaningful peak;
    # keep the baseline sampler rather than giving early frames arbitrary bias.
    if energy.max() <= 1e-6:
        return resample(a, n), source_time

    peak_idx = np.argsort(-energy, kind="stable")[:peaks]
    remaining = np.setdiff1d(np.arange(total), peak_idx, assume_unique=False)
    context_count = n - len(peak_idx)
    context_pos = np.rint(np.linspace(0, len(remaining) - 1, context_count)).astype(int)
    context_idx = remaining[context_pos]
    selected = np.sort(np.concatenate([peak_idx, context_idx])).astype(int)
    return a[selected].astype(np.float32, copy=False), selected.astype(np.float32)


def interpolate_scalar(values, valid):
    """Use neighbouring valid frames for a derived scalar, never raw zeros."""
    good = np.flatnonzero(valid)
    if len(good) == 0:
        return np.zeros_like(values)
    return np.interp(np.arange(len(values)), good, values[good]).astype(np.float32)


def confidence_smooth_coordinates(xy, score, threshold=.05, alpha=.35):
    """Fill low-confidence coordinates and lightly smooth only their trajectory.

    This changes coordinates used as body anchors, but callers retain the
    original confidence mask for feature validity. Therefore interpolation does
    not turn a missing wrist/ankle into a trusted pose observation.
    """
    t, joints, _ = xy.shape
    result = np.zeros_like(xy)
    for j in range(joints):
        valid = score[:, j] > threshold
        weight = np.where(valid, np.clip(score[:, j], 0., 1.), 0.)
        good = np.flatnonzero(valid)
        if len(good) == 0:
            continue
        for d in range(2):
            filled = np.interp(np.arange(t), good, xy[good, j, d]).astype(np.float32)
            # Confidence-weighted 1-2-1 filter with edge replication. A small
            # blend avoids smoothing away genuine rapid hand motion.
            pv = np.pad(filled, (1, 1), mode="edge")
            pw = np.pad(weight, (1, 1), mode="edge")
            smooth = (pv[:-2] * pw[:-2] + 2 * pv[1:-1] * pw[1:-1] + pv[2:] * pw[2:])
            denom = np.maximum(pw[:-2] + 2 * pw[1:-1] + pw[2:], 1e-6)
            result[:, j, d] = (1 - alpha) * filled + alpha * (smooth / denom)
    return result


def make_features(keypoints, frames=48):
    """Baseline seven channels plus two torso-relative angle channels, (C,T,V).

    Translation is removed per frame using the midpoint of hips.  Scale uses
    torso length, so different camera distances/body sizes do not dominate.
    """
    a, source_time = temporal_sample(keypoints, frames)
    xy, score = a[..., :2], np.nan_to_num(a[..., 2:3], nan=0.0)
    if CFG["use_confidence_smoothing"]:
        xy = confidence_smooth_coordinates(
            xy, score[..., 0], alpha=CFG["confidence_smooth_alpha"])
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
    # Peak/context selection is deliberately non-uniform in source time. Scale
    # finite differences back to the old nominal resampling interval so a wide
    # contextual gap cannot masquerade as an extreme velocity spike.
    nominal_step = max((len(keypoints) - 1) / max(frames - 1, 1), 1.0)
    dt = np.maximum(np.diff(source_time), 1.0).astype(np.float32)
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) * (nominal_step / dt)[:, None, None]

    # One deliberately isolated experimental feature: each bone's orientation
    # relative to the torso. sin/cos avoids an artificial pi-boundary. It is
    # invariant to global camera rotation and has no value for the two roots.
    torso = shoulder - hip
    torso_angle = np.arctan2(torso[:, 1], torso[:, 0])[:, None]
    relative_angle = np.arctan2(bone[..., 1], bone[..., 0]) - torso_angle
    angle = np.stack([np.sin(relative_angle), np.cos(relative_angle)], axis=-1)
    angle[:, PARENT < 0] = 0.0

    features = [pos, score.clip(0, 1), bone, vel, angle]
    if CFG["use_height_motion"]:
        # Relative hip-to-ankle height retains the posture transition signal
        # removed by hip centring, while remaining invariant to camera motion.
        ankle = (xy[:, 15] + xy[:, 16]) * .5
        height_raw = np.linalg.norm(hip - ankle, axis=-1) / scale[:, 0, 0]
        height_valid = ((score[:, 11, 0] > .05) & (score[:, 12, 0] > .05) &
                        (score[:, 15, 0] > .05) & (score[:, 16, 0] > .05))
        height = interpolate_scalar(height_raw, height_valid)
        height_vel = np.zeros_like(height)
        height_vel[1:] = (height[1:] - height[:-1]) * (nominal_step / dt)
        posture = np.stack([height, height_vel], axis=-1)
        features.append(np.repeat(posture[:, None, :], 17, axis=1))
    feat = np.concatenate(features, axis=-1)
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
                                     action_name=action_dir.name,
                                     action_id=name_to_label[action_dir.name]))
    df = pd.DataFrame(rows)
    if df.empty: raise RuntimeError("No training skeleton clips found; check TRAIN_ROOT")
    return df


def cache_train(index):
    # Include both frame and channel count: feature-experiment caches cannot be reused.
    cache = OUT_DIR / f"train_{CFG['feature_version']}_c{CFG['channels']}_f{CFG['frames']}.npy"; OUT_DIR.mkdir(parents=True, exist_ok=True)
    shape = (len(index), CFG["channels"], CFG["frames"], 17)
    if cache.exists() and cache.stat().st_size == int(np.prod(shape)) * 4:
        return np.load(cache, mmap_mode="r")
    x = np.empty(shape, np.float32)
    for i, d in enumerate(tqdm(index.pred_dir, desc="Extract train features")):
        x[i] = load_feature(Path(d))
    np.save(cache, x)
    return np.load(cache, mmap_mode="r")


class SkeletonDataset(Dataset):
    def __init__(self, x, y=None, augment=False, subject_y=None):
        self.x, self.y, self.augment, self.subject_y = x, y, augment, subject_y
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
        item = (torch.from_numpy(x), -1 if self.y is None else int(self.y[i]))
        return item if self.subject_y is None else (*item, int(self.subject_y[i]))


def make_train_sampler(labels):
    """Softly balance action frequency without using aggressive 1/count weights."""
    counts = np.bincount(labels)
    class_weight = np.zeros_like(counts, dtype=np.float64)
    present = counts > 0
    # 1/sqrt(count): rare actions are seen more often but cannot dominate
    # every batch, which is important because leaderboard uses accuracy.
    class_weight[present] = counts[present].astype(np.float64) ** (-CFG["sampler_power"])
    sample_weight = class_weight[labels]
    sample_weight /= sample_weight.mean()
    return WeightedRandomSampler(torch.as_tensor(sample_weight, dtype=torch.double),
                                 num_samples=len(labels), replacement=True)


def mask_skeleton_tokens(x):
    """Mask whole time spans and whole joints, shared across feature channels."""
    n, _, t, v = x.shape
    mask = torch.zeros((n, 1, t, v), dtype=torch.bool, device=x.device)
    time_width = max(1, int(round(t * CFG["ssl_time_mask_ratio"])))
    joint_count = max(1, int(round(v * CFG["ssl_joint_mask_ratio"])))
    for i in range(n):
        start = torch.randint(0, t - time_width + 1, (1,), device=x.device).item()
        joints = torch.randperm(v, device=x.device)[:joint_count]
        mask[i, :, start:start + time_width, :] = True
        mask[i, :, :, joints] = True
    return x.masked_fill(mask.expand_as(x), 0.0), mask


def pretrain_masked_reconstruction(model, train_x, tag):
    """Train-fold-only SSL: reconstruct hidden joints/time spans before labels."""
    loader = DataLoader(SkeletonDataset(train_x, None, False), CFG["batch_size"],
                        shuffle=True, num_workers=CFG["workers"], pin_memory=True,
                        drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), CFG["ssl_lr"], weight_decay=CFG["weight_decay"])
    scaler = GradScaler(enabled=CFG["amp"] and DEVICE.type == "cuda")
    for epoch in range(CFG["ssl_epochs"]):
        model.train(); total_loss = 0.0; seen = 0
        for x, _ in loader:
            x = x.to(DEVICE, non_blocking=True)
            masked_x, mask = mask_skeleton_tokens(x)
            target = x.index_select(1, model.feature_indices)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                reconstruction = model.reconstruct_masked(masked_x)
                point_loss = F.smooth_l1_loss(reconstruction, target, reduction="none")
                loss = (point_loss * mask).sum() / (mask.sum() * target.size(1)).clamp_min(1)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item() * x.size(0); seen += x.size(0)
        print(f"SSL {tag} epoch {epoch + 1:02d}/{CFG['ssl_epochs']}: recon_loss={total_loss / max(seen, 1):.4f}")


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
    def __init__(self, cin, cout, adjacency, stride=1, drop=.12, use_multiscale=True):
        super().__init__()
        self.gcn = GraphConv(cin, cout, adjacency)
        self.tcn = (MultiScaleTemporalConv(cout, stride, drop)
                    if use_multiscale else
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


class STGCNEncoder(nn.Module):
    """One lightweight graph-temporal encoder for a coherent feature family."""
    def __init__(self, in_channels, widths=(40, 40, 64, 64, 96, 96)):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels * 17)
        a = spatial_adjacency()
        self.net = nn.Sequential(
            Block(in_channels, widths[0], a, drop=.05),
            Block(widths[0], widths[1], a),
            Block(widths[1], widths[2], a, 2),
            Block(widths[2], widths[3], a),
            Block(widths[3], widths[4], a, 2),
            Block(widths[4], widths[5], a),
        )

    def forward(self, x):
        n, c, t, v = x.shape
        x = self.bn(x.permute(0, 3, 1, 2).reshape(n, v * c, t))
        x = x.reshape(n, v, c, t).permute(0, 2, 3, 1)
        return self.net(x)


class GradientReversal(torch.autograd.Function):
    """Identity forward pass; reverse only the shared-feature gradient."""
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.strength * grad_output, None


def gradient_reverse(x, strength):
    return GradientReversal.apply(x, strength)


class TemporalPhaseHead(nn.Module):
    """Classify global and ordered action-phase summaries with a tiny head."""
    def __init__(self, channels, classes, phases=3, drop=.25):
        super().__init__()
        self.phases = phases
        self.drop = nn.Dropout(drop)
        # global (C) + early/middle/late (3C): 25,640 parameters at C=160.
        self.fc = nn.Linear(channels * (phases + 1), classes)

    def forward(self, x):
        # Spatial pooling first retains the temporal order established by the
        # graph/TCN encoder. Adaptive pooling makes this valid for 48 or 64
        # input frames without adding temporal-position parameters.
        temporal = x.mean(dim=-1)                 # (N, C, T)
        global_feature = temporal.mean(dim=-1)     # (N, C)
        phase_feature = F.adaptive_avg_pool1d(temporal, self.phases).flatten(1)
        return self.fc(self.drop(torch.cat([global_feature, phase_feature], dim=1)))


class SmallSTGCN(nn.Module):
    def __init__(self, classes, use_multiscale_tcn=None, feature_indices=None, n_subjects=None):
        super().__init__()
        use_multiscale_tcn = CFG["use_multiscale_tcn"] if use_multiscale_tcn is None else use_multiscale_tcn
        if feature_indices is None:
            feature_indices = tuple(range(CFG["channels"]))
        self.register_buffer("feature_indices", torch.tensor(feature_indices, dtype=torch.long))
        self.use_two_branch = CFG["use_two_branch_model"]
        if self.use_two_branch:
            if tuple(feature_indices) != tuple(range(CFG["channels"])):
                raise ValueError("two-branch model requires all nine baseline channels")
            # Static posture/visibility and relative motion geometry have very
            # different distributions; do not force the first GCN to mix them.
            self.pose_encoder = STGCNEncoder(3)       # position x/y + score
            self.motion_encoder = STGCNEncoder(6)     # bone, velocity, angle
            self.fuse = nn.Sequential(nn.Conv2d(192, 160, 1, bias=False),
                                      nn.BatchNorm2d(160), nn.ReLU(inplace=True),
                                      nn.Dropout(.10))
        else:
            self.bn = nn.BatchNorm1d(len(feature_indices) * 17)
            a = spatial_adjacency()
            self.net = nn.Sequential(
                Block(len(feature_indices), 64, a, drop=.05, use_multiscale=use_multiscale_tcn),
                Block(64, 64, a, use_multiscale=use_multiscale_tcn),
                Block(64, 96, a, 2, use_multiscale=use_multiscale_tcn),
                Block(96, 96, a, use_multiscale=use_multiscale_tcn),
                Block(96, 160, a, 2, use_multiscale=use_multiscale_tcn),
                Block(160, 160, a, use_multiscale=use_multiscale_tcn),
            )
        # Attention was tested without a stable CV gain. Keep it switchable for
        # an ablation, but disable it in the topology-only experiment.
        self.temporal_attention = (GlobalTemporalAttention(160, CFG["frames"])
                                   if CFG["use_temporal_attention"] else nn.Identity())
        # Do not instantiate experimental modules in the verified baseline:
        # besides checkpoint bytes, their random initialisation would change
        # the seeded classifier initialisation on an otherwise identical run.
        self.ssl_decoder = (nn.Conv2d(160, len(feature_indices), kernel_size=1)
                            if CFG["use_ssl_pretraining"] else None)
        self.head = (TemporalPhaseHead(160, classes) if CFG["use_phase_pooling_head"] else
                     nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(.25), nn.Linear(160, classes)))
        # This auxiliary head is created only for the training model. It is
        # deliberately absent from final exported models and checkpoints.
        self.subject_head = (nn.Sequential(nn.Linear(160, 96), nn.ReLU(inplace=True),
                                           nn.Dropout(.10), nn.Linear(96, n_subjects))
                             if CFG["use_subject_adversarial"] and n_subjects else None)

    def forward_features(self, x, already_selected=False):
        if not already_selected:
            x = x.index_select(1, self.feature_indices)
        if self.use_two_branch:
            x = self.fuse(torch.cat([self.pose_encoder(x[:, :3]),
                                     self.motion_encoder(x[:, 3:])], dim=1))
        else:
            n, c, t, v = x.shape
            x = self.bn(x.permute(0, 3, 1, 2).reshape(n, v*c, t)).reshape(n, v, c, t).permute(0, 2, 3, 1)
            x = self.net(x)
        return self.temporal_attention(x)

    def reconstruct_masked(self, x):
        """Reconstruct selected input channels at their original T,V shape."""
        if self.ssl_decoder is None:
            raise RuntimeError("SSL decoder is unavailable when SSL pretraining is disabled")
        target_t, target_v = x.shape[2:]
        z = self.forward_features(x)
        return F.interpolate(self.ssl_decoder(z), size=(target_t, target_v),
                             mode="bilinear", align_corners=False)

    def forward(self, x):
        return self.head(self.forward_features(x))

    def forward_with_subject(self, x, adversarial_strength):
        """Return action and training-only subject logits from one backbone pass."""
        if self.subject_head is None:
            raise RuntimeError("subject head was not created for this model")
        features = self.forward_features(x)
        pooled = features.mean(dim=(2, 3))
        return self.head(features), self.subject_head(gradient_reverse(pooled, adversarial_strength))


@torch.no_grad()
def mirror_skeleton_batch(x):
    """Apply the same physical left/right mirror used during training."""
    x = x.index_select(-1, torch.as_tensor(LEFT_RIGHT, device=x.device)).clone()
    # position-x, bone-x, velocity-x and torso-relative angle sine.
    x[:, (0, 3, 5), :, :] *= -1
    x[:, 7, :, :] *= -1
    return x


@torch.no_grad()
def predict(model, loader, mirrored=False):
    model.eval(); out = []
    for batch in loader:
        x = batch[0]
        x = x.to(DEVICE, non_blocking=True)
        if mirrored:
            x = mirror_skeleton_batch(x)
        out.append(torch.softmax(model(x), 1).cpu().numpy())
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


def subject_adv_strength(step, total_steps):
    """Warm domain pressure in smoothly after the action head has formed."""
    progress = step / max(total_steps - 1, 1)
    ramp = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
    return CFG["subject_adv_max_lambda"] * ramp


def train_one_fold(train_x, train_y, val_x, val_y, n_classes, fold, use_multiscale_tcn,
                   feature_indices=None, train_users=None):
    train_sampler = make_train_sampler(train_y) if CFG["use_soft_class_balanced_sampler"] else None
    subject_y, n_subjects = None, None
    if CFG["use_subject_adversarial"]:
        if train_users is None:
            raise ValueError("subject-adversarial training requires training user ids")
        subject_values = np.unique(train_users)
        subject_y = np.searchsorted(subject_values, train_users).astype(np.int64)
        n_subjects = len(subject_values)
    train_loader = DataLoader(
        SkeletonDataset(train_x, train_y, True, subject_y), CFG["batch_size"],
        shuffle=train_sampler is None, sampler=train_sampler,
        num_workers=CFG["workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(SkeletonDataset(val_x, val_y), CFG["batch_size"] * 2, num_workers=CFG["workers"], pin_memory=True)
    model = SmallSTGCN(n_classes, use_multiscale_tcn, feature_indices, n_subjects).to(DEVICE)
    if CFG["use_ssl_pretraining"]:
        pretrain_masked_reconstruction(model, train_x, f"fold{fold}")
    opt = torch.optim.AdamW(model.parameters(), CFG["lr"], weight_decay=CFG["weight_decay"])
    scaler = GradScaler(enabled=CFG["amp"] and DEVICE.type == "cuda")
    best, best_sd, best_epoch = -1., None, 0
    total_steps = len(train_loader) * CFG["epochs"]
    warmup_steps = len(train_loader) * CFG["warmup_epochs"]
    step = 0
    for epoch in range(CFG["epochs"]):
        model.train()
        domain_loss_sum, domain_count = 0., 0
        for batch in train_loader:
            x, y = batch[:2]
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            lr = scheduled_lr(step, total_steps, warmup_steps)
            for g in opt.param_groups: g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                if CFG["use_subject_adversarial"]:
                    subject_target = batch[2].to(DEVICE, non_blocking=True)
                    action_logits, subject_logits = model.forward_with_subject(
                        x, subject_adv_strength(step, total_steps))
                    action_loss = F.cross_entropy(action_logits, y, label_smoothing=.05)
                    domain_loss = F.cross_entropy(subject_logits, subject_target)
                    loss = action_loss + domain_loss
                    domain_loss_sum += domain_loss.detach().item() * len(y)
                    domain_count += len(y)
                else:
                    loss = F.cross_entropy(model(x), y, label_smoothing=.05)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 3.0); scaler.step(opt); scaler.update()
            step += 1
        p = predict(model, val_loader); acc = (p.argmax(1) == val_y).mean()
        if acc > best:
            best, best_epoch = acc, epoch + 1
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        extra = (f", subject_loss={domain_loss_sum / max(domain_count, 1):.4f}"
                 if CFG["use_subject_adversarial"] else "")
        print(f"fold {fold} epoch {epoch+1:02d}: val_acc={acc:.4f}, best={best:.4f}{extra}")
    # OOF predictions must come from the best validation epoch, never the last.
    model.load_state_dict(best_sd)
    val_prob = predict(model, val_loader)
    # This is deliberately evaluated from exactly the same best checkpoint as
    # the plain prediction.  It isolates TTA from training stochasticity.
    val_flip_prob = predict(model, val_loader, mirrored=True)
    return best, best_epoch, val_prob, val_flip_prob


def analysis_dir():
    path = OUT_DIR / "analysis"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_error_analysis(oof, class_names, tag):
    """Write OOF diagnostics without using any test-set label information."""
    out = analysis_dir()
    classes = np.arange(len(class_names))
    names = [class_names[i] for i in classes]
    cm = confusion_matrix(oof.true_class, oof.pred_class, labels=classes)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    pd.DataFrame(cm, index=names, columns=names).to_csv(out / f"confusion_matrix_count_{tag}.csv")
    pd.DataFrame(cm_norm, index=names, columns=names).to_csv(out / f"confusion_matrix_recall_{tag}.csv")

    precision, recall, f1, _ = precision_recall_fscore_support(
        oof.true_class, oof.pred_class, labels=classes, zero_division=0)
    class_stats = pd.DataFrame({
        "action_name": names, "samples": cm.sum(axis=1), "correct": np.diag(cm),
        "recall": recall, "precision": precision, "f1": f1,
    }).sort_values("recall")
    class_stats.to_csv(out / f"class_metrics_{tag}.csv", index=False)

    user_stats = (oof.groupby(["user", "fold"], as_index=False)
                  .agg(samples=("correct", "size"), accuracy=("correct", "mean"))
                  .sort_values("accuracy"))
    user_stats.to_csv(out / f"user_metrics_{tag}.csv", index=False)

    pairs = []
    for true_idx in classes:
        for pred_idx in classes:
            if true_idx != pred_idx and cm[true_idx, pred_idx]:
                pairs.append({"true_action": class_names[true_idx], "pred_action": class_names[pred_idx],
                              "count": int(cm[true_idx, pred_idx]),
                              "true_samples": int(cm[true_idx].sum()),
                              "pair_rate": float(cm[true_idx, pred_idx] / max(cm[true_idx].sum(), 1))})
    pair_df = pd.DataFrame(pairs, columns=["true_action", "pred_action", "count", "true_samples", "pair_rate"])
    pair_df.sort_values(["count", "pair_rate"], ascending=False).to_csv(
        out / f"top_confusions_{tag}.csv", index=False)
    oof.loc[~oof.correct].sort_values("confidence", ascending=False).to_csv(
        out / f"oof_errors_{tag}.csv", index=False)
    oof.to_csv(out / f"oof_{tag}.csv", index=False)

    # A PNG is convenient in a Kaggle notebook; detailed action names remain
    # available in the CSVs because 40 labels are too dense to read on a plot.
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(16, 14))
        image = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(f"OOF row-normalised confusion matrix: {tag}")
        ax.set_xlabel("Predicted action"); ax.set_ylabel("True action")
        ax.set_xticks(classes); ax.set_yticks(classes)
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_yticklabels(names, fontsize=6)
        fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
        fig.tight_layout(); fig.savefig(out / f"confusion_matrix_{tag}.png", dpi=180)
        plt.close(fig)
    except ImportError:
        pass


def write_ablation_comparison(single, multi):
    """Identify classes/samples improved or regressed by multi-scale TCN."""
    out = analysis_dir()
    cols = ["row_id", "true_action", "user", "fold", "pred_action", "correct", "confidence"]
    base = single[cols].rename(columns={"pred_action": "single_pred", "correct": "single_correct", "confidence": "single_conf"})
    dual = multi[cols].rename(columns={"pred_action": "multi_pred", "correct": "multi_correct", "confidence": "multi_conf"})
    merged = base.merge(dual, on=["row_id", "true_action", "user", "fold"], validate="one_to_one")
    merged["change"] = np.select(
        [~merged.single_correct & merged.multi_correct,
         merged.single_correct & ~merged.multi_correct],
        ["improved", "regressed"], default="unchanged")
    merged.to_csv(out / "oof_single_vs_multiscale.csv", index=False)
    summary = (merged.groupby("true_action", as_index=False)
               .agg(samples=("change", "size"), single_accuracy=("single_correct", "mean"),
                    multiscale_accuracy=("multi_correct", "mean"),
                    improved=("change", lambda s: int((s == "improved").sum())),
                    regressed=("change", lambda s: int((s == "regressed").sum()))))
    summary["accuracy_delta"] = summary.multiscale_accuracy - summary.single_accuracy
    summary.sort_values("accuracy_delta", ascending=False).to_csv(out / "class_delta_single_vs_multiscale.csv", index=False)
    merged.loc[merged.change == "improved"].to_csv(out / "oof_improved_by_multiscale.csv", index=False)
    merged.loc[merged.change == "regressed"].to_csv(out / "oof_regressed_by_multiscale.csv", index=False)


def run_oof_cv(x, y, groups, index, class_values, class_names, tag, use_multiscale_tcn, feature_indices=None):
    splitter = StratifiedGroupKFold(CFG["folds"], shuffle=True, random_state=CFG["seed"])
    oof_prob = np.zeros((len(y), len(class_values)), np.float32)
    fold_id = np.full(len(y), -1, dtype=int)
    scores, best_epochs = [], []
    for fold, (tr, va) in enumerate(splitter.split(x, y, groups)):
        seed_everything(CFG["seed"] + fold)
        score, epoch, prob, flip_prob = train_one_fold(
            x[tr], y[tr], x[va], y[va], len(class_values), fold,
            use_multiscale_tcn, feature_indices, groups[tr])
        chosen_prob = (prob + flip_prob) * .5 if CFG["use_flip_tta"] else prob
        oof_prob[va] = chosen_prob; fold_id[va] = fold
        plain_acc = (prob.argmax(1) == y[va]).mean()
        flip_acc = (chosen_prob.argmax(1) == y[va]).mean()
        print(f"fold {fold}: plain={plain_acc:.4f}, flip_tta={flip_acc:.4f} "
              f"(checkpoint selected by plain val={score:.4f})")
        scores.append(flip_acc if CFG["use_flip_tta"] else plain_acc)
        best_epochs.append(epoch)
    pred = oof_prob.argmax(1)
    oof = index[["user", "action_name", "action_id"]].copy()
    oof.insert(0, "row_id", np.arange(len(oof)))
    oof["fold"] = fold_id; oof["true_class"] = y; oof["pred_class"] = pred
    oof["true_action"] = [class_names[i] for i in y]
    oof["pred_action"] = [class_names[i] for i in pred]
    oof["confidence"] = oof_prob.max(axis=1)
    oof["correct"] = y == pred
    write_error_analysis(oof, class_names, tag)
    mode = "flip TTA" if CFG["use_flip_tta"] else "plain inference"
    print(f"{tag} 5-fold CV ({mode}): {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return oof, scores, best_epochs, oof_prob


def train_full(x, y, n_classes, epochs, model_id, feature_indices=None, users=None):
    """Train one final model on every available subject, with a unique seed."""
    train_sampler = make_train_sampler(y) if CFG["use_soft_class_balanced_sampler"] else None
    subject_y, n_subjects = None, None
    if CFG["use_subject_adversarial"]:
        if users is None:
            raise ValueError("subject-adversarial training requires training user ids")
        subject_values = np.unique(users)
        subject_y = np.searchsorted(subject_values, users).astype(np.int64)
        n_subjects = len(subject_values)
    loader = DataLoader(SkeletonDataset(x, y, True, subject_y), CFG["batch_size"],
                        shuffle=train_sampler is None, sampler=train_sampler,
                        num_workers=CFG["workers"], pin_memory=True, drop_last=True)
    model = SmallSTGCN(n_classes, feature_indices=feature_indices, n_subjects=n_subjects).to(DEVICE)
    if CFG["use_ssl_pretraining"]:
        pretrain_masked_reconstruction(model, x, f"final{model_id}")
    opt = torch.optim.AdamW(model.parameters(), CFG["lr"], weight_decay=CFG["weight_decay"])
    scaler = GradScaler(enabled=CFG["amp"] and DEVICE.type == "cuda")
    total_steps, warmup_steps, step = len(loader) * epochs, len(loader) * CFG["warmup_epochs"], 0
    for epoch in range(epochs):
        model.train(); loss_sum = 0.; domain_loss_sum = 0.
        for batch in loader:
            xb, yb = batch[:2]
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            lr = scheduled_lr(step, total_steps, warmup_steps)
            for group in opt.param_groups: group["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                if CFG["use_subject_adversarial"]:
                    subject_target = batch[2].to(DEVICE, non_blocking=True)
                    action_logits, subject_logits = model.forward_with_subject(
                        xb, subject_adv_strength(step, total_steps))
                    action_loss = F.cross_entropy(action_logits, yb, label_smoothing=.05)
                    domain_loss = F.cross_entropy(subject_logits, subject_target)
                    loss = action_loss + domain_loss
                    domain_loss_sum += domain_loss.detach().item() * len(yb)
                else:
                    loss = F.cross_entropy(model(xb), yb, label_smoothing=.05)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 3.0); scaler.step(opt); scaler.update()
            step += 1; loss_sum += loss.item() * len(yb)
        extra = (f", subject_loss={domain_loss_sum / len(x):.4f}"
                 if CFG["use_subject_adversarial"] else "")
        print(f"final model {model_id} epoch {epoch+1:02d}/{epochs}: train_loss={loss_sum / len(x):.4f}{extra}")
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    # The domain head is a training regulariser, not part of the edge model.
    sd = {k: v for k, v in sd.items() if not k.startswith("subject_head.")}
    size = save_compact(sd, n_classes, OUT_DIR / f"final_model_{model_id}.pt", epochs=epochs,
                        feature_indices=list(feature_indices or range(CFG["channels"])))
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
    mapping = pd.read_csv(CLASS_MAPPING_CSV, encoding="utf-8-sig")
    id_to_name = dict(zip(mapping.action_id, mapping.action_name.astype(str)))
    class_names = [id_to_name[v] for v in label_values]

    # Verified single-model feature set. Final models differ only by seed; this
    # is not a multi-stream or feature-specialist ensemble.
    full_features = tuple(range(CFG["channels"]))
    run_tag = ("subject_adversarial" if CFG["use_subject_adversarial"] else
               "phase_pooling" if CFG["use_phase_pooling_head"] else "full_features")
    multi_oof, scores, best_epochs, full_oof_prob = run_oof_cv(
        x, y, groups, index, label_values, class_names, run_tag, True, full_features)
    if CFG["run_single_scale_ablation"]:
        single_oof, _, _, _ = run_oof_cv(
            x, y, groups, index, label_values, class_names, "single_scale", False)
        write_ablation_comparison(single_oof, multi_oof)
        print(f"Wrote single-scale vs multi-scale OOF comparison to {analysis_dir()}")
    final_epochs = max(20, int(round(np.mean(best_epochs))))
    print(f"5-fold CV accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}; final epochs={final_epochs}")

    test = pd.read_csv(TEST_CSV); tx = test_features(test)
    test_loader = DataLoader(SkeletonDataset(tx), CFG["batch_size"] * 2, num_workers=CFG["workers"], pin_memory=True)
    prob = np.zeros((len(tx), len(label_values)), np.float32)
    for model_id in range(CFG["final_models"]):
        seed_everything(CFG["seed"] + 100 + model_id)
        sd = train_full(x, y, len(label_values), final_epochs, model_id, full_features, groups)
        model = SmallSTGCN(len(label_values), feature_indices=full_features).to(DEVICE)
        model.load_state_dict(sd)
        model_prob = predict(model, test_loader)
        if CFG["use_flip_tta"]:
            model_prob = (model_prob + predict(model, test_loader, mirrored=True)) * .5
        prob += model_prob / CFG["final_models"]
    pred = label_values[prob.argmax(1)]
    # Keep test.csv's identifier column and use the original notebook's label header.
    submission = test[["path"]].copy(); submission["prediction"] = pred
    submission.to_csv(SUBMISSION_PATH, index=False)
    total_size = sum(p.stat().st_size for p in OUT_DIR.glob("final_model_*.pt")) / 1024**2
    print(f"Final two-weight size: {total_size:.2f} MB / {CFG['max_weight_mb']:.2f} MB")
    print(f"Wrote {SUBMISSION_PATH} with columns {submission.columns.tolist()}")


if __name__ == "__main__": main()
