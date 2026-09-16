"""Native-sampling IMU Transformer: timestamp sort + padding, no resampling.

Audit-derived MAX_LEN=120 covers 99.13% of training sensor sequences and every
test sensor sequence. Each device is sorted independently, duplicate timestamps
are averaged, and short sequences are right-padded with an attention mask.
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


TRAIN_ROOT = Path("/kaggle/input/datasets/zhuowamg/imudata/IMU")
CLASS_MAPPING_CSV = Path("/kaggle/input/datasets/zhuowamg/class-mapping1/class_mapping.csv")
TEST_ROOT = Path("/kaggle/input/datasets/zhuowamg/cuhksmall/CUHK-X_dataset/Small-Model-Track/Testing/data/small_model_track_test/small_model_track_test")
TEST_CSV = Path("/kaggle/input/competitions/cuhk-x-competition-small-model-track/test.csv")
OUT_DIR = Path("/kaggle/working/imu_transformer_no_interp")
SUBMISSION_PATH = Path("/kaggle/working/submission.csv")

CFG = dict(
    seed=2026, max_len=120, folds=5, epochs=50, warmup_epochs=4,
    batch_size=64, workers=2, lr=1.5e-3, min_lr=1e-5,
    weight_decay=5e-4, label_smoothing=.05, amp=True,
    model_dim=96, sensor_dim=48, heads=4, temporal_layers=2,
    fusion_layers=1, ff_dim=192,
    dropout=.12, final_models=2, max_total_weight_mb=2.0,
    architecture_version="dual_stream_native_padding_v7",
    auxiliary_loss_weight=.30, initial_motion_weight=.65,
    paired_leg_dropout=.09, single_sensor_dropout=.10,
    use_magnetometer=False,
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SENSORS = ["WTC", "WTLA", "WTRA", "WTLL", "WTRL"]
SENSOR_TO_INDEX = {name: i for i, name in enumerate(SENSORS)}
ACC_COLS = ["加速度X(g)", "加速度Y(g)", "加速度Z(g)"]
GYRO_COLS = ["角速度X(°/s)", "角速度Y(°/s)", "角速度Z(°/s)"]
ANGLE_COLS = ["角度X(°)", "角度Y(°)", "角度Z(°)"]
MAG_COLS = ["磁场X(uT)", "磁场Y(uT)", "磁场Z(uT)"]
QUAT_COLS = ["四元数0()", "四元数1()", "四元数2()", "四元数3()"]
UP_FILE = "up(LA+RA+C).csv"
DOWN_FILE = "down(LL+RL).csv"


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def numeric_feature_dim():
    # acc(3), gyro(3), angle sin/cos(6), quaternion(4), two magnitudes = 18.
    # Optional magnetic vector plus its magnitude adds four.
    return 22 if CFG["use_magnetometer"] else 18


def device_prefix(value):
    match = re.match(r"([A-Za-z]+)", str(value).strip())
    return match.group(1).upper() if match else ""


def parse_seconds(series):
    """Parse time-only/full timestamps with unambiguous fractional seconds."""
    output = np.full(len(series), np.nan, np.float64)
    pattern = re.compile(
        r"^(?:(?P<date>.+?)[ T])?"
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2}):(?P<second>\d{2})"
        r"(?:[\.,:](?P<fraction>\d+))?$"
    )
    for i, value in enumerate(series.astype(str).str.strip()):
        match = pattern.match(value)
        if not match:
            continue
        hour, minute, second = (int(match.group(name)) for name in
                                ("hour", "minute", "second"))
        if hour > 23 or minute > 59 or second > 60:
            continue
        digits = match.group("fraction") or ""
        fraction = int(digits) / (10 ** len(digits)) if digits else 0.
        date_offset = 0.
        if match.group("date"):
            date = pd.to_datetime(match.group("date"), errors="coerce")
            if pd.isna(date):
                continue
            date_offset = float(date.normalize().value // 1_000_000_000)
        output[i] = date_offset + hour * 3600 + minute * 60 + second + fraction
    return output


def transform_numeric(frame):
    """Convert raw columns without temporal interpolation."""
    def values(columns):
        numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
        # Median filling only repairs isolated invalid cells; it does not create
        # synthetic time steps or smooth impact peaks.
        return numeric.fillna(numeric.median()).fillna(0).to_numpy(np.float64)
    acc, gyro = values(ACC_COLS), values(GYRO_COLS)
    angle = np.deg2rad(values(ANGLE_COLS))
    quat = values(QUAT_COLS)
    # Renormalize quaternions after CSV parsing/interpolation noise.
    quat /= np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-6)
    parts = [acc, gyro, np.sin(angle), np.cos(angle), quat,
             np.linalg.norm(acc, axis=1, keepdims=True),
             np.linalg.norm(gyro, axis=1, keepdims=True)]
    if CFG["use_magnetometer"]:
        mag = values(MAG_COLS)
        parts += [mag, np.linalg.norm(mag, axis=1, keepdims=True)]
    return np.concatenate(parts, axis=1)


def native_rows(part):
    """Return sorted/deduplicated native rows and their valid-token mask."""
    values = transform_numeric(part)
    times = parse_seconds(part["时间"])
    finite = np.isfinite(times)
    if finite.any():
        values, times = values[finite], times[finite]
    else:
        times = np.arange(len(values), dtype=np.float64)
    order = np.argsort(times, kind="stable")
    values, times = values[order], times[order]
    unique_times, inverse = np.unique(times, return_inverse=True)
    if len(unique_times) != len(times):
        merged = np.zeros((len(unique_times), values.shape[1]), np.float64)
        counts = np.zeros(len(unique_times), np.float64)
        np.add.at(merged, inverse, values); np.add.at(counts, inverse, 1.)
        values = merged / counts[:, None]
    if len(values) > CFG["max_len"]:
        # Only 0.87% of audited training sequences exceed 120. A centered
        # contiguous crop preserves native samples and avoids interpolation.
        start = (len(values) - CFG["max_len"]) // 2
        values = values[start:start + CFG["max_len"]]
    output = np.zeros((CFG["max_len"], values.shape[1]), np.float32)
    valid = np.zeros(CFG["max_len"], bool)
    output[:len(values)] = values.astype(np.float32)
    valid[:len(values)] = True
    return output, valid


def read_csv_robust(path):
    """Read one IMU CSV; return None for empty/corrupt/non-IMU files."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    for encoding in ("utf-8-sig", "gb18030", "utf-16"):
        try:
            frame = pd.read_csv(path, encoding=encoding)
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        frame.columns = [str(c).replace("\ufeff", "").strip() for c in frame.columns]
        if {"时间", "设备名称"}.issubset(frame.columns):
            # A partially exported file is still usable: unavailable numerical
            # channels become zero, while wholly invalid headers are rejected.
            needed_numeric = ACC_COLS + GYRO_COLS + ANGLE_COLS + QUAT_COLS
            if CFG["use_magnetometer"]:
                needed_numeric += MAG_COLS
            for column in needed_numeric:
                if column not in frame.columns:
                    frame[column] = 0.0
            return frame
    return None


def read_imu_pair(clip_dir):
    """Return padded native rows, sensor mask, and temporal attention mask."""
    frames = []
    for name in (UP_FILE, DOWN_FILE):
        path = clip_dir / name
        frame = read_csv_robust(path)
        if frame is not None and len(frame):
            frames.append(frame)
    feature_dim = numeric_feature_dim()
    output = np.zeros((CFG["max_len"], len(SENSORS), feature_dim), np.float32)
    mask = np.zeros(len(SENSORS), np.float32)
    time_mask = np.zeros((CFG["max_len"], len(SENSORS)), bool)
    if not frames:
        return output, mask, time_mask
    data = pd.concat(frames, ignore_index=True)
    data["_sensor"] = data["设备名称"].map(device_prefix)
    for sensor, sensor_idx in SENSOR_TO_INDEX.items():
        part = data.loc[data["_sensor"] == sensor].copy()
        if part.empty:
            continue
        output[:, sensor_idx], time_mask[:, sensor_idx] = native_rows(part)
        mask[sensor_idx] = 1.
    return (np.nan_to_num(output, nan=0., posinf=0., neginf=0.),
            mask, time_mask)


def build_train_index():
    mapping = pd.read_csv(CLASS_MAPPING_CSV, encoding="utf-8-sig")
    name_to_id = dict(zip(mapping.action_name.astype(str), mapping.action_id))
    records = []
    for action_dir in tqdm(sorted(TRAIN_ROOT.iterdir()), desc="Index IMU train"):
        if not action_dir.is_dir() or action_dir.name not in name_to_id:
            continue
        for user_dir in action_dir.iterdir():
            if not user_dir.is_dir(): continue
            for trial_dir in user_dir.iterdir():
                if trial_dir.is_dir() and ((trial_dir / UP_FILE).exists() or (trial_dir / DOWN_FILE).exists()):
                    records.append(dict(clip_dir=str(trial_dir), action_name=action_dir.name,
                                        action_id=name_to_id[action_dir.name], user=user_dir.name,
                                        trial=trial_dir.name))
    index = pd.DataFrame(records)
    if index.empty:
        raise RuntimeError(f"No IMU clips found under {TRAIN_ROOT}")
    return index


def build_cache(index, tag, clip_dirs):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fdim = numeric_feature_dim()
    x_path = OUT_DIR / f"{tag}_native_l{CFG['max_len']}_f{fdim}.npy"
    m_path = OUT_DIR / f"{tag}_sensor_mask_l{CFG['max_len']}.npy"
    tm_path = OUT_DIR / f"{tag}_time_mask_l{CFG['max_len']}.npy"
    expected_shape = (len(clip_dirs), CFG["max_len"], len(SENSORS), fdim)
    expected_mask_shape = (len(clip_dirs), len(SENSORS))
    expected_time_shape = (len(clip_dirs), CFG["max_len"], len(SENSORS))
    if x_path.exists() and m_path.exists() and tm_path.exists():
        try:
            cached_x = np.load(x_path, mmap_mode="r")
            cached_mask = np.load(m_path, mmap_mode="r")
            cached_time = np.load(tm_path, mmap_mode="r")
            if (cached_x.shape == expected_shape and
                    cached_mask.shape == expected_mask_shape and
                    cached_time.shape == expected_time_shape):
                return cached_x, cached_mask, cached_time
        except (OSError, ValueError):
            pass
    x = np.empty(expected_shape, np.float32)
    mask = np.empty((len(clip_dirs), len(SENSORS)), np.float32)
    time_mask = np.empty(expected_time_shape, bool)
    for i, directory in enumerate(tqdm(clip_dirs, desc=f"Extract {tag} IMU")):
        x[i], mask[i], time_mask[i] = read_imu_pair(Path(directory))
    np.save(x_path, x); np.save(m_path, mask); np.save(tm_path, time_mask)
    return (np.load(x_path, mmap_mode="r"), np.load(m_path, mmap_mode="r"),
            np.load(tm_path, mmap_mode="r"))


def compute_stats(x, time_mask, indices):
    """Per-sensor statistics fitted only on a fold's training subjects."""
    total = np.zeros((len(SENSORS), x.shape[-1]), np.float64)
    square = np.zeros_like(total); count = np.zeros((len(SENSORS), 1), np.float64)
    for start in range(0, len(indices), 256):
        ids = indices[start:start + 256]
        batch = np.asarray(x[ids], dtype=np.float64)
        valid = np.asarray(time_mask[ids], dtype=np.float64)[..., None]
        total += (batch * valid).sum(axis=(0, 1))
        square += (batch * batch * valid).sum(axis=(0, 1))
        count += valid.sum(axis=(0, 1))
    mean = total / np.maximum(count, 1)
    var = square / np.maximum(count, 1) - mean * mean
    return mean.astype(np.float32), np.sqrt(np.maximum(var, 1e-6)).astype(np.float32)


class IMUDataset(Dataset):
    def __init__(self, x, mask, time_mask, indices, mean, std, y=None, train=False):
        self.x, self.mask, self.time_mask = x, mask, time_mask
        self.indices = np.asarray(indices)
        self.mean, self.std = mean, std
        self.y, self.train = y, train
    def __len__(self): return len(self.indices)
    def __getitem__(self, item):
        source_idx = self.indices[item]
        sensor_mask = np.asarray(self.mask[source_idx]).copy()
        time_mask = np.asarray(self.time_mask[source_idx]).copy()
        x = np.asarray(self.x[source_idx]).copy()
        x = (x - self.mean[None]) / self.std[None]
        x *= time_mask[..., None]
        if self.train:
            # Mild calibration/noise changes preserve the action while reducing
            # dependence on a particular participant's device mounting.
            if random.random() < .5:
                x *= np.random.uniform(.92, 1.08, (1, len(SENSORS), 1)).astype(np.float32)
            if random.random() < .4:
                x += (np.random.normal(0, .025, x.shape).astype(np.float32) *
                      time_mask[..., None])
            # Match the test distribution: both leg devices are absent in many
            # more test clips than training clips. Dropping the pair together
            # teaches upper-body-only inference without fabricating values.
            if (random.random() < CFG["paired_leg_dropout"] and
                    sensor_mask[3] > 0 and sensor_mask[4] > 0):
                sensor_mask[[3, 4]] = 0.; time_mask[:, [3, 4]] = False
                x[:, [3, 4]] = 0.
            # Retain mild general sensor dropout, reduced from 15% so arm/chest
            # presence stays close to the observed test distribution.
            if (random.random() < CFG["single_sensor_dropout"] and
                    sensor_mask.sum() > 3):
                drop = random.choice(np.flatnonzero(sensor_mask).tolist())
                sensor_mask[drop] = 0.; time_mask[:, drop] = False
                x[:, drop] = 0.
        label = -1 if self.y is None else int(self.y[source_idx])
        return (torch.from_numpy(x.astype(np.float32)),
                torch.from_numpy(sensor_mask.astype(np.float32)),
                torch.from_numpy(time_mask), label)


class FactorizedSensorStream(nn.Module):
    """One complete sensor-temporal stream with no cross-stream interference."""
    def __init__(self, input_dim, n_classes):
        super().__init__()
        sd, d = CFG["sensor_dim"], CFG["model_dim"]
        self.sensor_proj = nn.Sequential(
            nn.Linear(input_dim, sd), nn.LayerNorm(sd), nn.GELU())
        self.temporal_cls = nn.Parameter(torch.zeros(1, 1, sd))
        self.time_position = nn.Parameter(torch.zeros(1, CFG["max_len"] + 1, sd))
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=sd, nhead=CFG["heads"], dim_feedforward=sd * 2,
            dropout=CFG["dropout"], activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, CFG["temporal_layers"])
        self.temporal_norm = nn.LayerNorm(sd)

        self.sensor_up = nn.Sequential(nn.Linear(sd, d), nn.LayerNorm(d), nn.GELU())
        self.sensor_embed = nn.Parameter(torch.zeros(1, len(SENSORS), d))
        self.global_cls = nn.Parameter(torch.zeros(1, 1, d))
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=CFG["heads"], dim_feedforward=CFG["ff_dim"],
            dropout=CFG["dropout"], activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(
            fusion_layer, CFG["fusion_layers"])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(CFG["dropout"]), nn.Linear(d, n_classes))
        nn.init.trunc_normal_(self.temporal_cls, std=.02)
        nn.init.trunc_normal_(self.time_position, std=.02)
        nn.init.trunc_normal_(self.sensor_embed, std=.02)
        nn.init.trunc_normal_(self.global_cls, std=.02)

    def forward(self, x, sensor_mask, time_mask):
        b, t, s, _ = x.shape
        z = self.sensor_proj(x)
        z = z * time_mask[..., None]
        z = z.permute(0, 2, 1, 3).reshape(b * s, t, -1)
        temporal_cls = self.temporal_cls.expand(b * s, -1, -1)
        z = torch.cat([temporal_cls, z], dim=1) + self.time_position[:, :t + 1]
        temporal_padding = ~time_mask.permute(0, 2, 1).reshape(b * s, t).bool()
        temporal_padding = torch.cat([
            torch.zeros((b * s, 1), dtype=torch.bool, device=x.device),
            temporal_padding,
        ], dim=1)
        z = self.temporal_norm(self.temporal_encoder(
            z, src_key_padding_mask=temporal_padding)[:, 0])
        z = self.sensor_up(z.reshape(b, s, -1)) + self.sensor_embed[:, :s]

        global_cls = self.global_cls.expand(b, -1, -1)
        z = torch.cat([global_cls, z], dim=1)
        padding_mask = torch.cat([
            torch.zeros((b, 1), dtype=torch.bool, device=x.device),
            sensor_mask[:, :s] <= 0,
        ], dim=1)
        z = self.fusion_encoder(z, src_key_padding_mask=padding_mask)
        return self.head(self.norm(z[:, 0]))


class TinyIMUTransformer(nn.Module):
    """Independent raw/motion streams fused only at the classifier logits."""
    def __init__(self, n_classes, feature_dim):
        super().__init__()
        self.raw_stream = FactorizedSensorStream(feature_dim, n_classes)
        self.motion_stream = FactorizedSensorStream(feature_dim * 2, n_classes)
        initial = float(CFG["initial_motion_weight"])
        self.motion_weight_logit = nn.Parameter(torch.tensor(
            math.log(initial / (1. - initial)), dtype=torch.float32))

    def forward(self, x, sensor_mask, time_mask, return_branches=False):
        delta = torch.zeros_like(x)
        delta[:, 1:] = x[:, 1:] - x[:, :-1]
        adjacent = time_mask[:, 1:] & time_mask[:, :-1]
        delta[:, 1:] *= adjacent[..., None]
        motion = torch.cat([delta, delta.abs()], dim=-1)
        raw_logits = self.raw_stream(x, sensor_mask, time_mask)
        motion_logits = self.motion_stream(motion, sensor_mask, time_mask)
        motion_weight = torch.sigmoid(self.motion_weight_logit)
        fused_logits = ((1. - motion_weight) * raw_logits +
                        motion_weight * motion_logits)
        if return_branches:
            return fused_logits, raw_logits, motion_logits
        return fused_logits


def scheduled_lr(step, total, warmup):
    if step < warmup:
        return CFG["lr"] * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return CFG["min_lr"] + .5 * (CFG["lr"] - CFG["min_lr"]) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def predict(model, loader):
    model.eval(); probabilities = []
    for x, mask, time_mask, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)
        time_mask = time_mask.to(DEVICE, non_blocking=True)
        probabilities.append(torch.softmax(
            model(x, mask, time_mask), 1).cpu().numpy())
    return np.concatenate(probabilities)


def train_model(train_ds, val_ds, n_classes, epochs, seed, fold=None):
    seed_everything(seed)
    train_loader = DataLoader(train_ds, CFG["batch_size"], shuffle=True, drop_last=False,
                              num_workers=CFG["workers"], pin_memory=True)
    val_loader = None if val_ds is None else DataLoader(
        val_ds, CFG["batch_size"] * 2, shuffle=False,
        num_workers=CFG["workers"], pin_memory=True)
    model = TinyIMUTransformer(n_classes, numeric_feature_dim()).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scaler = GradScaler(enabled=CFG["amp"] and DEVICE.type == "cuda")
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * min(CFG["warmup_epochs"], max(epochs // 3, 1))
    best_acc, best_epoch, best_sd, step = -1., epochs, None, 0
    for epoch in range(1, epochs + 1):
        model.train(); loss_sum = fused_sum = raw_sum = motion_sum = 0.; seen = 0
        for x, mask, time_mask, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            time_mask = time_mask.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            lr = scheduled_lr(step, total_steps, warmup_steps)
            for group in optimizer.param_groups: group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                fused_logits, raw_logits, motion_logits = model(
                    x, mask, time_mask, return_branches=True)
                fused_loss = F.cross_entropy(
                    fused_logits, y, label_smoothing=CFG["label_smoothing"])
                raw_loss = F.cross_entropy(
                    raw_logits, y, label_smoothing=CFG["label_smoothing"])
                motion_loss = F.cross_entropy(
                    motion_logits, y, label_smoothing=CFG["label_smoothing"])
                loss = (fused_loss + CFG["auxiliary_loss_weight"] *
                        (raw_loss + motion_loss))
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 3.)
            scaler.step(optimizer); scaler.update(); step += 1
            loss_sum += loss.item() * len(y)
            fused_sum += fused_loss.item() * len(y)
            raw_sum += raw_loss.item() * len(y)
            motion_sum += motion_loss.item() * len(y)
            seen += len(y)
        if val_loader is not None:
            prob = predict(model, val_loader)
            val_y = np.array([val_ds.y[i] for i in val_ds.indices])
            accuracy = float((prob.argmax(1) == val_y).mean())
            if accuracy > best_acc:
                best_acc, best_epoch = accuracy, epoch
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            weight = torch.sigmoid(model.motion_weight_logit.detach()).item()
            print(f"fold {fold} epoch {epoch:02d}/{epochs}: "
                  f"fused={fused_sum/max(seen,1):.4f} "
                  f"raw={raw_sum/max(seen,1):.4f} "
                  f"motion={motion_sum/max(seen,1):.4f} "
                  f"motion_w={weight:.3f} val={accuracy:.4f} best={best_acc:.4f}")
        else:
            weight = torch.sigmoid(model.motion_weight_logit.detach()).item()
            print(f"final seed {seed} epoch {epoch:02d}/{epochs}: "
                  f"fused={fused_sum/max(seen,1):.4f} "
                  f"raw={raw_sum/max(seen,1):.4f} "
                  f"motion={motion_sum/max(seen,1):.4f} motion_w={weight:.3f}")
    if val_loader is None:
        best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_sd)
    motion_weight = torch.sigmoid(model.motion_weight_logit.detach()).item()
    prefix = "final" if fold is None else f"fold {fold}"
    print(f"{prefix}: learned late-fusion motion weight={motion_weight:.3f}")
    return model, best_acc, best_epoch, (None if val_loader is None else predict(model, val_loader))


def compact_state_dict(state_dict):
    return {k: (v.half() if v.is_floating_point() else v) for k, v in state_dict.items()}


def save_final(model, model_id, mean, std, label_values, epochs):
    path = OUT_DIR / f"imu_final_{model_id}.pt"
    torch.save(dict(state_dict=compact_state_dict(model.state_dict()), mean=mean,
                    std=std, label_values=np.asarray(label_values),
                    sensors=SENSORS, feature_dim=numeric_feature_dim(),
                    max_len=CFG["max_len"],
                    architecture_version=CFG["architecture_version"], epochs=epochs), path)
    total_mb = sum(p.stat().st_size for p in OUT_DIR.glob("imu_final_*.pt")) / 1024**2
    if total_mb > CFG["max_total_weight_mb"]:
        raise RuntimeError(f"IMU final weights total {total_mb:.3f} MB exceeds {CFG['max_total_weight_mb']} MB")
    print(f"Saved {path.name}: {path.stat().st_size/1024**2:.3f} MB; total={total_mb:.3f} MB")


def test_clip_dirs(test):
    return [str(TEST_ROOT / Path(str(p).rstrip("/")).name / "IMU") for p in test["path"]]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("imu_final_*.pt"):
        old.unlink()
    seed_everything(CFG["seed"])
    index = build_train_index()
    train_x, train_mask, train_time_mask = build_cache(
        index, "train", index.clip_dir.tolist())
    presence = np.asarray(train_mask).mean(axis=0)
    print("Training clips:", len(index), "sensor presence:",
          {name: f"{presence[i]:.2%}" for i, name in enumerate(SENSORS)})
    print("Training sensor dropout:",
          {"paired_legs": f"{CFG['paired_leg_dropout']:.1%}",
           "single_sensor": f"{CFG['single_sensor_dropout']:.1%}"})
    # Validate every test file before spending time on cross-validation.
    test = pd.read_csv(TEST_CSV)
    if not {"path", "prediction"}.issubset(test.columns):
        raise ValueError(f"Unexpected test.csv columns: {test.columns.tolist()}")
    test_x, test_mask, test_time_mask = build_cache(
        test, "test", test_clip_dirs(test))
    test_presence = np.asarray(test_mask).mean(axis=0)
    empty_test = np.flatnonzero(np.asarray(test_mask).sum(axis=1) == 0)
    print("Test clips:", len(test), "sensor presence:",
          {name: f"{test_presence[i]:.2%}" for i, name in enumerate(SENSORS)})
    train_lengths = np.asarray(train_time_mask).sum(axis=1)
    test_lengths = np.asarray(test_time_mask).sum(axis=1)
    print("Native valid rows per present sensor (train/test medians):",
          {name: (float(np.median(train_lengths[train_mask[:, i] > 0, i])),
                  float(np.median(test_lengths[test_mask[:, i] > 0, i])))
           for i, name in enumerate(SENSORS)})
    if len(empty_test):
        names = [Path(str(test.iloc[i]["path"]).rstrip("/")).name for i in empty_test[:20]]
        print(f"WARNING: {len(empty_test)} test clips have no readable IMU sensor; zero features used. First clips: {names}")
    label_values = np.sort(index.action_id.unique())
    to_class = {value: i for i, value in enumerate(label_values)}
    y = index.action_id.map(to_class).to_numpy(np.int64)
    groups = index.user.astype(str).to_numpy()
    probe = TinyIMUTransformer(len(label_values), numeric_feature_dim())
    params = sum(p.numel() for p in probe.parameters())
    print(f"IMU {CFG['architecture_version']} parameters: {params:,}; "
          f"estimated two-model FP16 weights: {params*2*CFG['final_models']/1024**2:.3f} MB")
    del probe

    oof = np.zeros((len(index), len(label_values)), np.float32)
    fold_ids = np.full(len(index), -1, np.int64)
    fold_scores, best_epochs = [], []
    splitter = StratifiedGroupKFold(CFG["folds"], shuffle=True, random_state=CFG["seed"])
    for fold, (tr, va) in enumerate(splitter.split(index, y, groups)):
        mean, std = compute_stats(train_x, train_time_mask, tr)
        train_ds = IMUDataset(
            train_x, train_mask, train_time_mask, tr, mean, std, y, True)
        val_ds = IMUDataset(
            train_x, train_mask, train_time_mask, va, mean, std, y, False)
        _, score, best_epoch, prob = train_model(
            train_ds, val_ds, len(label_values), CFG["epochs"], CFG["seed"] + fold, fold)
        oof[va] = prob; fold_ids[va] = fold
        fold_scores.append(score); best_epochs.append(best_epoch)
    oof_pred = oof.argmax(1)
    print(f"IMU 5-fold CV: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    final_epochs = max(12, int(round(np.mean(best_epochs))))
    oof_meta = index[["action_name", "action_id", "user", "trial"]].copy()
    oof_meta.insert(0, "row_id", np.arange(len(index)))
    oof_meta["fold"] = fold_ids; oof_meta["prediction"] = label_values[oof_pred]
    oof_meta["correct"] = oof_pred == y; oof_meta["confidence"] = oof.max(1)
    oof_meta.to_csv(OUT_DIR / "imu_oof_meta.csv", index=False)
    np.save(OUT_DIR / "imu_oof_probs.npy", oof)
    np.save(OUT_DIR / "imu_label_values.npy", label_values)

    all_indices = np.arange(len(index)); test_indices = np.arange(len(test))
    mean, std = compute_stats(train_x, train_time_mask, all_indices)
    test_ds = IMUDataset(
        test_x, test_mask, test_time_mask, test_indices,
        mean, std, None, False)
    test_loader = DataLoader(test_ds, CFG["batch_size"] * 2, shuffle=False,
                             num_workers=CFG["workers"], pin_memory=True)
    test_prob = np.zeros((len(test), len(label_values)), np.float32)
    for model_id in range(CFG["final_models"]):
        train_ds = IMUDataset(
            train_x, train_mask, train_time_mask, all_indices,
            mean, std, y, True)
        model, _, _, _ = train_model(train_ds, None, len(label_values), final_epochs,
                                     CFG["seed"] + 100 + model_id)
        test_prob += predict(model, test_loader) / CFG["final_models"]
        save_final(model, model_id, mean, std, label_values, final_epochs)
    np.save(OUT_DIR / "imu_test_probs.npy", test_prob)
    submission = test[["path"]].copy()
    submission["prediction"] = label_values[test_prob.argmax(1)]
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Wrote {SUBMISSION_PATH}; rows={len(submission)}")


if __name__ == "__main__":
    main()
