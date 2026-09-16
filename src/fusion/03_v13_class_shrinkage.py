from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


@dataclass(frozen=True)
class Config:
    v12_root: Path = Path("/kaggle/input/data-ooo/v12_visual_thermal_v16_fusion")
    visual_val_probs: Path = Path("/kaggle/input/datasets/zhuowamg/ir-depth-color/outputs/val_probs_fold0.npy")
    visual_val_meta: Path = Path("/kaggle/input/datasets/zhuowamg/ir-depth-color/outputs/val_pred_fold0.parquet")
    visual_test_probs: Path = Path("/kaggle/input/datasets/zhuowamg/newensembledata/visual_test_probs.npy")
    thermal_test_probs: Path = Path("/kaggle/input/datasets/zhuowamg/thermalv16/thermal_test_probs (2).npy")
    test_csv: Path = Path("/kaggle/input/datasets/zhuowamg/cuhksmall/CUHK-X_dataset/Small-Model-Track/Testing/test_file/test.csv")
    test_root: Path = Path("/kaggle/input/datasets/zhuowamg/cuhksmall/CUHK-X_dataset/Small-Model-Track/Testing/data/small_model_track_test/small_model_track_test")
    work_root: Path = Path("/kaggle/working")

    n_classes: int = 40
    expected_aligned: int = 541
    expected_test: int = 405
    expected_users: tuple[str, ...] = ("user9", "user16", "user18")
    expected_missing_thermal: int = 10

    # Fixed before looking at V13 labels: V12's central calibration is used only
    # to define a Visual->Thermal direction. Delta=0 is exactly the V12 anchor.
    visual_temperature: float = 2.0
    thermal_temperature: float = 0.75
    l2_strength: float = 4.0
    delta_bound: float = 0.08
    min_abs_delta: float = 0.01
    min_class_support_per_user: int = 2
    conservative_shrink: float = 0.50

    @property
    def out_dir(self) -> Path:
        return self.work_root / "v13_class_shrinkage"


CFG = Config()
EPS = 1e-12
LABEL_VALUES = np.arange(CFG.n_classes, dtype=np.int64)


def normalize_probs(values, name: str, expected_rows: int | None = None) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != CFG.n_classes:
        raise ValueError(f"{name}: expected (N,{CFG.n_classes}), got {p.shape}")
    if expected_rows is not None and len(p) != expected_rows:
        raise ValueError(f"{name}: expected {expected_rows} rows, got {len(p)}")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError(f"{name}: contains NaN/Inf or negative values")
    mass = p.sum(axis=1, keepdims=True)
    if (mass <= 0).any():
        raise ValueError(f"{name}: contains a zero-mass row")
    return p / mass


def temperature_scale(p: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"Invalid temperature: {temperature}")
    logits = np.log(np.clip(p, EPS, 1.0)) / temperature
    logits -= logsumexp(logits, axis=1, keepdims=True)
    return np.exp(logits)


def class_direction(visual: np.ndarray, thermal: np.ndarray) -> np.ndarray:
    qv = temperature_scale(visual, CFG.visual_temperature)
    qt = temperature_scale(thermal, CFG.thermal_temperature)
    return np.log(np.clip(qt, EPS, 1.0)) - np.log(np.clip(qv, EPS, 1.0))


def apply_delta(base: np.ndarray, direction: np.ndarray, delta: np.ndarray) -> np.ndarray:
    if base.shape != direction.shape or delta.shape != (CFG.n_classes,):
        raise ValueError(f"apply_delta shape mismatch: base={base.shape}, direction={direction.shape}, delta={delta.shape}")
    logits = np.log(np.clip(base, EPS, 1.0)) + direction * delta[None, :]
    logits -= logsumexp(logits, axis=1, keepdims=True)
    return np.exp(logits)


def macro_user_objective(delta: np.ndarray, base: np.ndarray, direction: np.ndarray,
                         y: np.ndarray, users: np.ndarray) -> tuple[float, np.ndarray]:
    losses, gradients = [], []
    for user in sorted(np.unique(users)):
        ids = np.flatnonzero(users == user)
        logits = np.log(np.clip(base[ids], EPS, 1.0)) + direction[ids] * delta[None, :]
        log_norm = logsumexp(logits, axis=1, keepdims=True)
        probs = np.exp(logits - log_norm)
        losses.append(float(np.mean(log_norm[:, 0] - logits[np.arange(len(ids)), y[ids]])))
        residual = probs
        residual[np.arange(len(ids)), y[ids]] -= 1.0
        gradients.append(np.mean(residual * direction[ids], axis=0))
    loss = float(np.mean(losses) + CFG.l2_strength * np.mean(delta * delta))
    grad = np.mean(gradients, axis=0) + (2.0 * CFG.l2_strength / CFG.n_classes) * delta
    return loss, grad


def gradient_at_zero(base: np.ndarray, direction: np.ndarray, y: np.ndarray,
                     users: np.ndarray, user: str) -> np.ndarray:
    ids = np.flatnonzero(users == user)
    residual = base[ids].copy()
    residual[np.arange(len(ids)), y[ids]] -= 1.0
    return np.mean(residual * direction[ids], axis=0)


def fit_filtered_delta(train_ids: np.ndarray, base: np.ndarray, direction: np.ndarray,
                       y: np.ndarray, users: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    train_users = sorted(np.unique(users[train_ids]))
    if len(train_users) != 2:
        raise ValueError(f"Each outer fit must use exactly two users, got {train_users}")
    b, d, target, group = base[train_ids], direction[train_ids], y[train_ids], users[train_ids]
    result = minimize(
        fun=lambda x: macro_user_objective(x, b, d, target, group),
        x0=np.zeros(CFG.n_classes, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-CFG.delta_bound, CFG.delta_bound)] * CFG.n_classes,
        options={"maxiter": 250, "ftol": 1e-12, "gtol": 1e-8, "maxls": 30},
    )
    if not result.success:
        raise RuntimeError(f"L-BFGS-B failed: {result.message}")
    raw = np.asarray(result.x, dtype=np.float64)
    descent_signs, supports = [], []
    for user in train_users:
        descent_signs.append(np.sign(-gradient_at_zero(b, d, target, group, user)))
        supports.append(np.bincount(target[group == user], minlength=CFG.n_classes))
    descent_signs = np.stack(descent_signs)
    supports = np.stack(supports)
    stable_sign = (descent_signs[0] != 0) & (descent_signs[0] == descent_signs[1])
    agrees_with_fit = np.sign(raw) == descent_signs[0]
    enough_support = supports.min(axis=0) >= CFG.min_class_support_per_user
    large_enough = np.abs(raw) >= CFG.min_abs_delta
    keep = stable_sign & agrees_with_fit & enough_support & large_enough
    filtered = np.where(keep, raw, 0.0)
    info = {
        "optimizer_iterations": int(result.nit),
        "optimizer_objective": float(result.fun),
        "raw_nonzero": int((np.abs(raw) >= CFG.min_abs_delta).sum()),
        "kept_classes": int(keep.sum()),
    }
    return raw, filtered, {**info, "supports": supports, "descent_signs": descent_signs, "keep": keep}


def classification_stats(name: str, p: np.ndarray, y: np.ndarray,
                         reference: np.ndarray) -> dict:
    pred, ref = p.argmax(axis=1), reference.argmax(axis=1)
    correct, ref_correct = pred == y, ref == y
    return {
        "method": name,
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0)).mean()),
        "rescued_vs_v12": int((~ref_correct & correct).sum()),
        "harmed_vs_v12": int((ref_correct & ~correct).sum()),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_submission(p: np.ndarray, path: Path, test_csv: pd.DataFrame) -> pd.DataFrame:
    submission = test_csv[["path"]].copy()
    submission["prediction"] = LABEL_VALUES[p.argmax(axis=1)]
    if (len(submission) != CFG.expected_test or submission.prediction.isna().any() or
            not submission.prediction.between(0, CFG.n_classes - 1).all()):
        raise ValueError(f"Submission contract failed: {path}")
    submission.to_csv(path, index=False)
    return submission


start_time = time.perf_counter()
CFG.out_dir.mkdir(parents=True, exist_ok=True)
required = [
    CFG.v12_root / "v16_aligned_probs.npy",
    CFG.v12_root / "aligned_meta.parquet",
    CFG.v12_root / "v12_shrunk_oof_probs.npy",
    CFG.v12_root / "v12_shrunk_test_probs.npy",
    CFG.v12_root / "outer_user_parameters.csv",
    CFG.visual_val_probs,
    CFG.visual_val_meta,
    CFG.visual_test_probs,
    CFG.thermal_test_probs,
    CFG.test_csv,
    CFG.test_root,
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("Missing V13 input(s):\n" + "\n".join(missing))

meta = pd.read_parquet(CFG.v12_root / "aligned_meta.parquet").reset_index(drop=True)
needed_columns = {"clip_id", "user", "action_id"}
if not needed_columns.issubset(meta.columns):
    raise ValueError(f"aligned_meta.parquet missing {needed_columns - set(meta.columns)}")
if len(meta) != CFG.expected_aligned or meta.clip_id.astype(str).duplicated().any():
    raise ValueError(f"Expected {CFG.expected_aligned} unique aligned rows, got {len(meta)}")
users = meta.user.astype(str).to_numpy()
y = meta.action_id.to_numpy(np.int64)
if tuple(sorted(np.unique(users))) != tuple(sorted(CFG.expected_users)):
    raise ValueError(f"Unexpected aligned users: {sorted(np.unique(users))}")
if not np.isin(y, LABEL_VALUES).all():
    raise ValueError("aligned_meta contains an action_id outside 0..39")

base_oof = normalize_probs(np.load(CFG.v12_root / "v12_shrunk_oof_probs.npy"), "V12 OOF", CFG.expected_aligned)
base_test = normalize_probs(np.load(CFG.v12_root / "v12_shrunk_test_probs.npy"), "V12 test", CFG.expected_test)
thermal_oof = normalize_probs(np.load(CFG.v12_root / "v16_aligned_probs.npy"), "V16 aligned", CFG.expected_aligned)

visual_meta = pd.read_parquet(CFG.visual_val_meta).reset_index(drop=True)
visual_all = normalize_probs(np.load(CFG.visual_val_probs), "Visual validation")
if len(visual_meta) != len(visual_all) or "clip_id" not in visual_meta.columns:
    raise ValueError("Visual validation metadata/probability mismatch")
visual_index = {str(clip_id): row for row, clip_id in enumerate(visual_meta.clip_id)}
if len(visual_index) != len(visual_meta):
    raise ValueError("Visual validation clip_id is not unique")
unknown = [str(clip_id) for clip_id in meta.clip_id if str(clip_id) not in visual_index]
if unknown:
    raise ValueError(f"Visual validation is missing aligned clip_id: {unknown[:3]}")
visual_oof = visual_all[[visual_index[str(clip_id)] for clip_id in meta.clip_id]]

test_csv = pd.read_csv(CFG.test_csv)
if "path" not in test_csv.columns or len(test_csv) != CFG.expected_test:
    raise ValueError("test.csv must contain exactly 405 rows and a path column")
visual_test = normalize_probs(np.load(CFG.visual_test_probs), "Visual test", CFG.expected_test)
thermal_test = normalize_probs(np.load(CFG.thermal_test_probs), "V16 Thermal test", CFG.expected_test)

image_suffixes = {".jpg", ".jpeg", ".png"}
thermal_available = []
for path_value in test_csv.path.astype(str):
    clip_id = path_value.strip("/\\").replace("\\", "/").split("/")[-1]
    folder = CFG.test_root / clip_id / "Thermal"
    thermal_available.append(folder.is_dir() and any(p.suffix.lower() in image_suffixes for p in folder.iterdir()))
thermal_available = np.asarray(thermal_available, dtype=bool)
if int((~thermal_available).sum()) != CFG.expected_missing_thermal:
    raise ValueError(f"Expected {CFG.expected_missing_thermal} missing Thermal clips, found {int((~thermal_available).sum())}")

direction_oof = class_direction(visual_oof, thermal_oof)
direction_test = class_direction(visual_test, thermal_test)
zero_anchor = apply_delta(base_oof, direction_oof, np.zeros(CFG.n_classes))
if not np.allclose(zero_anchor, base_oof, rtol=1e-10, atol=1e-12):
    raise RuntimeError("Delta=0 failed to reproduce the V12 anchor")

oof_conservative = np.zeros_like(base_oof)
oof_moderate = np.zeros_like(base_oof)
test_conservative_folds, test_moderate_folds = [], []
fold_rows, class_rows = [], []

for held_user in sorted(np.unique(users)):
    train_ids = np.flatnonzero(users != held_user)
    held_ids = np.flatnonzero(users == held_user)
    raw, filtered, info = fit_filtered_delta(train_ids, base_oof, direction_oof, y, users)
    conservative = CFG.conservative_shrink * filtered
    fold_oof_conservative = apply_delta(base_oof[held_ids], direction_oof[held_ids], conservative)
    fold_oof_moderate = apply_delta(base_oof[held_ids], direction_oof[held_ids], filtered)
    oof_conservative[held_ids] = fold_oof_conservative
    oof_moderate[held_ids] = fold_oof_moderate

    fold_test_conservative = apply_delta(base_test, direction_test, conservative)
    fold_test_moderate = apply_delta(base_test, direction_test, filtered)
    # No Thermal means no class-level Thermal correction: preserve the proven anchor exactly.
    fold_test_conservative[~thermal_available] = base_test[~thermal_available]
    fold_test_moderate[~thermal_available] = base_test[~thermal_available]
    test_conservative_folds.append(fold_test_conservative)
    test_moderate_folds.append(fold_test_moderate)

    base_pred = base_oof[held_ids].argmax(1)
    for candidate_name, candidate in (("conservative", fold_oof_conservative), ("moderate", fold_oof_moderate)):
        pred = candidate.argmax(1)
        base_ok, candidate_ok = base_pred == y[held_ids], pred == y[held_ids]
        fold_rows.append({
            "held_user": held_user,
            "candidate": candidate_name,
            "n": int(len(held_ids)),
            "correct": int(candidate_ok.sum()),
            "accuracy": float(candidate_ok.mean()),
            "v12_correct": int(base_ok.sum()),
            "rescued_vs_v12": int((~base_ok & candidate_ok).sum()),
            "harmed_vs_v12": int((base_ok & ~candidate_ok).sum()),
            "kept_classes": info["kept_classes"],
            "optimizer_iterations": info["optimizer_iterations"],
            "optimizer_objective": info["optimizer_objective"],
        })
    train_users = sorted(np.unique(users[train_ids]))
    for class_id in LABEL_VALUES:
        class_rows.append({
            "held_user": held_user,
            "class_id": int(class_id),
            "raw_delta": float(raw[class_id]),
            "filtered_delta": float(filtered[class_id]),
            "conservative_delta": float(conservative[class_id]),
            "kept": bool(info["keep"][class_id]),
            f"support_{train_users[0]}": int(info["supports"][0, class_id]),
            f"support_{train_users[1]}": int(info["supports"][1, class_id]),
            f"descent_sign_{train_users[0]}": int(info["descent_signs"][0, class_id]),
            f"descent_sign_{train_users[1]}": int(info["descent_signs"][1, class_id]),
        })

test_conservative = normalize_probs(np.mean(test_conservative_folds, axis=0), "V13 conservative test", CFG.expected_test)
test_moderate = normalize_probs(np.mean(test_moderate_folds, axis=0), "V13 moderate test", CFG.expected_test)
test_conservative[~thermal_available] = base_test[~thermal_available]
test_moderate[~thermal_available] = base_test[~thermal_available]

scores = pd.DataFrame([
    classification_stats("V12_0766_anchor", base_oof, y, base_oof),
    classification_stats("V13_class_conservative", oof_conservative, y, base_oof),
    classification_stats("V13_class_moderate", oof_moderate, y, base_oof),
])
fold_report = pd.DataFrame(fold_rows)
class_report = pd.DataFrame(class_rows)

primary_path = CFG.work_root / "submission.csv"
moderate_path = CFG.work_root / "submission_v13_class_moderate.csv"
reference_path = CFG.work_root / "submission_v12_0766_reference.csv"
primary_submission = write_submission(test_conservative, primary_path, test_csv)
write_submission(test_moderate, moderate_path, test_csv)
write_submission(base_test, reference_path, test_csv)

np.save(CFG.out_dir / "v13_class_conservative_oof_probs.npy", oof_conservative.astype(np.float32))
np.save(CFG.out_dir / "v13_class_conservative_test_probs.npy", test_conservative.astype(np.float32))
np.save(CFG.out_dir / "v13_class_moderate_oof_probs.npy", oof_moderate.astype(np.float32))
np.save(CFG.out_dir / "v13_class_moderate_test_probs.npy", test_moderate.astype(np.float32))
meta.to_parquet(CFG.out_dir / "aligned_meta.parquet", index=False)
scores.to_csv(CFG.out_dir / "fusion_scores.csv", index=False)
fold_report.to_csv(CFG.out_dir / "outer_user_report.csv", index=False)
class_report.to_csv(CFG.out_dir / "outer_user_class_deltas.csv", index=False)

base_test_pred = base_test.argmax(1)
primary_test_pred = test_conservative.argmax(1)
moderate_test_pred = test_moderate.argmax(1)
summary = {
    "method": "V13 class-shrunken residual fusion anchored at V12",
    "public_score_anchor_user_reported": 0.766,
    "aligned_rows": int(len(y)),
    "aligned_users": sorted(np.unique(users).tolist()),
    "missing_thermal_test": int((~thermal_available).sum()),
    "v12_oof_correct": int((base_oof.argmax(1) == y).sum()),
    "v13_conservative_oof_correct": int((oof_conservative.argmax(1) == y).sum()),
    "v13_moderate_oof_correct": int((oof_moderate.argmax(1) == y).sum()),
    "test_changes_conservative_vs_v12": int((primary_test_pred != base_test_pred).sum()),
    "test_changes_moderate_vs_v12": int((moderate_test_pred != base_test_pred).sum()),
    "test_changes_moderate_vs_conservative": int((moderate_test_pred != primary_test_pred).sum()),
    "kept_classes_per_outer_fold": class_report.groupby("held_user").kept.sum().astype(int).to_dict(),
    "config": {
        "visual_temperature": CFG.visual_temperature,
        "thermal_temperature": CFG.thermal_temperature,
        "l2_strength": CFG.l2_strength,
        "delta_bound": CFG.delta_bound,
        "min_abs_delta": CFG.min_abs_delta,
        "min_class_support_per_user": CFG.min_class_support_per_user,
        "conservative_shrink": CFG.conservative_shrink,
    },
    "runtime_seconds": float(time.perf_counter() - start_time),
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "primary_submission": str(primary_path),
    "primary_sha256": sha256(primary_path),
    "note": "No leaderboard labels, model inference, automatic fallback, or hyperparameter grid were used.",
}
(CFG.out_dir / "fusion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\nV13 CLASS-SHRINKAGE OOF SCORES")
print(scores.to_string(index=False))
print("\nOUTER-USER DIAGNOSTICS")
print(fold_report.to_string(index=False))
print("\nV13 FINAL SUMMARY")
print(json.dumps(summary, indent=2, ensure_ascii=False))
if summary["test_changes_conservative_vs_v12"] > 25:
    print("WARNING: conservative V13 changes more than 25/405 test labels; inspect reports before submission.")
if summary["test_changes_conservative_vs_v12"] == 0:
    print("WARNING: V13 primary is prediction-identical to the V12 0.766 anchor.")
print("\nWROTE")
print("/kaggle/working/submission.csv                         # V13 conservative primary")
print("/kaggle/working/submission_v13_class_moderate.csv      # larger diagnostic candidate")
print("/kaggle/working/submission_v12_0766_reference.csv      # exact V12 reference")
print("/kaggle/working/v13_class_shrinkage/                   # probabilities and audit reports")
print(primary_submission.head())
