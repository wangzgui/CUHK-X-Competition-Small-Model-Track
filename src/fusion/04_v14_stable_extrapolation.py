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


@dataclass(frozen=True)
class Config:
    v13_root: Path = Path("/kaggle/input/datasets/zhuowamg/v13output/v13_class_shrinkage")
    test_csv: Path = Path("/kaggle/input/datasets/zhuowamg/cuhksmall/CUHK-X_dataset/Small-Model-Track/Testing/test_file/test.csv")
    work_root: Path = Path("/kaggle/working")

    n_classes: int = 40
    expected_oof_rows: int = 541
    expected_test_rows: int = 405
    expected_users: tuple[str, ...] = ("user9", "user16", "user18")
    v13_delta_bound: float = 0.08
    bound_tolerance: float = 1e-6

    # V13 moderate is scale=1.0. Only classes that were retained with the same
    # sign and hit the V13 bound in all three outer-user fits are extrapolated.
    primary_scale: float = 2.5
    exploratory_scale: float = 3.0
    expected_stable_classes: tuple[int, ...] = (17, 21, 37)
    expected_primary_test_changes: int = 4
    expected_exploratory_test_changes: int = 5

    @property
    def out_dir(self) -> Path:
        return self.work_root / "v14_stable_class_extrapolation"


CFG = Config()
EPS = 1e-12
LABEL_VALUES = np.arange(CFG.n_classes, dtype=np.int64)


def normalize_probs(values, name: str, rows: int) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    if p.shape != (rows, CFG.n_classes):
        raise ValueError(f"{name}: expected {(rows, CFG.n_classes)}, got {p.shape}")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError(f"{name}: contains NaN/Inf or negative values")
    mass = p.sum(axis=1, keepdims=True)
    if (mass <= 0).any():
        raise ValueError(f"{name}: contains a zero-mass row")
    return p / mass


def centered_log_probs(p: np.ndarray) -> np.ndarray:
    """Canonical log-ratio coordinates, invariant to softmax row constants."""
    z = np.log(np.clip(p, EPS, 1.0))
    return z - z.mean(axis=1, keepdims=True)


def discover_stable_saturated_classes(delta_table: pd.DataFrame) -> tuple[int, ...]:
    required = {"held_user", "class_id", "raw_delta", "filtered_delta", "kept"}
    if not required.issubset(delta_table.columns):
        raise ValueError(f"outer_user_class_deltas.csv missing {required - set(delta_table.columns)}")
    table = delta_table.copy()
    table["held_user"] = table.held_user.astype(str)
    table["class_id"] = table.class_id.astype(int)
    if tuple(sorted(table.held_user.unique())) != tuple(sorted(CFG.expected_users)):
        raise ValueError(f"Unexpected outer users: {sorted(table.held_user.unique())}")
    if len(table) != len(CFG.expected_users) * CFG.n_classes:
        raise ValueError(f"Expected {len(CFG.expected_users) * CFG.n_classes} class rows, got {len(table)}")
    if table.duplicated(["held_user", "class_id"]).any():
        raise ValueError("Duplicate held_user/class_id rows in class-delta table")
    if not np.isin(table.class_id, LABEL_VALUES).all():
        raise ValueError("Class-delta table contains a class outside 0..39")
    if table.kept.dtype != bool:
        table["kept"] = table.kept.astype(str).str.lower().map({"true": True, "false": False})
    if table.kept.isna().any():
        raise ValueError("Could not parse the kept column as booleans")

    selected = []
    for class_id, group in table.groupby("class_id", sort=True):
        if len(group) != len(CFG.expected_users) or not bool(group.kept.all()):
            continue
        filtered = group.filtered_delta.to_numpy(np.float64)
        raw = group.raw_delta.to_numpy(np.float64)
        same_nonzero_sign = np.all(np.sign(filtered) == np.sign(filtered[0])) and np.sign(filtered[0]) != 0
        all_at_bound = np.all(np.abs(raw) >= CFG.v13_delta_bound - CFG.bound_tolerance)
        if same_nonzero_sign and all_at_bound:
            selected.append(int(class_id))
    return tuple(selected)


def selective_extrapolate(conservative: np.ndarray, moderate: np.ndarray,
                          selected_classes: tuple[int, ...], scale: float) -> np.ndarray:
    if scale < 1.0:
        raise ValueError("V14 extrapolation scale must be at least 1.0")
    # V13 conservative used 0.5*delta and moderate used 1.0*delta. Therefore
    # 2*(moderate-conservative) is one full delta step in log-ratio space.
    step = centered_log_probs(moderate) - centered_log_probs(conservative)
    class_multiplier = np.zeros(CFG.n_classes, dtype=np.float64)
    class_multiplier[list(selected_classes)] = 2.0 * (scale - 1.0)
    logits = np.log(np.clip(moderate, EPS, 1.0)) + step * class_multiplier[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def nll(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0)).mean())


def score(name: str, p: np.ndarray, y: np.ndarray, reference: np.ndarray) -> dict:
    pred, ref = p.argmax(axis=1), reference.argmax(axis=1)
    correct, ref_correct = pred == y, ref == y
    return {
        "method": name,
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "nll": nll(p, y),
        "changes_vs_v13_moderate": int((pred != ref).sum()),
        "rescued_vs_v13_moderate": int((~ref_correct & correct).sum()),
        "harmed_vs_v13_moderate": int((ref_correct & ~correct).sum()),
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
    if (len(submission) != CFG.expected_test_rows or submission.prediction.isna().any() or
            not submission.prediction.between(0, CFG.n_classes - 1).all()):
        raise ValueError(f"Submission contract failed: {path}")
    submission.to_csv(path, index=False)
    return submission


start = time.perf_counter()
CFG.out_dir.mkdir(parents=True, exist_ok=True)
paths = {
    "conservative_oof": CFG.v13_root / "v13_class_conservative_oof_probs.npy",
    "conservative_test": CFG.v13_root / "v13_class_conservative_test_probs.npy",
    "moderate_oof": CFG.v13_root / "v13_class_moderate_oof_probs.npy",
    "moderate_test": CFG.v13_root / "v13_class_moderate_test_probs.npy",
    "meta": CFG.v13_root / "aligned_meta.parquet",
    "deltas": CFG.v13_root / "outer_user_class_deltas.csv",
    "test_csv": CFG.test_csv,
}
missing = [str(path) for path in paths.values() if not path.exists()]
if missing:
    raise FileNotFoundError("Missing V14 input(s):\n" + "\n".join(missing))

meta = pd.read_parquet(paths["meta"]).reset_index(drop=True)
required_meta = {"clip_id", "user", "action_id"}
if not required_meta.issubset(meta.columns):
    raise ValueError(f"aligned_meta.parquet missing {required_meta - set(meta.columns)}")
if len(meta) != CFG.expected_oof_rows or meta.clip_id.astype(str).duplicated().any():
    raise ValueError(f"Expected {CFG.expected_oof_rows} unique OOF rows, got {len(meta)}")
if tuple(sorted(meta.user.astype(str).unique())) != tuple(sorted(CFG.expected_users)):
    raise ValueError(f"Unexpected metadata users: {sorted(meta.user.astype(str).unique())}")
y = meta.action_id.to_numpy(np.int64)
if not np.isin(y, LABEL_VALUES).all():
    raise ValueError("OOF action_id contains a value outside 0..39")

conservative_oof = normalize_probs(np.load(paths["conservative_oof"]), "V13 conservative OOF", CFG.expected_oof_rows)
moderate_oof = normalize_probs(np.load(paths["moderate_oof"]), "V13 moderate OOF", CFG.expected_oof_rows)
conservative_test = normalize_probs(np.load(paths["conservative_test"]), "V13 conservative test", CFG.expected_test_rows)
moderate_test = normalize_probs(np.load(paths["moderate_test"]), "V13 moderate test", CFG.expected_test_rows)
test_csv = pd.read_csv(paths["test_csv"])
if len(test_csv) != CFG.expected_test_rows or "path" not in test_csv.columns:
    raise ValueError("test.csv must contain exactly 405 rows and a path column")

delta_table = pd.read_csv(paths["deltas"])
stable_classes = discover_stable_saturated_classes(delta_table)
if stable_classes != CFG.expected_stable_classes:
    raise ValueError(f"Stable saturated classes changed: found {stable_classes}, expected {CFG.expected_stable_classes}")

primary_oof = selective_extrapolate(conservative_oof, moderate_oof, stable_classes, CFG.primary_scale)
primary_test = selective_extrapolate(conservative_test, moderate_test, stable_classes, CFG.primary_scale)
exploratory_oof = selective_extrapolate(conservative_oof, moderate_oof, stable_classes, CFG.exploratory_scale)
exploratory_test = selective_extrapolate(conservative_test, moderate_test, stable_classes, CFG.exploratory_scale)

# Rows on which V13 conservative and moderate are exactly equal have zero CLR
# direction and must remain unchanged (this includes every missing-Thermal row).
zero_direction_oof = np.all(np.isclose(centered_log_probs(conservative_oof), centered_log_probs(moderate_oof), atol=1e-10), axis=1)
zero_direction_test = np.all(np.isclose(centered_log_probs(conservative_test), centered_log_probs(moderate_test), atol=1e-10), axis=1)
if not np.allclose(primary_oof[zero_direction_oof], moderate_oof[zero_direction_oof], atol=1e-12):
    raise RuntimeError("Zero-direction OOF rows changed")
if not np.allclose(primary_test[zero_direction_test], moderate_test[zero_direction_test], atol=1e-12):
    raise RuntimeError("Zero-direction test rows changed")

moderate_test_pred = moderate_test.argmax(axis=1)
primary_test_pred = primary_test.argmax(axis=1)
exploratory_test_pred = exploratory_test.argmax(axis=1)
primary_changes = int((primary_test_pred != moderate_test_pred).sum())
exploratory_changes = int((exploratory_test_pred != moderate_test_pred).sum())
if primary_changes != CFG.expected_primary_test_changes:
    raise RuntimeError(f"Artifact-integrity guard: primary changed {primary_changes} labels, expected {CFG.expected_primary_test_changes}")
if exploratory_changes != CFG.expected_exploratory_test_changes:
    raise RuntimeError(f"Artifact-integrity guard: exploratory changed {exploratory_changes} labels, expected {CFG.expected_exploratory_test_changes}")

scores = pd.DataFrame([
    score("V13_moderate_077611_reference", moderate_oof, y, moderate_oof),
    score("V14_stable25_primary", primary_oof, y, moderate_oof),
    score("V14_stable30_exploratory", exploratory_oof, y, moderate_oof),
])
user_rows = []
for user in sorted(meta.user.astype(str).unique()):
    ids = np.flatnonzero(meta.user.astype(str).to_numpy() == user)
    for name, p in (("V13_moderate", moderate_oof), ("V14_stable25", primary_oof),
                    ("V14_stable30", exploratory_oof)):
        user_rows.append({
            "user": user,
            "method": name,
            "n": int(len(ids)),
            "correct": int((p[ids].argmax(1) == y[ids]).sum()),
            "accuracy": float((p[ids].argmax(1) == y[ids]).mean()),
            "nll": nll(p[ids], y[ids]),
        })
user_report = pd.DataFrame(user_rows)

change_rows = []
for name, p in (("stable25", primary_test), ("stable30", exploratory_test)):
    pred = p.argmax(axis=1)
    for row in np.flatnonzero(pred != moderate_test_pred):
        change_rows.append({
            "candidate": name,
            "test_row": int(row),
            "path": str(test_csv.iloc[row].path),
            "v13_prediction": int(moderate_test_pred[row]),
            "v14_prediction": int(pred[row]),
            "v13_probability_old_class": float(moderate_test[row, moderate_test_pred[row]]),
            "v14_probability_new_class": float(p[row, pred[row]]),
        })
change_report = pd.DataFrame(change_rows)

primary_path = CFG.work_root / "submission.csv"
exploratory_path = CFG.work_root / "submission_v14_stable30.csv"
reference_path = CFG.work_root / "submission_v13_077611_reference.csv"
primary_submission = write_submission(primary_test, primary_path, test_csv)
write_submission(exploratory_test, exploratory_path, test_csv)
write_submission(moderate_test, reference_path, test_csv)

np.save(CFG.out_dir / "v14_stable25_oof_probs.npy", primary_oof.astype(np.float32))
np.save(CFG.out_dir / "v14_stable25_test_probs.npy", primary_test.astype(np.float32))
np.save(CFG.out_dir / "v14_stable30_oof_probs.npy", exploratory_oof.astype(np.float32))
np.save(CFG.out_dir / "v14_stable30_test_probs.npy", exploratory_test.astype(np.float32))
meta.to_parquet(CFG.out_dir / "aligned_meta.parquet", index=False)
scores.to_csv(CFG.out_dir / "fusion_scores.csv", index=False)
user_report.to_csv(CFG.out_dir / "per_user_scores.csv", index=False)
change_report.to_csv(CFG.out_dir / "test_prediction_changes.csv", index=False)

summary = {
    "method": "V14 selective extrapolation of V13 stable saturated classes",
    "v13_public_score_user_reported": 0.77611,
    "stable_saturated_classes": list(stable_classes),
    "primary_scale": CFG.primary_scale,
    "exploratory_scale": CFG.exploratory_scale,
    "v13_oof_correct": int((moderate_oof.argmax(1) == y).sum()),
    "primary_oof_correct": int((primary_oof.argmax(1) == y).sum()),
    "exploratory_oof_correct": int((exploratory_oof.argmax(1) == y).sum()),
    "v13_oof_nll": nll(moderate_oof, y),
    "primary_oof_nll": nll(primary_oof, y),
    "exploratory_oof_nll": nll(exploratory_oof, y),
    "primary_test_changes_vs_v13": primary_changes,
    "exploratory_test_changes_vs_v13": exploratory_changes,
    "exploratory_changes_vs_primary": int((exploratory_test_pred != primary_test_pred).sum()),
    "zero_direction_test_rows": int(zero_direction_test.sum()),
    "runtime_seconds": float(time.perf_counter() - start),
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "primary_submission": str(primary_path),
    "primary_sha256": sha256(primary_path),
    "exploratory_sha256": sha256(exploratory_path),
    "reference_sha256": sha256(reference_path),
    "submission_order": [
        "Submit submission.csv first.",
        "Evaluate submission_v14_stable30.csv only after reviewing the primary result.",
    ],
    "note": "No labels beyond the saved aligned OOF labels and no neural inference are used.",
}
(CFG.out_dir / "fusion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("V14 STABLE SATURATED CLASSES:", stable_classes)
print("\nOOF SCORES")
print(scores.to_string(index=False))
print("\nPER-USER SCORES")
print(user_report.to_string(index=False))
print("\nTEST PREDICTION CHANGES")
print(change_report.to_string(index=False))
print("\nV14 FINAL SUMMARY")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print("\nWROTE")
print("/kaggle/working/submission.csv                         # V14 stable classes, 2.5x primary")
print("/kaggle/working/submission_v14_stable30.csv            # 3.0x exploratory; submit second only")
print("/kaggle/working/submission_v13_077611_reference.csv    # exact V13 moderate reference")
print("/kaggle/working/v14_stable_class_extrapolation/        # probabilities and audit reports")
print(primary_submission.head())
