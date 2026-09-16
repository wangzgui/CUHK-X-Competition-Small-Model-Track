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
from scipy.stats import beta as beta_distribution


@dataclass(frozen=True)
class Config:
    v14_root: Path = Path("/kaggle/input/models/zhuowamg/v15model/pytorch/default/2/v14_stable_class_extrapolation")
    v12_root: Path = Path("/kaggle/input/data-ooo/v12_visual_thermal_v16_fusion")
    visual_val_probs: Path = Path("/kaggle/input/datasets/zhuowamg/ir-depth-color/outputs/val_probs_fold0.npy")
    visual_val_meta: Path = Path("/kaggle/input/datasets/zhuowamg/ir-depth-color/outputs/val_pred_fold0.parquet")
    visual_test_probs: Path = Path("/kaggle/input/datasets/zhuowamg/newensembledata/visual_test_probs.npy")
    thermal_test_probs: Path = Path("/kaggle/input/datasets/zhuowamg/thermalv16/thermal_test_probs (2).npy")
    test_csv: Path = Path("/kaggle/input/datasets/zhuowamg/cuhksmall/CUHK-X_dataset/Small-Model-Track/Testing/test_file/test.csv")
    test_root: Path = Path("/kaggle/input/datasets/zhuowamg/cuhksmall/CUHK-X_dataset/Small-Model-Track/Testing/data/small_model_track_test/small_model_track_test")
    work_root: Path = Path("/kaggle/working")

    n_classes: int = 40
    expected_oof: int = 541
    expected_test: int = 405
    expected_users: tuple[str, ...] = ("user9", "user16", "user18")
    expected_missing_thermal: int = 10
    visual_temperature: float = 2.0
    thermal_temperature: float = 0.75
    beta_prior_strength: float = 8.0
    decision_logit_margin: float = 0.02

    @property
    def out_dir(self) -> Path:
        return self.work_root / "v15_bayesian_disagreement_arbitration"


CFG = Config()
EPS = 1e-12
LABEL_VALUES = np.arange(CFG.n_classes, dtype=np.int64)


@dataclass(frozen=True)
class RuleConfig:
    name: str
    min_support_per_user: int
    min_total_support: int
    posterior_quantile: float
    max_anchor_margin: float
    min_js: float
    chosen_margin_tolerance: float


STRICT = RuleConfig("strict", 2, 6, 0.10, 0.25, 0.020, 0.06)
RELAXED = RuleConfig("relaxed", 1, 4, 0.20, 0.35, 0.010, 0.12)


def normalize_probs(values, name: str, rows: int | None = None) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != CFG.n_classes:
        raise ValueError(f"{name}: expected (N,{CFG.n_classes}), got {p.shape}")
    if rows is not None and len(p) != rows:
        raise ValueError(f"{name}: expected {rows} rows, got {len(p)}")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError(f"{name}: contains NaN/Inf or negative values")
    mass = p.sum(axis=1, keepdims=True)
    if (mass <= 0).any():
        raise ValueError(f"{name}: contains zero probability mass")
    return p / mass


def temperature_scale(p: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(p, EPS, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    q = np.exp(logits)
    return q / q.sum(axis=1, keepdims=True)


def top_margin(p: np.ndarray) -> np.ndarray:
    top2 = np.partition(p, -2, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * (np.log(np.clip(p, EPS, 1.0)) - np.log(np.clip(m, EPS, 1.0))), axis=1) + \
           0.5 * np.sum(q * (np.log(np.clip(q, EPS, 1.0)) - np.log(np.clip(m, EPS, 1.0))), axis=1)


def modality_features(visual: np.ndarray, thermal: np.ndarray, anchor: np.ndarray) -> dict[str, np.ndarray]:
    qv = temperature_scale(visual, CFG.visual_temperature)
    qt = temperature_scale(thermal, CFG.thermal_temperature)
    return {
        "qv": qv,
        "qt": qt,
        "visual_pred": qv.argmax(axis=1),
        "thermal_pred": qt.argmax(axis=1),
        "visual_margin": top_margin(qv),
        "thermal_margin": top_margin(qt),
        "anchor_margin": top_margin(anchor),
        "js": js_divergence(qv, qt),
    }


def build_pair_rules(train_ids: np.ndarray, features: dict[str, np.ndarray], y: np.ndarray,
                     users: np.ndarray, rule_cfg: RuleConfig) -> tuple[dict[tuple[int, int], dict], list[dict]]:
    vp, tp = features["visual_pred"], features["thermal_pred"]
    ids = train_ids[vp[train_ids] != tp[train_ids]]
    visual_wins = y[ids] == vp[ids]
    thermal_wins = y[ids] == tp[ids]
    informative = visual_wins | thermal_wins
    global_t = int(thermal_wins[informative].sum())
    global_v = int(visual_wins[informative].sum())
    global_rate = (global_t + 1.0) / (global_t + global_v + 2.0)
    prior_a = CFG.beta_prior_strength * global_rate
    prior_b = CFG.beta_prior_strength * (1.0 - global_rate)

    train_users = sorted(np.unique(users[train_ids]))
    if len(train_users) != 2:
        raise ValueError(f"Outer rule training expected two users, got {train_users}")
    rules, audit = {}, []
    pairs = sorted(set(zip(vp[ids].tolist(), tp[ids].tolist())))
    for visual_class, thermal_class in pairs:
        pair_ids = ids[(vp[ids] == visual_class) & (tp[ids] == thermal_class)]
        per_user = []
        for user in train_users:
            user_ids = pair_ids[users[pair_ids] == user]
            v_count = int((y[user_ids] == visual_class).sum())
            t_count = int((y[user_ids] == thermal_class).sum())
            per_user.append((v_count, t_count, v_count + t_count))
        total_v = sum(item[0] for item in per_user)
        total_t = sum(item[1] for item in per_user)
        total = total_v + total_t
        directions = [int(np.sign(t_count - v_count)) for v_count, t_count, _ in per_user]
        support_ok = (total >= rule_cfg.min_total_support and
                      all(item[2] >= rule_cfg.min_support_per_user for item in per_user))
        direction_ok = directions[0] != 0 and directions[0] == directions[1]
        posterior_a, posterior_b = prior_a + total_t, prior_b + total_v
        lower = float(beta_distribution.ppf(rule_cfg.posterior_quantile, posterior_a, posterior_b))
        upper = float(beta_distribution.ppf(1.0 - rule_cfg.posterior_quantile, posterior_a, posterior_b))
        choice = None
        if support_ok and direction_ok and directions[0] > 0 and lower > 0.5:
            choice = "thermal"
        elif support_ok and direction_ok and directions[0] < 0 and upper < 0.5:
            choice = "visual"
        record = {
            "visual_class": int(visual_class),
            "thermal_class": int(thermal_class),
            "choice": choice or "none",
            "total_visual_wins": total_v,
            "total_thermal_wins": total_t,
            "total_informative": total,
            "posterior_thermal_mean": float(posterior_a / (posterior_a + posterior_b)),
            "posterior_lower": lower,
            "posterior_upper": upper,
            "global_thermal_win_rate": float(global_rate),
            "train_users": ",".join(train_users),
        }
        for user, (v_count, t_count, support) in zip(train_users, per_user):
            record[f"{user}_visual_wins"] = v_count
            record[f"{user}_thermal_wins"] = t_count
            record[f"{user}_support"] = support
        audit.append(record)
        if choice is not None:
            rules[(int(visual_class), int(thermal_class))] = record
    return rules, audit


def apply_rule_votes(ids: np.ndarray, features: dict[str, np.ndarray], rules: dict,
                     rule_cfg: RuleConfig, available: np.ndarray) -> np.ndarray:
    votes = np.full(len(ids), -1, dtype=np.int64)
    for local_row, row in enumerate(ids):
        if not available[row]:
            continue
        visual_class = int(features["visual_pred"][row])
        thermal_class = int(features["thermal_pred"][row])
        rule = rules.get((visual_class, thermal_class))
        if rule is None or features["anchor_margin"][row] > rule_cfg.max_anchor_margin or features["js"][row] < rule_cfg.min_js:
            continue
        if rule["choice"] == "thermal":
            chosen, chosen_margin, other_margin = thermal_class, features["thermal_margin"][row], features["visual_margin"][row]
        else:
            chosen, chosen_margin, other_margin = visual_class, features["visual_margin"][row], features["thermal_margin"][row]
        if chosen_margin + rule_cfg.chosen_margin_tolerance < other_margin:
            continue
        votes[local_row] = chosen
    return votes


def minimal_override(anchor: np.ndarray, chosen_labels: np.ndarray) -> np.ndarray:
    output = anchor.copy()
    for row, chosen in enumerate(chosen_labels):
        if chosen < 0 or chosen == int(anchor[row].argmax()):
            continue
        logits = np.log(np.clip(anchor[row], EPS, 1.0))
        other_max = np.max(np.delete(logits, chosen))
        logits[chosen] = max(logits[chosen], other_max + CFG.decision_logit_margin)
        logits -= logits.max()
        output[row] = np.exp(logits) / np.exp(logits).sum()
    return output


def score(name: str, p: np.ndarray, y: np.ndarray, anchor: np.ndarray) -> dict:
    pred, base = p.argmax(1), anchor.argmax(1)
    correct, base_correct = pred == y, base == y
    return {
        "method": name,
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0)).mean()),
        "changes_vs_v14": int((pred != base).sum()),
        "rescued_vs_v14": int((~base_correct & correct).sum()),
        "harmed_vs_v14": int((base_correct & ~correct).sum()),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_submission(p: np.ndarray, path: Path, test_csv: pd.DataFrame) -> pd.DataFrame:
    submission = test_csv[["path"]].copy()
    submission["prediction"] = LABEL_VALUES[p.argmax(1)]
    if (len(submission) != CFG.expected_test or submission.prediction.isna().any() or
            not submission.prediction.between(0, CFG.n_classes - 1).all()):
        raise ValueError(f"Submission contract failed: {path}")
    submission.to_csv(path, index=False)
    return submission


start = time.perf_counter()
CFG.out_dir.mkdir(parents=True, exist_ok=True)
required = {
    "v14_oof": CFG.v14_root / "v14_stable25_oof_probs.npy",
    "v14_test": CFG.v14_root / "v14_stable25_test_probs.npy",
    "v14_meta": CFG.v14_root / "aligned_meta.parquet",
    "v16_oof": CFG.v12_root / "v16_aligned_probs.npy",
    "v12_meta": CFG.v12_root / "aligned_meta.parquet",
    "visual_val_probs": CFG.visual_val_probs,
    "visual_val_meta": CFG.visual_val_meta,
    "visual_test_probs": CFG.visual_test_probs,
    "thermal_test_probs": CFG.thermal_test_probs,
    "test_csv": CFG.test_csv,
    "test_root": CFG.test_root,
}
missing = [str(path) for path in required.values() if not path.exists()]
if missing:
    raise FileNotFoundError("Missing V15 input(s):\n" + "\n".join(missing))

meta = pd.read_parquet(required["v14_meta"]).reset_index(drop=True)
if not {"clip_id", "user", "action_id"}.issubset(meta.columns):
    raise ValueError("V14 aligned_meta must contain clip_id, user, action_id")
if len(meta) != CFG.expected_oof or meta.clip_id.astype(str).duplicated().any():
    raise ValueError("V14 aligned_meta must contain 541 unique clips")
users = meta.user.astype(str).to_numpy()
y = meta.action_id.to_numpy(np.int64)
if tuple(sorted(np.unique(users))) != tuple(sorted(CFG.expected_users)) or not np.isin(y, LABEL_VALUES).all():
    raise ValueError("Unexpected V14 users or labels")

anchor_oof = normalize_probs(np.load(required["v14_oof"]), "V14 OOF", CFG.expected_oof)
anchor_test = normalize_probs(np.load(required["v14_test"]), "V14 test", CFG.expected_test)

# Align Visual OOF by clip ID.
visual_meta = pd.read_parquet(required["visual_val_meta"]).reset_index(drop=True)
visual_all = normalize_probs(np.load(required["visual_val_probs"]), "Visual validation")
if len(visual_meta) != len(visual_all) or "clip_id" not in visual_meta:
    raise ValueError("Visual validation metadata/probability mismatch")
visual_rows = {str(clip_id): row for row, clip_id in enumerate(visual_meta.clip_id)}
if len(visual_rows) != len(visual_meta) or any(str(cid) not in visual_rows for cid in meta.clip_id):
    raise ValueError("Visual validation clip IDs do not cover V14 aligned metadata")
visual_order = [visual_rows[str(cid)] for cid in meta.clip_id]
if "action_id" in visual_meta and not np.array_equal(visual_meta.iloc[visual_order].action_id.to_numpy(np.int64), y):
    raise ValueError("Visual and V14 labels disagree after clip-ID alignment")
visual_oof = visual_all[visual_order]

# Align V16 OOF using the V12 metadata rather than assuming row order.
v12_meta = pd.read_parquet(required["v12_meta"]).reset_index(drop=True)
thermal_all = normalize_probs(np.load(required["v16_oof"]), "V16 aligned validation")
if len(v12_meta) != len(thermal_all) or "clip_id" not in v12_meta:
    raise ValueError("V12 Thermal metadata/probability mismatch")
thermal_rows = {str(clip_id): row for row, clip_id in enumerate(v12_meta.clip_id)}
if len(thermal_rows) != len(v12_meta) or any(str(cid) not in thermal_rows for cid in meta.clip_id):
    raise ValueError("V16 validation clip IDs do not cover V14 aligned metadata")
thermal_order = [thermal_rows[str(cid)] for cid in meta.clip_id]
if "action_id" in v12_meta and not np.array_equal(v12_meta.iloc[thermal_order].action_id.to_numpy(np.int64), y):
    raise ValueError("Thermal and V14 labels disagree after clip-ID alignment")
thermal_oof = thermal_all[thermal_order]

test_csv = pd.read_csv(required["test_csv"])
if len(test_csv) != CFG.expected_test or "path" not in test_csv:
    raise ValueError("test.csv must contain exactly 405 rows and a path column")
visual_test = normalize_probs(np.load(required["visual_test_probs"]), "Visual test", CFG.expected_test)
thermal_test = normalize_probs(np.load(required["thermal_test_probs"]), "V16 Thermal test", CFG.expected_test)
image_suffixes = {".jpg", ".jpeg", ".png"}
thermal_available = []
for value in test_csv.path.astype(str):
    clip_id = value.strip("/\\").replace("\\", "/").split("/")[-1]
    folder = CFG.test_root / clip_id / "Thermal"
    thermal_available.append(folder.is_dir() and any(p.suffix.lower() in image_suffixes for p in folder.iterdir()))
thermal_available = np.asarray(thermal_available, dtype=bool)
if int((~thermal_available).sum()) != CFG.expected_missing_thermal:
    raise ValueError(f"Expected 10 missing-Thermal test clips, found {int((~thermal_available).sum())}")

oof_features = modality_features(visual_oof, thermal_oof, anchor_oof)
test_features = modality_features(visual_test, thermal_test, anchor_test)
oof_available = np.ones(CFG.expected_oof, dtype=bool)


def run_protocol(rule_cfg: RuleConfig):
    oof_choices = np.full(CFG.expected_oof, -1, dtype=np.int64)
    test_vote_matrix = np.full((len(CFG.expected_users), CFG.expected_test), -1, dtype=np.int64)
    rule_rows = []
    for fold, held_user in enumerate(sorted(CFG.expected_users)):
        train_ids = np.flatnonzero(users != held_user)
        held_ids = np.flatnonzero(users == held_user)
        rules, audit = build_pair_rules(train_ids, oof_features, y, users, rule_cfg)
        oof_choices[held_ids] = apply_rule_votes(held_ids, oof_features, rules, rule_cfg, oof_available)
        test_ids = np.arange(CFG.expected_test)
        test_vote_matrix[fold] = apply_rule_votes(test_ids, test_features, rules, rule_cfg, thermal_available)
        for row in audit:
            row.update({"protocol": rule_cfg.name, "held_user": held_user, "accepted_rule": row["choice"] != "none"})
            rule_rows.append(row)

    test_choices = np.full(CFG.expected_test, -1, dtype=np.int64)
    vote_rows = []
    for row in range(CFG.expected_test):
        valid = test_vote_matrix[:, row]
        valid = valid[valid >= 0]
        if len(valid):
            labels, counts = np.unique(valid, return_counts=True)
            winner = int(labels[np.argmax(counts)])
            votes = int(counts.max())
            if votes >= 2:
                test_choices[row] = winner
        vote_rows.append({
            "protocol": rule_cfg.name,
            "test_row": row,
            "path": str(test_csv.iloc[row].path),
            "visual_prediction": int(test_features["visual_pred"][row]),
            "thermal_prediction": int(test_features["thermal_pred"][row]),
            "v14_prediction": int(anchor_test[row].argmax()),
            "fold_votes": ",".join(map(str, test_vote_matrix[:, row].tolist())),
            "majority_choice": int(test_choices[row]),
        })
    oof_probs = minimal_override(anchor_oof, oof_choices)
    test_probs = minimal_override(anchor_test, test_choices)
    return oof_probs, test_probs, rule_rows, vote_rows


strict_oof, strict_test, strict_rules, strict_votes = run_protocol(STRICT)
relaxed_oof, relaxed_test, relaxed_rules, relaxed_votes = run_protocol(RELAXED)
scores = pd.DataFrame([
    score("V14_078606_reference", anchor_oof, y, anchor_oof),
    score("V15_bayesian_strict", strict_oof, y, anchor_oof),
    score("V15_bayesian_relaxed", relaxed_oof, y, anchor_oof),
])
per_user_rows = []
for user in sorted(np.unique(users)):
    ids = np.flatnonzero(users == user)
    for name, p in (("V14_anchor", anchor_oof), ("V15_strict", strict_oof), ("V15_relaxed", relaxed_oof)):
        row_score = score(name, p[ids], y[ids], anchor_oof[ids])
        row_score.update({"user": user, "n": int(len(ids))})
        per_user_rows.append(row_score)
per_user_scores = pd.DataFrame(per_user_rows)

oof_change_rows = []
for name, p in (("strict", strict_oof), ("relaxed", relaxed_oof)):
    pred, anchor_pred_oof = p.argmax(1), anchor_oof.argmax(1)
    for row in np.flatnonzero(pred != anchor_pred_oof):
        oof_change_rows.append({
            "candidate": name,
            "clip_id": str(meta.iloc[row].clip_id),
            "user": str(users[row]),
            "action_id": int(y[row]),
            "visual_prediction": int(oof_features["visual_pred"][row]),
            "thermal_prediction": int(oof_features["thermal_pred"][row]),
            "v14_prediction": int(anchor_pred_oof[row]),
            "v15_prediction": int(pred[row]),
            "rescued": bool(anchor_pred_oof[row] != y[row] and pred[row] == y[row]),
            "harmed": bool(anchor_pred_oof[row] == y[row] and pred[row] != y[row]),
        })
oof_change_report = pd.DataFrame(oof_change_rows)

strict_pred, relaxed_pred, anchor_pred = strict_test.argmax(1), relaxed_test.argmax(1), anchor_test.argmax(1)
changes = []
for name, p, pred in (("strict", strict_test, strict_pred), ("relaxed", relaxed_test, relaxed_pred)):
    for row in np.flatnonzero(pred != anchor_pred):
        changes.append({
            "candidate": name,
            "test_row": int(row),
            "path": str(test_csv.iloc[row].path),
            "visual_prediction": int(test_features["visual_pred"][row]),
            "thermal_prediction": int(test_features["thermal_pred"][row]),
            "v14_prediction": int(anchor_pred[row]),
            "v15_prediction": int(pred[row]),
            "visual_margin": float(test_features["visual_margin"][row]),
            "thermal_margin": float(test_features["thermal_margin"][row]),
            "v14_margin": float(test_features["anchor_margin"][row]),
            "js_divergence": float(test_features["js"][row]),
        })
change_report = pd.DataFrame(changes)

primary_path = CFG.work_root / "submission.csv"
relaxed_path = CFG.work_root / "submission_v15_relaxed.csv"
reference_path = CFG.work_root / "submission_v14_078606_reference.csv"
primary_submission = write_submission(strict_test, primary_path, test_csv)
write_submission(relaxed_test, relaxed_path, test_csv)
write_submission(anchor_test, reference_path, test_csv)

np.save(CFG.out_dir / "v15_strict_oof_probs.npy", strict_oof.astype(np.float32))
np.save(CFG.out_dir / "v15_strict_test_probs.npy", strict_test.astype(np.float32))
np.save(CFG.out_dir / "v15_relaxed_oof_probs.npy", relaxed_oof.astype(np.float32))
np.save(CFG.out_dir / "v15_relaxed_test_probs.npy", relaxed_test.astype(np.float32))
meta.to_parquet(CFG.out_dir / "aligned_meta.parquet", index=False)
scores.to_csv(CFG.out_dir / "fusion_scores.csv", index=False)
per_user_scores.to_csv(CFG.out_dir / "per_user_scores.csv", index=False)
oof_change_report.to_csv(CFG.out_dir / "oof_prediction_changes.csv", index=False)
pd.DataFrame(strict_rules + relaxed_rules).to_csv(CFG.out_dir / "outer_pair_rule_audit.csv", index=False)
pd.DataFrame(strict_votes + relaxed_votes).to_csv(CFG.out_dir / "test_vote_audit.csv", index=False)
change_report.to_csv(CFG.out_dir / "test_prediction_changes.csv", index=False)

summary = {
    "method": "V15 cross-user Bayesian arbitration for Visual/Thermal disagreements",
    "public_anchor_score_user_reported": 0.78606,
    "public_or_private": "public leaderboard",
    "remaining_submissions_before_v15": 2,
    "strict_oof_correct": int((strict_oof.argmax(1) == y).sum()),
    "relaxed_oof_correct": int((relaxed_oof.argmax(1) == y).sum()),
    "anchor_oof_correct": int((anchor_oof.argmax(1) == y).sum()),
    "strict_test_changes": int((strict_pred != anchor_pred).sum()),
    "relaxed_test_changes": int((relaxed_pred != anchor_pred).sum()),
    "strict_vs_relaxed_test_changes": int((strict_pred != relaxed_pred).sum()),
    "strict_accepted_outer_rules": int(sum(row["accepted_rule"] for row in strict_rules)),
    "relaxed_accepted_outer_rules": int(sum(row["accepted_rule"] for row in relaxed_rules)),
    "missing_thermal_test": int((~thermal_available).sum()),
    "runtime_seconds": float(time.perf_counter() - start),
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "primary_submission": str(primary_path),
    "primary_sha256": sha256(primary_path),
    "relaxed_sha256": sha256(relaxed_path),
    "reference_sha256": sha256(reference_path),
    "decision": "Do not submit automatically. Inspect test_prediction_changes.csv and the printed diagnostics first.",
}
(CFG.out_dir / "fusion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("V15 OOF SCORES")
print(scores.to_string(index=False))
print("\nV15 PER-USER OOF SCORES")
print(per_user_scores.to_string(index=False))
print("\nV15 TEST CHANGES")
print(change_report.to_string(index=False) if len(change_report) else "No candidate changes the V14 labels.")
print("\nV15 FINAL SUMMARY")
print(json.dumps(summary, indent=2, ensure_ascii=False))
if summary["strict_test_changes"] == 0:
    print("WARNING: submission.csv is label-identical to V14; do not spend a submission on it.")
if summary["relaxed_test_changes"] == 0:
    print("WARNING: submission_v15_relaxed.csv is label-identical to V14; do not submit it.")
if summary["strict_test_changes"] > 12 or summary["relaxed_test_changes"] > 20:
    print("WARNING: candidate changes are too numerous for the remaining submission budget; return diagnostics before submission.")
print("\nWROTE")
print("/kaggle/working/submission.csv                       # strict candidate; inspect before submitting")
print("/kaggle/working/submission_v15_relaxed.csv           # second candidate; inspect before submitting")
print("/kaggle/working/submission_v14_078606_reference.csv  # exact V14 champion reference")
print("/kaggle/working/v15_bayesian_disagreement_arbitration/")
print(primary_submission.head())
