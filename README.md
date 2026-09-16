# CUHK-X Small Model Track: cross-subject multimodal HAR

This repository is a research release for a 40-class, cross-subject human-action-recognition entry in the CUHK-X Small Model Track.

The final entry ranked **41st on the public leaderboard** and **44th on the private leaderboard**, with a final private score of **0.77941**. Its best private-board result during the competition was **0.78431** (V21). These are competition results, not estimates from local validation.

The central lesson of this project is simple: with unseen test subjects, subject-disjoint evidence matters far more than random clip splits; and a weaker Thermal stream can substantially complement a stronger IR + Depth_Color visual stream.

> **Important attribution.** The IR + Depth_Color visual starting point is not our original model. It is derived from the Kaggle public notebook by **`welshonionman`**, *[LB0.667] baseline with YOLO person crop*. That public line of work supplies the 4-channel R(2+1)D visual pipeline and IR-based person crop idea. This repository documents our downstream Thermal, IMU, Skeleton, and fusion research, and does not redistribute that upstream visual baseline or its checkpoints. See [Credits and licensing](docs/credits-and-licensing.md).

## Results at a glance

| Component / stage | Public score | Private score | Role |
| --- | ---: | ---: | --- |
| Upstream IR + Depth_Color visual baseline | 0.72139 | — | Public Kaggle starting point; not claimed as original work |
| Final Thermal-only stream used for fusion | 0.70149 | 0.71580 | Our main complementary stream |
| V14 Stable25 Visual + Thermal fusion | 0.78606 | — | Best early public result |
| Thermal-original Visual + Thermal view | 0.79106 | — | Best public result reported during the competition |
| Final competition entry | — | **0.77941** | 44th private leaderboard |
| V21 late candidate | — | **0.78431** | Best private result observed |

`—` means that the number was not retained as a verified scoreboard record in this release; it must not be read as zero or as a claim of equality across boards.

## What is released

- A curated, executable research trail for the **Thermal-only** branch: R(2+1)D baseline, MViTv2-S transition, temporal sampling studies, MaskFeat-style adaptation, 160-pixel fine-tuning, and later controlled variants.
- The **IMU** and **Skeleton** subject-disjoint baselines and their main improvements, including OOF-export contracts for downstream analysis.
- The Visual–Thermal fusion lineage: calibrated geometric fusion, cross-user shrinkage, class-level residuals, and stable-class extrapolation.
- Honest negative results: global IMU/Skeleton blending, conflict arbitration, visual temporal resampling, visual MixUp, and several visual fine-tuning changes.

The detailed chronology, evidence boundary, and source-file mapping are in [the experiment log](docs/experiment-log.md) and [the release manifest](docs/release-manifest.md).

## What is deliberately not released

This repository must not contain any CUHK-X competition data, extracted sensor frames, memmap caches, test predictions, OOF probability arrays, submission CSVs, trained checkpoints, Kaggle secrets, or third-party assets whose redistribution rights are unclear. The `.gitignore` enforces these exclusions.

You must obtain the original data through the competition/organizer channels and comply with their terms. Paths in the scripts are Kaggle-oriented examples and need to be adapted to your own mounted dataset names.

## Repository layout

```text
src/
  thermal/    curated training milestones for the Thermal stream
  imu/        lightweight five-sensor Transformer progression
  skeleton/   compact skeleton recognition progression
  fusion/     probability-level Visual–Thermal and diagnostic fusion
docs/
  experiment-log.md         verified score history and negative results
  release-manifest.md       exactly what to copy into each source folder
  credits-and-licensing.md  upstream and third-party attribution
  reproducibility.md        data, environment, and validation constraints
scripts/
  prepare_release.ps1       copies only reviewed source files into this tree
```

## Reproducibility notes

The code was developed as Kaggle one-cell scripts/notebooks. It expects CUDA, PyTorch, torchvision, NumPy, pandas, scikit-learn, Pillow, tqdm, and—where person crops are reproduced—Ultralytics. Individual scripts install missing packages in Kaggle; for a controlled local environment, pin versions first.

The competition's local evidence was intentionally limited: the key aligned fusion set had 541 samples from three held-out users. Several ideas improved local top-1 but degraded on the public board, while class-level shrinkage sometimes did the reverse. The tables therefore distinguish measured leaderboard results from local diagnostics and do not claim that any local score generalizes.

## Quick release procedure

1. Read [the manifest](docs/release-manifest.md), then run `scripts/prepare_release.ps1` from this directory. It copies only the selected `.py` sources and never copies data, weights, outputs, or notebooks.
2. Inspect `git status --ignored` and confirm no ignored artifact was force-added.
3. Verify that the upstream visual-notebook attribution in `docs/credits-and-licensing.md` remains intact when making later edits.
4. The repository's existing MIT license applies only to the original source and documentation contributed here. It does not license CUHK-X data, checkpoints, or third-party components; see [Credits and licensing](docs/credits-and-licensing.md).
5. Create an empty GitHub repository, initialize this directory, review the diff, and push.

## Citation

If this release is useful, please cite the CUHK-X dataset/competition paper and the CUHK AIoT Lab according to the organizer's requested form, as well as the upstream Kaggle visual baseline above. A project-specific citation file can be added after the GitHub repository URL and author name are finalized.
