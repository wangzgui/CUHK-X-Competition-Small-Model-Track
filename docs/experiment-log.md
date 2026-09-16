# Experiment log and lessons learned

This is a selective, evidence-bounded research log. “Public” and “private” mean competition leaderboards; they are not interchangeable with local validation. Values reported only in earlier working notes are marked approximate.

## 1. Visual branch: upstream baseline and deliberately unshared follow-ups

The visual starting point was the public Kaggle IR + Depth_Color baseline credited in [credits-and-licensing.md](credits-and-licensing.md): 16 uniformly sampled frames, 128x128, four input channels, an IR-derived fixed YOLO person crop, R(2+1)D-34 with IG-65M/Kinetics lineage, horizontal-flip TTA, and compact checkpoint storage. Public score: **0.72139**.

We do **not** release our visual-side modifications because the repository's contribution is the multimodal study, and the visual source has upstream ownership/provenance requirements. The following outcomes are retained to prevent repeated dead ends:

| Experiment | Local observation | Public result | Decision |
| --- | --- | ---: | --- |
| 32-frame cache, random 16-frame training, complementary temporal views | Small single-fold gain | 0.68159 | Rejected |
| Cross-subject clip-level MixUp | Local gain | 0.69154 | Rejected |
| Layer-wise LR + boundary validation + full-data 30 epochs | Confounded multi-change experiment | 0.68159 | Rejected as a package |
| Visual original/flip decoupling | No useful change | — | Rejected |

Interpretation: clip-level MixUp created implausible mixtures of people, backgrounds, pose, and small objects; temporal-view averaging could dilute short interactions. These are hypotheses, not causal proof.

## 2. Thermal-only progression (released)

The Thermal branch was the main original modeling contribution. It used IR-derived, clip-level fixed person crops for spatial alignment, then moved from compact R(2+1)D-18 toward MViTv2-S. Important milestones are released as source files in `src/thermal/` after running the release-preparation script.

| Stage | Main change | Reported competition evidence | Status |
| --- | --- | --- | --- |
| Early R(2+1)D-18 | Thermal-only baseline, fixed IR crop | ~0.60 | Historical baseline |
| Full 32-frame R(2+1)D-18 | Complete temporal coverage | 0.61691 | Positive early step |
| MViTv2-S transition | Kinetics-pretrained MViTv2-S, calibration augmentation, MixUp, EMA, INT8 | 0.64676 | Early main backbone |
| V8 | 64-frame cache with ordered random 32-frame sampling | ~0.661 | Positive progression |
| V9 | Mild full-span time warp | recorded experiment | Diagnostic stage |
| V11 | MaskFeat-style domain adaptation then fine-tuning | 0.680 | Positive progression |
| V16 / final Thermal line | 160-pixel MaskFeat-style continuation from V11 | **0.70149 public / 0.71580 private** | Final Thermal stream used in fusion |
| Later V17–V23 variants | grid, resolution, LPFT, progressive fine-tuning, consistency/reweighting | no confirmed final gain retained | Exploratory only |

The score-to-script mapping for the final 0.70149/0.71580 artifact must be verified against the saved Kaggle run before release. The repository preserves the full source lineage but must not label a later experimental script as the final winning Thermal checkpoint merely because it has a higher version number.

### Thermal methods that did not become the final recipe

- Layer-wise learning-rate decay plus hybrid/Transformer-aware quantization.
- Dual spatial/temporal test views, Tube Erasing, high-resolution R(2+1)D, FrameNet-only 2D, MixStyle variants, X3D-M + precise BN, and dynamic tubes.
- Later additions after V16, unless a run record demonstrates an independent gain.

These are released selectively as code only where they are useful for understanding the progression; no benchmark claim should be inferred from their filename alone.

## 3. IMU and Skeleton: useful diagnostics, not final decision makers (released)

Early effort was disproportionately concentrated on IMU and Skeleton. This was a strategic mistake for the final score, but the experiments are worth releasing because they are clean subject-disjoint baselines and reveal why low-accuracy modalities are hard to fuse safely.

### IMU

The IMU line evolved from interpolation to native timestamp sorting/padding, dual sensor/motion streams, structured sensor dropout, user balancing, timing features, energy features, and quaternion-sign continuity. The most useful output contract was full OOF/test probability export for later fusion.

- Reported single-stream quality: approximately **0.43** on the leaderboard; aligned OOF notes show a baseline-level accuracy near **0.4171**.
- Size: roughly 1–2 MB class of model, depending on the final ensemble/checkpoint format.
- Takeaway: inexpensive and potentially complementary, but its calibration and top-1 accuracy were insufficient for naive global blending.

### Skeleton

The Skeleton line used compact subject-disjoint spatio-temporal models with root-relative geometry, motion/angle features, multiscale temporal convolutions, and later audits/transfer attempts.

- Reported five-fold result: **0.4476 ± 0.0263**; reported leaderboard result approximately **0.43**.
- Takeaway: geometry is informative, but pose-estimation and cross-subject error make it unsafe to let Skeleton override the stronger streams broadly.

### Why the third-modality arbitration failed

On the 541-sample aligned set, Visual + Thermal had meaningful complementarity, and an oracle that also considered IMU/Skeleton could be stronger. But the usable decision set was tiny and only spanned three held-out subjects. A strict V20 Visual–Thermal conflict-arbitration rule had no public-board improvement (and produced no effective test-label changes in the strict submission). This is evidence against that particular low-data routing rule—not evidence that IMU or Skeleton are inherently useless.

## 4. Fusion progression (released)

| Version | Method | Public score | Interpretation |
| --- | --- | ---: | --- |
| Early VT | Temperature calibration + conservative geometric Visual/Thermal fusion | 0.73631 | Thermal has complementary signal |
| Improved VT | Stronger Thermal stream + same conservative approach | 0.74626 | Thermal quality matters |
| V11 | Calibrated Visual + V11 Thermal | 0.75140 | Continued gain |
| V12 | V16 Thermal, cross-user leave-one-user-out parameter estimate and shrinkage | 0.76600 | Safer than choosing from one fold |
| V13 Moderate | Class-level residual along calibrated V–T direction | 0.77611 | Local top-1 and public board disagreed |
| V14 Stable25 | Extrapolate only stable saturated class directions (17, 21, 37) | 0.78606 | Best early public fusion |
| Original Thermal view through V14 | Frozen V12/V13/V14 propagation using Thermal original view | **0.79106** | Best public score reported |
| Final entry | Competition final submission | private **0.77941** | Final rank: 44th private |
| V21 | Late candidate | private **0.78431** | Best private result observed |

The final protocol stayed deliberately low dimensional: temperature-calibrated log-probability fusion, cross-user shrinkage, a bounded per-class residual, then a three-class stable extrapolation. It did **not** use a high-capacity 80-dimensional stacking/MLP fusion because 541 aligned OOF rows and three users were insufficient to support it.

## 5. General lessons

1. **Use subject-disjoint OOF, not random clip splits.**
2. **Keep modality fusion conservative when OOF is small.** Complementarity oracle numbers do not provide a learnable routing policy by themselves.
3. **Treat leaderboard feedback as noisy.** The public/private shake confirms that small leaderboard changes were not stable estimates.
4. **Publish failures.** The negative experiments here are part of the result, not discarded history.

