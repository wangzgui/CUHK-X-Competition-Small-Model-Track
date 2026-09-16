# Release manifest

Run the supplied preparation script to copy source only. The release deliberately uses `.py` as the canonical form: the source is reviewable in GitHub and avoids storing duplicated notebook outputs. Keep the original one-cell notebooks privately unless you separately clear their outputs and third-party assets.

## Thermal (`src/thermal`)

| Destination filename | Working-tree source | Why include |
| --- | --- | --- |
| `01_r2plus1d18_baseline.py` | `thermal_r2plus1d18.py` | Initial compact Thermal stream |
| `02_mvitv2s_full32.py` | `thermal_full32_calibration_mixup_mvitv2s_boundary_submission.py` | R(2+1)D to MViTv2-S transition |
| `03_v8_ordered_temporal_sampling.py` | `thermal_v8_mvitv2s_cache64_sample32_submission.py` | 64->32 ordered temporal sampling |
| `04_v9_fullspan_timewarp.py` | `thermal_v9_mvitv2s_fullspan_timewarp_submission.py` | Controlled time-warp study |
| `05_v11_maskfeat.py` | `thermal_v11_mvitv2s_maskfeat_pretrain_v8_finetune_submission.py` | MaskFeat-style adaptation |
| `06_v16_160px_maskfeat.py` | `thermal_v16_mvitv2s_maskfeat_160_v11_finetune_submission.py` | Final-line 160-pixel milestone |
| `07_v17_highgrid.py` | `thermal_v17_mvitv2s_maskfeat_highgrid_160_v16_finetune_submission.py` | Later controlled variant |
| `08_v18_192px.py` | `thermal_v18_mvitv2s_maskfeat_192_v16_finetune_submission.py` | Resolution study |
| `09_v19_lpft.py` | `thermal_v19_mvitv2s_maskfeat_160_lpft_v16_submission.py` | LPFT study |
| `10_v20_progressive_160to192.py` | `thermal_v20_mvitv2s_160to192_progressive_finetune_submission.py` | Progressive-fine-tuning study |
| `11_v21_late_mpa.py` | `thermal_v21_mvitv2s_maskfeat_160_late_mpa_submission.py` | Late adaptation study |
| `12_v22_temporal_consistency.py` | `thermal_v22_mvitv2s_160_temporal_latent_consistency_submission.py` | Consistency study |
| `13_v23_visual_oof_reweight.py` | `thermal_v23_mvitv2s_160_visual_oof_reweight_submission.py` | Cross-stream reweight study |

## IMU (`src/imu`)

| Destination filename | Working-tree source | Why include |
| --- | --- | --- |
| `01_interpolated_baseline.py` | `imu_transformer_baseline.py` | Original 64-step interpolation baseline |
| `02_native_sampling.py` | `imu_transformer_no_interp.py` | Timestamp-native preprocessing |
| `03_structured_dropout.py` | `imu_transformer_structured_dropout_v5.py` | Sensor robustness study |
| `04_user_balanced.py` | `imu_transformer_user_balanced_v81.py` | Subject balancing study |
| `05_time_aware.py` | `imu_transformer_time_aware_v9.py` | Explicit timing features |
| `06_energy_branch.py` | `imu_transformer_energy_branch_v10.py` | Invariant energy branch |
| `07_quaternion_continuity.py` | `imu_transformer_quaternion_continuity_v12.py` | Quaternion sign continuity |

## Skeleton (`src/skeleton`)

| Destination filename | Working-tree source | Why include |
| --- | --- | --- |
| `01_cross_subject_baseline.py` | `skeleton_cross_subject.py` | Subject-disjoint baseline |
| `02_oof_export.py` | `skeleton_best_with_oof.py` | OOF contract for fusion diagnostics |
| `03_hip_scale.py` | `skeleton_hip_scale.py` | Geometry normalization study |
| `04_body_part_pool.py` | `skeleton_body_part_pool.py` | Body-part pooling study |
| `05_ema.py` | `skeleton_ema.py` | EMA study |
| *(not copied yet)* `06_motionbert_transfer.py` | `skeleton_motionbert_v21.py` | Optional transfer-learning experiment. Excluded from this first release because it downloads third-party source/checkpoints; add it only after a separate MotionBERT license and provenance review. |

## Fusion (`src/fusion`)

| Destination filename | Working-tree source | Why include |
| --- | --- | --- |
| `01_v11_geometric_fusion.py` | `v11_visual_thermal_geometric_fusion_one_cell.py` | Calibrated geometric fusion |
| `02_v12_cross_user_shrinkage.py` | `v12_visual_thermal_v16_geometric_fusion_one_cell.py` | LOOU and conservative shrinkage |
| `03_v13_class_shrinkage.py` | `v13_visual_thermal_class_shrinkage_one_cell.py` | Bounded class residuals |
| `04_v14_stable_extrapolation.py` | `v14_stable_saturated_class_extrapolation_one_cell.py` | Stable three-class extrapolation |
| `05_v15_bayesian_arbitration.py` | `v15_bayesian_disagreement_arbitration_one_cell.py` | Negative-result routing study |
| `06_v20_imu_skeleton_arbitration.py` | `v20_imu_skeleton_vt_conflict_arbitration_one_cell.py` | Third-modality diagnostic; label clearly as negative result |

## Never copy

- All `*.pt`, `*.npy`, `*.npz`, `*.parquet`, `*.csv`, archives, Kaggle working outputs, diagnostic galleries, caches, and raw data.
- `visual_model_archive/ensemble_packed.pt`, `visual_model_archive/yolo11n.pt`, and any copied visual baseline code/checkpoint until the upstream license and reuse permission are explicitly reviewed.
- Any notebook with saved outputs, hidden paths, API tokens, mounted private Kaggle datasets, or competition submissions.
