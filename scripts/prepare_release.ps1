$ErrorActionPreference = 'Stop'

# Run from the release directory. This script copies only reviewed Python source
# from the adjacent original working tree; it refuses to copy any data or model asset.
$releaseRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = Split-Path -Parent $releaseRoot

$manifest = @{
  'src/thermal/01_r2plus1d18_baseline.py' = 'thermal_r2plus1d18.py'
  'src/thermal/02_mvitv2s_full32.py' = 'thermal_full32_calibration_mixup_mvitv2s_boundary_submission.py'
  'src/thermal/03_v8_ordered_temporal_sampling.py' = 'thermal_v8_mvitv2s_cache64_sample32_submission.py'
  'src/thermal/04_v9_fullspan_timewarp.py' = 'thermal_v9_mvitv2s_fullspan_timewarp_submission.py'
  'src/thermal/05_v11_maskfeat.py' = 'thermal_v11_mvitv2s_maskfeat_pretrain_v8_finetune_submission.py'
  'src/thermal/06_v16_160px_maskfeat.py' = 'thermal_v16_mvitv2s_maskfeat_160_v11_finetune_submission.py'
  'src/thermal/07_v17_highgrid.py' = 'thermal_v17_mvitv2s_maskfeat_highgrid_160_v16_finetune_submission.py'
  'src/thermal/08_v18_192px.py' = 'thermal_v18_mvitv2s_maskfeat_192_v16_finetune_submission.py'
  'src/thermal/09_v19_lpft.py' = 'thermal_v19_mvitv2s_maskfeat_160_lpft_v16_submission.py'
  'src/thermal/10_v20_progressive_160to192.py' = 'thermal_v20_mvitv2s_160to192_progressive_finetune_submission.py'
  'src/thermal/11_v21_late_mpa.py' = 'thermal_v21_mvitv2s_maskfeat_160_late_mpa_submission.py'
  'src/thermal/12_v22_temporal_consistency.py' = 'thermal_v22_mvitv2s_160_temporal_latent_consistency_submission.py'
  'src/thermal/13_v23_visual_oof_reweight.py' = 'thermal_v23_mvitv2s_160_visual_oof_reweight_submission.py'
  'src/imu/01_interpolated_baseline.py' = 'imu_transformer_baseline.py'
  'src/imu/02_native_sampling.py' = 'imu_transformer_no_interp.py'
  'src/imu/03_structured_dropout.py' = 'imu_transformer_structured_dropout_v5.py'
  'src/imu/04_user_balanced.py' = 'imu_transformer_user_balanced_v81.py'
  'src/imu/05_time_aware.py' = 'imu_transformer_time_aware_v9.py'
  'src/imu/06_energy_branch.py' = 'imu_transformer_energy_branch_v10.py'
  'src/imu/07_quaternion_continuity.py' = 'imu_transformer_quaternion_continuity_v12.py'
  'src/skeleton/01_cross_subject_baseline.py' = 'skeleton_cross_subject.py'
  'src/skeleton/02_oof_export.py' = 'skeleton_best_with_oof.py'
  'src/skeleton/03_hip_scale.py' = 'skeleton_hip_scale.py'
  'src/skeleton/04_body_part_pool.py' = 'skeleton_body_part_pool.py'
  'src/skeleton/05_ema.py' = 'skeleton_ema.py'
  'src/fusion/01_v11_geometric_fusion.py' = 'v11_visual_thermal_geometric_fusion_one_cell.py'
  'src/fusion/02_v12_cross_user_shrinkage.py' = 'v12_visual_thermal_v16_geometric_fusion_one_cell.py'
  'src/fusion/03_v13_class_shrinkage.py' = 'v13_visual_thermal_class_shrinkage_one_cell.py'
  'src/fusion/04_v14_stable_extrapolation.py' = 'v14_stable_saturated_class_extrapolation_one_cell.py'
  'src/fusion/05_v15_bayesian_arbitration.py' = 'v15_bayesian_disagreement_arbitration_one_cell.py'
  'src/fusion/06_v20_imu_skeleton_arbitration.py' = 'v20_imu_skeleton_vt_conflict_arbitration_one_cell.py'
}

foreach ($entry in $manifest.GetEnumerator()) {
  $source = Join-Path $sourceRoot $entry.Value
  $destination = Join-Path $releaseRoot $entry.Key
  if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
    throw "Missing reviewed source: $source"
  }
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
  Copy-Item -LiteralPath $source -Destination $destination -Force
}

Write-Host "Copied $($manifest.Count) reviewed Python sources. Review git diff before publishing."
