# Reproducibility and evaluation protocol

## Task

- 40 action classes.
- Train subjects: `user1`–`user9`, `user16`–`user24`.
- Test subjects: `user10`, `user11`, `user25`, `user26`.
- The competition imposed a 100 MB total model/detector constraint; check the exact organizer byte accounting before claiming compliance for any new deployment.

## Evaluation

Use subject-disjoint folds. Random clip splits leak person, room, clothing, and acquisition cues and are not a credible estimate for this task.

For fusion, the aligned OOF set contained 541 clips across `user9`, `user16`, and `user18`. This was adequate for low-dimensional, shrinkage-based diagnostics but not for a high-capacity 80-to-40 stacking model. Public/private leaderboard feedback is reported only as post-hoc competition evidence, never used as a replacement for test labels.

## Running a source file

Each released source file is Kaggle-oriented and contains a path/configuration block near the top. Before running:

1. Attach the official data through permitted channels.
2. Change only the path/configuration block to match the mounted dataset names.
3. Enable a CUDA GPU.
4. Start with CV/OOF mode, inspect data alignment and model-size outputs, then run final training/inference only if that experiment is justified.

The scripts intentionally write to `/kaggle/working`. Do not commit those outputs.

