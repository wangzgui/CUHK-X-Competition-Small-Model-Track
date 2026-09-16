# Credits, provenance, and licensing boundary

## Upstream visual baseline

The IR + Depth_Color visual stream is derived from the public Kaggle notebook by **`welshonionman`**:

- Title: *[LB0.667] baseline with YOLO person crop* ([Kaggle notebook](https://www.kaggle.com/code/welshonionman/lb0-667-baseline-with-yolo-person-crop/notebook))
- Contribution used as the starting point: IR-based YOLO person crop, four-channel IR + Depth_Color video input, and an R(2+1)D visual baseline.
- This project reported that line at **0.72139** on the public board. It is credited as an upstream public baseline, not presented as this repository's original visual-model contribution.

The Kaggle page lists the notebook as Apache-2.0. This repository nevertheless does not copy its visual source or trained assets: attribution is preserved and the release scope remains focused on the downstream multimodal research.

## CUHK-X data and competition assets

No CUHK-X data, test labels, submissions, derived prediction arrays, frame caches, fine-tuned weights, or organizer-controlled files are included here. Access and use remain subject to the organizer's competition and dataset terms, including any non-commercial and attribution restrictions.

## Third-party code and model lineage

- MViTv2-S and R(2+1)D architectures: torchvision / PyTorch project licenses.
- IG-65M R(2+1)D lineage used by the upstream visual branch: `moabitcoin/ig65m-pytorch` (MIT); see the original repository before redistributing any architecture files.
- YOLO11n / Ultralytics: subject to Ultralytics licensing. This release does not bundle `yolo11n.pt`.
- MotionBERT experiments: subject to MotionBERT's original repository and checkpoint licenses; no model code or checkpoint should be copied until those terms have been checked.

## License for this repository

The root [MIT License](../LICENSE) applies only to original source code and documentation contributed by this repository's authors. It does **not** grant rights to CUHK-X data, derived assets, weights, external checkpoints, or third-party components. Those remain governed by their respective terms above.
