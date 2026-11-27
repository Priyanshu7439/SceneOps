# Experiment tracking

src/sceneops/tracking/wandb_logging.py provides a disabled-by-default wrapper.

- SCENEOPS_WANDB_MODE=disabled: no import, run, or upload.
- offline: initializes W&B offline and logs locally.
- online: requires installed W&B and user-managed authentication.

The wrapper is designed to record configuration, dataset metadata, model metadata, training/validation metrics, evaluation metrics, and clean termination. It never uploads full datasets or model artifacts automatically. Mock/offline behavior is tested; no remote run or W&B metric is claimed.
