"""Opt-in W&B tracking with disabled and offline modes."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


def build_run_payload(
    *, config: Mapping[str, Any], metrics: Mapping[str, float | int]
) -> dict[str, dict[str, Any]]:
    """Return the metadata/metrics payload planned for a W&B run."""

    return {"config": dict(config), "metrics": dict(metrics)}


class WandbTracker:
    """Small optional tracker suitable for training/evaluation wrappers."""

    def __init__(
        self,
        *,
        project: str,
        run_name: str,
        config: Mapping[str, Any],
        mode: str | None = None,
        enabled: bool | None = None,
        wandb_module: Any | None = None,
    ) -> None:
        selected_mode = mode or os.getenv("SCENEOPS_WANDB_MODE", "disabled")
        if selected_mode not in {"disabled", "offline", "online"}:
            raise ValueError("W&B mode must be disabled, offline, or online")
        if enabled is None:
            enabled = selected_mode != "disabled"

        self.enabled = enabled
        self._run = None
        self.mode = selected_mode
        if not enabled:
            return

        module = wandb_module
        if module is None:
            try:
                import wandb as module
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "W&B is optional. Install wandb to enable tracking."
                ) from exc

        self._run = module.init(
            project=project,
            name=run_name,
            mode=selected_mode,
            config=dict(config),
        )

    def log_dataset_metadata(
        self,
        *,
        split_counts: Mapping[str, int],
        dataset_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._update_config(
            {
                "dataset_split_counts": dict(split_counts),
                "dataset_metadata": dict(dataset_metadata or {}),
            }
        )

    def log_metrics(self, metrics: Mapping[str, float | int]) -> None:
        if self._run is not None:
            self._run.log(dict(metrics))

    def log_model_metadata(self, metadata: Mapping[str, Any]) -> None:
        self._update_config({"model_metadata": dict(metadata)})

    def _update_config(self, values: Mapping[str, Any]) -> None:
        if self._run is not None:
            try:
                self._run.config.update(dict(values), allow_val_change=True)
            except TypeError:
                self._run.config.update(dict(values))

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


def log_offline(
    *,
    project: str,
    run_name: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, float | int],
) -> None:
    """Log locally in W&B offline mode; never uploads or logs model files."""

    tracker = WandbTracker(
        project=project,
        run_name=run_name,
        config=config,
        mode="offline",
    )
    try:
        tracker.log_metrics(metrics)
    finally:
        tracker.finish()
