"""Structural validation for the optional Axolotl reference configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_KEYS = {
    "base_model",
    "datasets",
    "output_dir",
    "sequence_len",
    "adapter",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
}


def validate_axolotl_config(
    path: Path, *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Validate local structure and paths without invoking Axolotl training."""

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("Axolotl configuration must be a YAML mapping.")

    missing = sorted(REQUIRED_KEYS - data.keys())
    if missing:
        raise ValueError(f"Axolotl configuration is missing keys: {missing}")
    if data["base_model"] != "Qwen/Qwen2.5-1.5B-Instruct":
        raise ValueError("Unexpected base model identifier.")
    if data["adapter"] != "lora":
        raise ValueError("The reference configuration must use LoRA.")
    if not isinstance(data["datasets"], list) or len(data["datasets"]) != 1:
        raise ValueError("The reference configuration must declare one dataset.")

    dataset = data["datasets"][0]
    if not isinstance(dataset, dict) or dataset.get("type") != "chat_template":
        raise ValueError("The dataset must use Axolotl chat_template format.")
    root = repo_root or path.resolve().parents[2]
    dataset_path = root / dataset["path"]
    if not dataset_path.is_file():
        raise ValueError(f"Configured dataset path does not exist: {dataset_path}")

    if data["lora_r"] != 16 or data["lora_alpha"] != 32:
        raise ValueError("LoRA rank/alpha do not match the documented setup.")
    if data["sequence_len"] != 512:
        raise ValueError("Sequence length does not match the documented setup.")
    return data
