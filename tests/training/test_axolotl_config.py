from pathlib import Path

from sceneops.training.axolotl_config import validate_axolotl_config


def test_optional_axolotl_config_is_structurally_valid():
    root = Path(__file__).parents[2]
    config = validate_axolotl_config(
        root / "configs" / "axolotl_sceneops_lora.yml", repo_root=root
    )
    assert config["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config["datasets"][0]["type"] == "chat_template"
