from pathlib import Path

from inference.config import load_inference_settings


def test_missing_adapter_path_uses_base_model(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[artifacts]
pretrained = "artifacts/pretrained"

[model]
default_family = "qwen_vl"
default_size = "tiny"

[inference]
precision = "auto"
max_new_tokens = 64
""".strip()
    )

    settings = load_inference_settings(config_path)

    assert settings.model.adapter_path is None
    assert settings.model.model_id == "Qwen/Qwen2-VL-2B-Instruct"
    assert settings.model.model_path == Path("artifacts/pretrained/tiny")
    assert settings.config.image_max_pixels == 1_003_520


def test_adapter_path_loads_lora_adapter(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[artifacts]
pretrained = "artifacts/pretrained"

[model]
default_family = "qwen_vl"
default_size = "tiny"

[inference]
precision = "auto"
adapter_path = "artifacts/adapters/demo"
max_new_tokens = 64
image_max_pixels = 786432
""".strip()
    )

    settings = load_inference_settings(config_path)

    assert settings.model.adapter_path == Path("artifacts/adapters/demo")
    assert settings.config.image_max_pixels == 786432
