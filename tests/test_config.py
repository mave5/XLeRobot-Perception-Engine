from textwrap import dedent

from xlerobot_personality.config import AppConfig, _update_dataclass
from xlerobot_personality.config import load_config


def test_tuple_conversion_from_list():
    cfg = AppConfig()
    _update_dataclass(cfg, {"hardware": {"servo": {"pan_limits_deg": [-10.0, 15.0]}}})
    assert cfg.hardware.servo.pan_limits_deg == (-10.0, 15.0)


def test_nested_speech_runtime_config_updates():
    cfg = AppConfig()
    _update_dataclass(
        cfg,
        {
            "runtime": {
                "speech": {
                    "enabled": True,
                    "backend": "piper",
                    "model_path": "/tmp/voice.onnx",
                    "audio_player": "ffplay",
                    "speaker_id": 2,
                    "length_scale": 0.85,
                    "lead_in_ms": 180,
                }
            }
        },
    )

    assert cfg.runtime.speech.enabled is True
    assert cfg.runtime.speech.backend == "piper"
    assert cfg.runtime.speech.model_path == "/tmp/voice.onnx"
    assert cfg.runtime.speech.audio_player == "ffplay"
    assert cfg.runtime.speech.speaker_id == 2
    assert cfg.runtime.speech.length_scale == 0.85
    assert cfg.runtime.speech.lead_in_ms == 180


def test_load_config_expands_home_and_relative_paths(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "portable.yaml"
    config_path.write_text(
        dedent(
            """
            hardware:
              calibration_dir: ~/.cache/huggingface/lerobot/calibration/robots/xlerobot_head
            runtime:
              speech:
                model_path: ../en_US-lessac-medium.onnx
                config_path: ${HOME}/voices/en_US-lessac-medium.onnx.json
            """
        ).strip()
        + "\n"
    )

    cfg = load_config(config_path)

    assert cfg.hardware.calibration_dir == str(
        home_dir / ".cache/huggingface/lerobot/calibration/robots/xlerobot_head"
    )
    assert cfg.runtime.speech.model_path == str((tmp_path / "en_US-lessac-medium.onnx").resolve(strict=False))
    assert cfg.runtime.speech.config_path == str(home_dir / "voices/en_US-lessac-medium.onnx.json")


def test_load_config_accepts_legacy_ollama_scene_keys(tmp_path):
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(
        dedent(
            """
            scene:
              ollama_model: qwen2.5vl:3b
              ollama_max_image_dim_px: 512
              use_ollama_brain: true
              ollama_brain_model: qwen2.5:1.5b
              ollama_brain_temperature: 0.7
              ollama_brain_max_tokens: 144
            """
        ).strip()
        + "\n"
    )

    cfg = load_config(config_path)

    assert cfg.scene.vlm_model == "qwen2.5vl:3b"
    assert cfg.scene.max_image_dim_px == 512
    assert cfg.scene.use_brain is True
    assert cfg.scene.brain_model == "qwen2.5:1.5b"
    assert cfg.scene.brain_temperature == 0.7
    assert cfg.scene.brain_max_tokens == 144
