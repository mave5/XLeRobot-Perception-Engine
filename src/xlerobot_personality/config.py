from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import os
from pathlib import Path
from typing import Any
import json


@dataclass
class PIDConfig:
    kp: float = 0.7
    ki: float = 0.0
    kd: float = 0.08
    integral_limit: float = 25.0


@dataclass
class ServoConfig:
    pan_limits_deg: tuple[float, float] = (-80.0, 80.0)
    tilt_limits_deg: tuple[float, float] = (-35.0, 45.0)
    max_step_deg: float = 4.0
    max_speed_deg_s: float = 120.0
    manual_speed_deg_s: float = 35.0
    command_smoothing_s: float = 0.12
    transition_smoothing_s: float = 0.30
    position_p_coefficient: int | None = None
    position_i_coefficient: int | None = None
    position_d_coefficient: int | None = None


@dataclass
class HeadHardwareConfig:
    robot_id: str = "xlerobot_head"
    calibration_dir: str | None = None
    use_mock: bool = True
    serial_port: str | None = None
    motor_model: str = "sts3215"
    pan_id: int = 1
    tilt_id: int = 2
    use_degrees: bool = True
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | None = 8.0
    servo: ServoConfig = field(default_factory=ServoConfig)


@dataclass
class CameraRuntimeConfig:
    key: str = "head_cam"
    source: str = "synthetic"  # synthetic | opencv | lerobot
    device_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class TrackingConfig:
    start_tracking_enabled: bool = False
    loop_hz: float = 30.0
    backend: str = "auto"  # auto | motion | hog_person | yolo_person | yolo_pose_person
    hfov_deg: float = 70.0
    vfov_deg: float = 43.0
    frame_target_x_ratio: float = 0.5
    frame_target_y_ratio: float = 0.58
    tilt_tracking_sign: float = -1.0
    tilt_error_gain: float = 0.75
    min_area_px: int = 700
    detector_confidence: float = 0.2
    acquire_confirm_frames: int = 3
    detector_interval: int = 3
    detector_max_dim_px: int = 480
    track_hold_s: float = 0.4
    track_match_distance_px: float = 180.0
    target_selection_mode: str = "sticky"  # sticky | most_centered
    person_target_x_ratio: float = 0.5
    person_target_y_ratio: float = 0.62
    person_min_full_body_aspect_ratio: float = 2.4
    person_top_frame_ratio: float = 0.18
    person_top_framing_gain: float = 0.45
    person_closeup_height_ratio: float = 0.45
    person_closeup_top_gain: float = 0.9
    yolo_model: str = "yolov8n.pt"
    yolo_pose_model: str = "yolov8n-pose.pt"
    yolo_device: str = "cpu"
    yolo_imgsz: int = 640
    target_timeout_s: float = 0.7
    reacquire_duration_s: float = 1.0
    scan_speed_deg_s: float = 20.0
    scan_tilt_center_deg: float = 0.0
    scan_tilt_amplitude_deg: float = 10.0
    scan_nod_hz: float = 0.2
    pan_pid: PIDConfig = field(default_factory=PIDConfig)
    tilt_pid: PIDConfig = field(default_factory=lambda: PIDConfig(kp=0.6, kd=0.06, integral_limit=20.0))


@dataclass
class SceneConfig:
    summary_hz: float = 1.0
    memory_seconds: float = 30.0
    interactive_hold_s: float = 2.0
    use_ollama: bool = False
    vlm_model: str = "qwen2.5vl:3b"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_timeout_s: float = 20.0
    ollama_keep_alive: str = "10m"
    max_image_dim_px: int = 672
    vlm_only_on_significant_change: bool = True
    vlm_change_threshold: float = 0.08
    vlm_change_sample_dim_px: int = 96
    vlm_target_center_change_ratio: float = 0.12
    vlm_target_area_change_ratio: float = 0.28
    vlm_force_refresh_s: float = 20.0
    use_brain: bool = False
    brain_model: str = "qwen2.5:1.5b"
    brain_temperature: float = 0.5
    brain_max_tokens: int = 96
    include_chat_history: bool = True
    chat_history_max_turns: int = 10
    use_openai: bool = False
    openai_model: str = "gpt-4.1-mini"
    openai_api_key_env: str = "OPENAI_API_KEY"


@dataclass
class RuntimeConfig:
    @dataclass
    class SpeechConfig:
        enabled: bool = False
        backend: str = "piper"  # piper
        model_path: str | None = None
        config_path: str | None = None
        audio_player: str = "auto"  # auto | aplay | paplay | ffplay | afplay
        speaker_id: int | None = None
        length_scale: float | None = None
        noise_scale: float | None = None
        noise_w_scale: float | None = None
        volume: float = 1.0
        use_cuda: bool = False
        lead_in_ms: int = 250

    run_duration_s: float = 0.0
    enable_dialog: bool = True
    visualize: bool = True
    web_preview: bool = False
    web_host: str = "127.0.0.1"
    web_port: int = 8765
    speech: SpeechConfig = field(default_factory=SpeechConfig)


@dataclass
class AppConfig:
    hardware: HeadHardwareConfig = field(default_factory=HeadHardwareConfig)
    camera: CameraRuntimeConfig = field(default_factory=CameraRuntimeConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _update_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    if isinstance(instance, SceneConfig):
        legacy_key_map = {
            "ollama_model": "vlm_model",
            "ollama_max_image_dim_px": "max_image_dim_px",
            "use_ollama_brain": "use_brain",
            "ollama_brain_model": "brain_model",
            "ollama_brain_temperature": "brain_temperature",
            "ollama_brain_max_tokens": "brain_max_tokens",
        }
        remapped_values = None
        for legacy_key, new_key in legacy_key_map.items():
            if legacy_key in values and new_key not in values:
                if remapped_values is None:
                    remapped_values = dict(values)
                remapped_values[new_key] = remapped_values.pop(legacy_key)
        if remapped_values is not None:
            values = remapped_values

    for key, value in values.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
            continue
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(instance, key, value)
    return instance


def _load_data(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required for YAML config files. Install with: pip install PyYAML") from exc

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} did not parse as a dictionary")
    return data


def _normalize_path_value(value: str | None, *, base_dir: Path) -> str | None:
    if not value:
        return value

    normalized = Path(os.path.expandvars(value)).expanduser()
    if not normalized.is_absolute():
        normalized = (base_dir / normalized).resolve(strict=False)
    return str(normalized)


def _normalize_config_paths(config: AppConfig, *, base_dir: Path) -> AppConfig:
    config.hardware.calibration_dir = _normalize_path_value(config.hardware.calibration_dir, base_dir=base_dir)
    config.runtime.speech.model_path = _normalize_path_value(config.runtime.speech.model_path, base_dir=base_dir)
    config.runtime.speech.config_path = _normalize_path_value(config.runtime.speech.config_path, base_dir=base_dir)
    return config


def load_config(path: str | Path | None = None) -> AppConfig:
    config = AppConfig()
    if path is None:
        return config

    cfg_path = Path(os.path.expandvars(str(path))).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    data = _load_data(cfg_path)
    config = _update_dataclass(config, data)
    return _normalize_config_paths(config, base_dir=cfg_path.parent)
