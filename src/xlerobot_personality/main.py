from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys

from .config import AppConfig, load_config
from .orchestrator import SystemOrchestrator
from .scene_agent import SceneAgent
from .speech import TextToSpeechPlayer
from .tracking_controller import PIDController, PanTiltTrackingController
from .xlerobot_head import XLERobotHead, build_head_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XLeRobot personality starter")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config")
    parser.add_argument("--dry-run", action="store_true", help="Force mock hardware + synthetic camera")
    parser.add_argument(
        "--camera-source",
        type=str,
        choices=["synthetic", "opencv", "lerobot"],
        default=None,
        help="Override camera source",
    )
    parser.add_argument("--duration", type=float, default=None, help="Run duration in seconds (0 = infinite)")
    parser.add_argument("--no-dialog", action="store_true", help="Disable terminal Q/A loop")
    parser.add_argument("--visualize", action="store_true", help="Enable OpenCV window")
    parser.add_argument("--no-visualize", action="store_true", help="Disable OpenCV window")
    parser.add_argument("--web-preview", action="store_true", help="Serve browser preview on an HTTP port")
    parser.add_argument("--no-web-preview", action="store_true", help="Disable browser preview")
    parser.add_argument("--web-host", type=str, default=None, help="Bind host for browser preview")
    parser.add_argument("--web-port", type=int, default=None, help="Bind port for browser preview")
    return parser.parse_args()


def apply_cli_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.dry_run:
        cfg.hardware.use_mock = True
        cfg.camera.source = "synthetic"

    if args.camera_source is not None:
        cfg.camera.source = args.camera_source

    if args.duration is not None:
        cfg.runtime.run_duration_s = args.duration

    if args.no_dialog:
        cfg.runtime.enable_dialog = False

    if args.visualize:
        cfg.runtime.visualize = True

    if args.no_visualize:
        cfg.runtime.visualize = False

    if args.web_preview:
        cfg.runtime.web_preview = True

    if args.no_web_preview:
        cfg.runtime.web_preview = False

    if args.web_host is not None:
        cfg.runtime.web_host = args.web_host

    if args.web_port is not None:
        cfg.runtime.web_port = args.web_port

    return cfg


def build_controller(cfg: AppConfig, robot: XLERobotHead) -> PanTiltTrackingController:
    backend = cfg.tracking.backend.lower()
    if backend == "auto":
        if cfg.camera.source.lower() == "synthetic":
            backend = "motion"
        else:
            backend = "yolo_pose_person" if importlib.util.find_spec("ultralytics") is not None else "hog_person"

    controller = PanTiltTrackingController(
        robot=robot,
        camera_key=cfg.camera.key,
        tracking_backend=backend,
        hfov_deg=cfg.tracking.hfov_deg,
        vfov_deg=cfg.tracking.vfov_deg,
        frame_target_x_ratio=cfg.tracking.frame_target_x_ratio,
        frame_target_y_ratio=cfg.tracking.frame_target_y_ratio,
        tilt_tracking_sign=cfg.tracking.tilt_tracking_sign,
        tilt_error_gain=cfg.tracking.tilt_error_gain,
        pan_limits_deg=cfg.hardware.servo.pan_limits_deg,
        tilt_limits_deg=cfg.hardware.servo.tilt_limits_deg,
        max_step_deg=cfg.hardware.servo.max_step_deg,
        max_speed_deg_s=cfg.hardware.servo.max_speed_deg_s,
        manual_speed_deg_s=cfg.hardware.servo.manual_speed_deg_s,
        command_smoothing_s=cfg.hardware.servo.command_smoothing_s,
        transition_smoothing_s=cfg.hardware.servo.transition_smoothing_s,
        detector_confidence=cfg.tracking.detector_confidence,
        acquire_confirm_frames=cfg.tracking.acquire_confirm_frames,
        detector_interval=cfg.tracking.detector_interval,
        detector_max_dim_px=cfg.tracking.detector_max_dim_px,
        track_hold_s=cfg.tracking.track_hold_s,
        track_match_distance_px=cfg.tracking.track_match_distance_px,
        target_selection_mode=cfg.tracking.target_selection_mode,
        person_target_x_ratio=cfg.tracking.person_target_x_ratio,
        person_target_y_ratio=cfg.tracking.person_target_y_ratio,
        person_min_full_body_aspect_ratio=cfg.tracking.person_min_full_body_aspect_ratio,
        person_top_frame_ratio=cfg.tracking.person_top_frame_ratio,
        person_top_framing_gain=cfg.tracking.person_top_framing_gain,
        person_closeup_height_ratio=cfg.tracking.person_closeup_height_ratio,
        person_closeup_top_gain=cfg.tracking.person_closeup_top_gain,
        yolo_model=cfg.tracking.yolo_model,
        yolo_pose_model=cfg.tracking.yolo_pose_model,
        yolo_device=cfg.tracking.yolo_device,
        yolo_imgsz=cfg.tracking.yolo_imgsz,
        target_timeout_s=cfg.tracking.target_timeout_s,
        reacquire_duration_s=cfg.tracking.reacquire_duration_s,
        scan_speed_deg_s=cfg.tracking.scan_speed_deg_s,
        scan_tilt_center_deg=cfg.tracking.scan_tilt_center_deg,
        scan_tilt_amplitude_deg=cfg.tracking.scan_tilt_amplitude_deg,
        scan_nod_hz=cfg.tracking.scan_nod_hz,
        pan_pid=PIDController(
            kp=cfg.tracking.pan_pid.kp,
            ki=cfg.tracking.pan_pid.ki,
            kd=cfg.tracking.pan_pid.kd,
            integral_limit=cfg.tracking.pan_pid.integral_limit,
        ),
        tilt_pid=PIDController(
            kp=cfg.tracking.tilt_pid.kp,
            ki=cfg.tracking.tilt_pid.ki,
            kd=cfg.tracking.tilt_pid.kd,
            integral_limit=cfg.tracking.tilt_pid.integral_limit,
        ),
        min_area_px=cfg.tracking.min_area_px,
    )
    controller.set_tracking_enabled(cfg.tracking.start_tracking_enabled)
    return controller


def build_speaker(cfg: AppConfig) -> TextToSpeechPlayer | None:
    if not cfg.runtime.speech.enabled:
        return None

    speaker = TextToSpeechPlayer(
        backend=cfg.runtime.speech.backend,
        model_path=cfg.runtime.speech.model_path,
        config_path=cfg.runtime.speech.config_path,
        audio_player=cfg.runtime.speech.audio_player,
        speaker_id=cfg.runtime.speech.speaker_id,
        length_scale=cfg.runtime.speech.length_scale,
        noise_scale=cfg.runtime.speech.noise_scale,
        noise_w_scale=cfg.runtime.speech.noise_w_scale,
        volume=cfg.runtime.speech.volume,
        use_cuda=cfg.runtime.speech.use_cuda,
        lead_in_ms=cfg.runtime.speech.lead_in_ms,
    )
    if not speaker.available:
        print(
            f"[warn] Speech is enabled, but Piper is unavailable: {speaker.error}",
            flush=True,
        )
        return None

    print(
        f"[speech] Enabled with `{speaker.backend_name}` via `{speaker.player_name}`.",
        flush=True,
    )
    return speaker


def run(cfg: AppConfig) -> None:
    head_cfg = build_head_config(
        robot_id=cfg.hardware.robot_id,
        calibration_dir=cfg.hardware.calibration_dir,
        use_mock=cfg.hardware.use_mock,
        serial_port=cfg.hardware.serial_port,
        motor_model=cfg.hardware.motor_model,
        pan_id=cfg.hardware.pan_id,
        tilt_id=cfg.hardware.tilt_id,
        use_degrees=cfg.hardware.use_degrees,
        disable_torque_on_disconnect=cfg.hardware.disable_torque_on_disconnect,
        max_relative_target=cfg.hardware.max_relative_target,
        pan_limits_deg=cfg.hardware.servo.pan_limits_deg,
        tilt_limits_deg=cfg.hardware.servo.tilt_limits_deg,
        position_p_coefficient=cfg.hardware.servo.position_p_coefficient,
        position_i_coefficient=cfg.hardware.servo.position_i_coefficient,
        position_d_coefficient=cfg.hardware.servo.position_d_coefficient,
        camera_key=cfg.camera.key,
        camera_source=cfg.camera.source,
        camera_index=cfg.camera.device_index,
        camera_width=cfg.camera.width,
        camera_height=cfg.camera.height,
        camera_fps=cfg.camera.fps,
    )

    robot = XLERobotHead(head_cfg)
    controller = build_controller(cfg, robot)
    speaker = build_speaker(cfg)
    scene_agent = SceneAgent(
        memory_seconds=cfg.scene.memory_seconds,
        use_ollama=cfg.scene.use_ollama,
        vlm_model=cfg.scene.vlm_model,
        ollama_url=cfg.scene.ollama_url,
        ollama_timeout_s=cfg.scene.ollama_timeout_s,
        ollama_keep_alive=cfg.scene.ollama_keep_alive,
        max_image_dim_px=cfg.scene.max_image_dim_px,
        vlm_only_on_significant_change=cfg.scene.vlm_only_on_significant_change,
        vlm_change_threshold=cfg.scene.vlm_change_threshold,
        vlm_change_sample_dim_px=cfg.scene.vlm_change_sample_dim_px,
        vlm_target_center_change_ratio=cfg.scene.vlm_target_center_change_ratio,
        vlm_target_area_change_ratio=cfg.scene.vlm_target_area_change_ratio,
        vlm_force_refresh_s=cfg.scene.vlm_force_refresh_s,
        use_brain=cfg.scene.use_brain,
        brain_model=cfg.scene.brain_model,
        brain_temperature=cfg.scene.brain_temperature,
        brain_max_tokens=cfg.scene.brain_max_tokens,
        include_chat_history=cfg.scene.include_chat_history,
        chat_history_max_turns=cfg.scene.chat_history_max_turns,
        use_openai=cfg.scene.use_openai,
        openai_model=cfg.scene.openai_model,
        openai_api_key_env=cfg.scene.openai_api_key_env,
    )

    orchestrator = SystemOrchestrator(
        robot=robot,
        controller=controller,
        scene_agent=scene_agent,
        tracking_hz=cfg.tracking.loop_hz,
        scene_summary_hz=cfg.scene.summary_hz,
        interactive_hold_s=cfg.scene.interactive_hold_s,
        enable_dialog=cfg.runtime.enable_dialog,
        speaker=speaker,
        visualize=cfg.runtime.visualize,
        web_preview=cfg.runtime.web_preview,
        web_host=cfg.runtime.web_host,
        web_port=cfg.runtime.web_port,
        camera_key=cfg.camera.key,
        run_duration_s=cfg.runtime.run_duration_s,
    )

    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        pass


def _display_available() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    if cfg.runtime.visualize and not _display_available():
        print(
            "[warn] No graphical display detected (DISPLAY/WAYLAND_DISPLAY unset). "
            "Disabling visualization to avoid OpenCV Qt crash in headless/SSH sessions.",
            flush=True,
        )
        cfg.runtime.visualize = False

    if cfg.runtime.enable_dialog and not sys.stdin.isatty():
        print(
            "[warn] Standard input is not an interactive terminal. Disabling dialog loop.",
            flush=True,
        )
        cfg.runtime.enable_dialog = False

    run(cfg)


if __name__ == "__main__":
    main()
