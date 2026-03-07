from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from .config import AppConfig, load_config

ROBOT_TYPE = "xlerobot_head"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactively calibrate the XLERobot pan/tilt head and save a LeRobot calibration JSON."
    )
    parser.add_argument("--config", type=Path, help="Optional app config file to read hardware defaults from.")
    parser.add_argument("--robot-id", help="Robot id used for the calibration filename <robot-id>.json.")
    parser.add_argument("--calibration-dir", type=Path, help="Directory where the calibration JSON is stored.")
    parser.add_argument("--serial-port", help="Feetech serial port, for example /dev/ttyACM0.")
    parser.add_argument("--motor-model", help="Motor model shared by the head servos, for example sts3215.")
    parser.add_argument("--pan-id", type=int, help="Servo id for the pan axis.")
    parser.add_argument("--tilt-id", type=int, help="Servo id for the tilt axis.")
    parser.add_argument(
        "--use-degrees",
        dest="use_degrees",
        action="store_true",
        help="Use MotorNormMode.DEGREES for the created bus definition.",
    )
    parser.add_argument(
        "--use-range",
        dest="use_degrees",
        action="store_false",
        help="Use MotorNormMode.RANGE_M100_100 for the created bus definition.",
    )
    parser.set_defaults(use_degrees=None)
    parser.add_argument(
        "--pan-drive-mode",
        type=int,
        choices=(0, 1),
        default=0,
        help="Calibration drive_mode for pan. 0=non-inverted, 1=inverted.",
    )
    parser.add_argument(
        "--tilt-drive-mode",
        type=int,
        choices=(0, 1),
        default=0,
        help="Calibration drive_mode for tilt. 0=non-inverted, 1=inverted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing calibration file without an interactive confirmation prompt.",
    )
    return parser


def require_lerobot() -> dict[str, Any]:
    try:
        from lerobot.motors import Motor, MotorCalibration, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
        from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LeRobot Feetech support is required for head calibration. Install lerobot with the feetech extra "
            "in this virtualenv before running this utility."
        ) from exc

    return {
        "Motor": Motor,
        "MotorCalibration": MotorCalibration,
        "MotorNormMode": MotorNormMode,
        "FeetechMotorsBus": FeetechMotorsBus,
        "OperatingMode": OperatingMode,
        "HF_LEROBOT_CALIBRATION": HF_LEROBOT_CALIBRATION,
        "ROBOTS": ROBOTS,
    }


def resolve_settings(args: argparse.Namespace, cfg: AppConfig | None, hf_calibration_dir: Path) -> dict[str, Any]:
    hardware = cfg.hardware if cfg is not None else None

    robot_id = args.robot_id or (hardware.robot_id if hardware is not None else "xlerobot_head_real")
    serial_port = args.serial_port or (hardware.serial_port if hardware is not None else None)
    motor_model = args.motor_model or (hardware.motor_model if hardware is not None else "sts3215")
    pan_id = args.pan_id if args.pan_id is not None else (hardware.pan_id if hardware is not None else 1)
    tilt_id = args.tilt_id if args.tilt_id is not None else (hardware.tilt_id if hardware is not None else 2)
    use_degrees = args.use_degrees if args.use_degrees is not None else (
        hardware.use_degrees if hardware is not None else True
    )

    calibration_dir = args.calibration_dir
    if calibration_dir is None and hardware is not None and hardware.calibration_dir:
        calibration_dir = Path(hardware.calibration_dir)
    if calibration_dir is None:
        calibration_dir = hf_calibration_dir

    if not serial_port:
        raise ValueError("No serial port provided. Use --serial-port or a config file with hardware.serial_port.")
    if pan_id == tilt_id:
        raise ValueError(f"pan_id and tilt_id must be different, got {pan_id}.")
    if cfg is not None and cfg.hardware.use_mock:
        raise ValueError("The selected config has hardware.use_mock=true. Use a real-hardware config for calibration.")

    return {
        "robot_id": robot_id,
        "serial_port": serial_port,
        "motor_model": motor_model,
        "pan_id": pan_id,
        "tilt_id": tilt_id,
        "use_degrees": use_degrees,
        "calibration_dir": calibration_dir,
        "pan_drive_mode": args.pan_drive_mode,
        "tilt_drive_mode": args.tilt_drive_mode,
        "force": args.force,
    }


def ensure_output_path(calibration_path: Path, force: bool) -> None:
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    if not calibration_path.exists():
        return

    if not force:
        answer = input(
            f"Calibration file already exists at {calibration_path}. Type 'overwrite' to replace it, or press ENTER to abort: "
        ).strip()
        if answer.lower() != "overwrite":
            raise RuntimeError("Calibration aborted. Existing file was left unchanged.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = calibration_path.with_name(f"{calibration_path.stem}.{timestamp}.bak.json")
    shutil.copy2(calibration_path, backup_path)
    print(f"Backed up existing calibration to {backup_path}")


def save_calibration(calibration_path: Path, calibration: dict[str, Any]) -> None:
    payload = {motor_name: asdict(motor_calibration) for motor_name, motor_calibration in calibration.items()}
    calibration_path.write_text(json.dumps(payload, indent=4) + "\n")


def print_summary(calibration_path: Path, calibration: dict[str, Any]) -> None:
    print("\nSaved calibration:")
    for motor_name, motor_calibration in calibration.items():
        print(
            f"  {motor_name:<4} id={motor_calibration.id:<3} "
            f"offset={motor_calibration.homing_offset:<6} "
            f"range=[{motor_calibration.range_min}, {motor_calibration.range_max}] "
            f"drive_mode={motor_calibration.drive_mode}"
        )
    print(f"\nCalibration file: {calibration_path}")


def run_calibration(settings: dict[str, Any], deps: dict[str, Any]) -> Path:
    Motor = deps["Motor"]
    MotorCalibration = deps["MotorCalibration"]
    MotorNormMode = deps["MotorNormMode"]
    FeetechMotorsBus = deps["FeetechMotorsBus"]
    OperatingMode = deps["OperatingMode"]

    norm_mode = MotorNormMode.DEGREES if settings["use_degrees"] else MotorNormMode.RANGE_M100_100
    calibration_path = settings["calibration_dir"] / f"{settings['robot_id']}.json"
    ensure_output_path(calibration_path, settings["force"])

    print("Preparing head calibration.")
    print(f"  robot_id: {settings['robot_id']}")
    print(f"  serial_port: {settings['serial_port']}")
    print(f"  calibration_path: {calibration_path}")
    print("  motors: pan, tilt")
    print("\nBefore continuing:")
    print("  1. Make sure the head servos are on the specified serial bus.")
    print("  2. Support the head by hand before torque is disabled.")
    print("  3. Keep movement within the safe mechanical range.")
    input("\nPress ENTER to start calibration, or Ctrl+C to abort...")

    bus = FeetechMotorsBus(
        port=settings["serial_port"],
        motors={
            "pan": Motor(settings["pan_id"], settings["motor_model"], norm_mode),
            "tilt": Motor(settings["tilt_id"], settings["motor_model"], norm_mode),
        },
        calibration={},
    )

    try:
        bus.connect()
        bus.disable_torque()
        bus.configure_motors()
        for motor_name in bus.motors:
            bus.write("Operating_Mode", motor_name, OperatingMode.POSITION.value)

        input("\nMove the head to its neutral forward-looking pose and press ENTER...")
        homing_offsets = bus.set_half_turn_homings(["pan", "tilt"])

        print(
            "\nMove pan and tilt slowly through their full safe ranges.\n"
            "The utility will keep updating the observed min/max encoder values.\n"
            "Press ENTER once both axes have covered their full range."
        )
        range_mins, range_maxes = bus.record_ranges_of_motion(["pan", "tilt"])

        calibration = {
            "pan": MotorCalibration(
                id=settings["pan_id"],
                drive_mode=settings["pan_drive_mode"],
                homing_offset=int(homing_offsets["pan"]),
                range_min=int(range_mins["pan"]),
                range_max=int(range_maxes["pan"]),
            ),
            "tilt": MotorCalibration(
                id=settings["tilt_id"],
                drive_mode=settings["tilt_drive_mode"],
                homing_offset=int(homing_offsets["tilt"]),
                range_min=int(range_mins["tilt"]),
                range_max=int(range_maxes["tilt"]),
            ),
        }

        bus.write_calibration(calibration)
        save_calibration(calibration_path, calibration)
        print_summary(calibration_path, calibration)
        return calibration_path
    finally:
        try:
            bus.disconnect(disable_torque=True)
        except Exception:
            pass


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not sys.stdin.isatty():
        print("head_calibrate requires an interactive terminal.", file=sys.stderr)
        return 2

    cfg = load_config(args.config) if args.config else None

    try:
        deps = require_lerobot()
        hf_calibration_dir = deps["HF_LEROBOT_CALIBRATION"] / deps["ROBOTS"] / ROBOT_TYPE
        settings = resolve_settings(args, cfg, hf_calibration_dir)
        run_calibration(settings, deps)
    except KeyboardInterrupt:
        print("\nCalibration aborted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
