from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .camera_sources import OpenCVCameraSource, SyntheticMovingBlobCamera

HAS_LEROBOT = True

try:
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.cameras.utils import make_cameras_from_configs
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
    from lerobot.robots.config import RobotConfig
    from lerobot.robots.robot import Robot
    from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
except ModuleNotFoundError:
    HAS_LEROBOT = False

    class DeviceAlreadyConnectedError(RuntimeError):
        pass

    class DeviceNotConnectedError(RuntimeError):
        pass

    @dataclass(kw_only=True)
    class RobotConfig:
        id: str | None = None
        calibration_dir: Path | None = None

    class Robot:
        config_class: type[RobotConfig]
        name: str = "robot"

        def __init__(self, config: RobotConfig):
            self.config = config
            self.id = getattr(config, "id", None)
            self.calibration = {}

    class OpenCVCameraConfig:  # type: ignore[override]
        def __init__(self, index_or_path: int, fps: int, width: int, height: int):
            self.index_or_path = index_or_path
            self.fps = fps
            self.width = width
            self.height = height

    def make_cameras_from_configs(_: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("LeRobot camera backends are unavailable")

    class MotorNormMode:
        DEGREES = "degrees"
        RANGE_M100_100 = "range_m100_100"

    class Motor:
        def __init__(self, motor_id: int, model: str, norm_mode: str):
            self.id = motor_id
            self.model = model
            self.norm_mode = norm_mode

    class OperatingMode:
        POSITION = 0

    class FeetechMotorsBus:
        def __init__(self, *args: Any, **kwargs: Any):
            raise RuntimeError("LeRobot Feetech backend is unavailable. Use mock mode or install lerobot.")


def _register_robot_subclass(name: str):
    if HAS_LEROBOT and hasattr(RobotConfig, "register_subclass"):
        return RobotConfig.register_subclass(name)  # type: ignore[return-value]

    def _noop(cls):
        return cls

    return _noop


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@_register_robot_subclass("xlerobot_head")
@dataclass
class XLERobotHeadConfig(RobotConfig):
    id: str | None = "xlerobot_head"
    use_mock: bool = True

    port: str | None = None
    motor_model: str = "sts3215"
    pan_id: int = 1
    tilt_id: int = 2
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | None = 8.0
    use_degrees: bool = True

    pan_limits_deg: tuple[float, float] = (-80.0, 80.0)
    tilt_limits_deg: tuple[float, float] = (-35.0, 45.0)
    position_p_coefficient: int | None = None
    position_i_coefficient: int | None = None
    position_d_coefficient: int | None = None

    camera_key: str = "head_cam"
    camera_source: str = "synthetic"  # synthetic | opencv | lerobot
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    cameras: dict[str, Any] = field(default_factory=dict)


class XLERobotHead(Robot):
    """
    Starter pan/tilt head robot wrapper with LeRobot-compatible API.

    Action keys:
    - pan.pos
    - tilt.pos

    Observation keys:
    - pan.pos
    - tilt.pos
    - <camera_key>
    """

    config_class = XLERobotHeadConfig
    name = "xlerobot_head"

    def __init__(self, config: XLERobotHeadConfig):
        super().__init__(config)
        self.config = config

        self._connected = False
        self._pan_pos = 0.0
        self._tilt_pos = 0.0

        self.bus: FeetechMotorsBus | None = None
        self.cameras: dict[str, Any] = {}

        self._build_motor_backend()
        self._build_camera_backend()

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        return {
            "pan.pos": float,
            "tilt.pos": float,
            self.config.camera_key: (self.config.camera_height, self.config.camera_width, 3),
        }

    @property
    def action_features(self) -> dict[str, type]:
        return {"pan.pos": float, "tilt.pos": float}

    @property
    def is_connected(self) -> bool:
        motor_ok = True if self.bus is None else bool(self.bus.is_connected)
        camera_ok = all(getattr(cam, "is_connected", False) for cam in self.cameras.values())
        return self._connected and motor_ok and camera_ok

    @property
    def is_calibrated(self) -> bool:
        if self.bus is None:
            return True
        return bool(self.bus.is_calibrated)

    def _build_motor_backend(self) -> None:
        if self.config.use_mock:
            self.bus = None
            return

        if not HAS_LEROBOT:
            raise RuntimeError("LeRobot is not installed. Use mock mode or install lerobot first.")

        if not self.config.port:
            raise ValueError("hardware serial port is required when use_mock=false")

        norm_mode = MotorNormMode.DEGREES if self.config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "pan": Motor(self.config.pan_id, self.config.motor_model, norm_mode),
                "tilt": Motor(self.config.tilt_id, self.config.motor_model, norm_mode),
            },
            calibration=getattr(self, "calibration", {}),
        )

    def _build_camera_backend(self) -> None:
        source = self.config.camera_source.lower()

        if source == "synthetic":
            self.cameras = {
                self.config.camera_key: SyntheticMovingBlobCamera(
                    width=self.config.camera_width,
                    height=self.config.camera_height,
                    fps=self.config.camera_fps,
                )
            }
            return

        if source == "opencv":
            self.cameras = {
                self.config.camera_key: OpenCVCameraSource(
                    index=self.config.camera_index,
                    width=self.config.camera_width,
                    height=self.config.camera_height,
                    fps=self.config.camera_fps,
                )
            }
            return

        if source == "lerobot":
            if not HAS_LEROBOT:
                raise RuntimeError("camera.source=lerobot requested but lerobot is unavailable")
            camera_cfgs = dict(self.config.cameras)
            if self.config.camera_key not in camera_cfgs:
                camera_cfgs[self.config.camera_key] = OpenCVCameraConfig(
                    index_or_path=self.config.camera_index,
                    fps=self.config.camera_fps,
                    width=self.config.camera_width,
                    height=self.config.camera_height,
                )
            self.cameras = make_cameras_from_configs(camera_cfgs)
            return

        raise ValueError(f"Unsupported camera source: {self.config.camera_source}")

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        if self.bus is not None:
            self.bus.connect()
            if calibrate and not self.is_calibrated:
                self.calibrate()
            self.configure()

        for cam in self.cameras.values():
            cam.connect()

        self._connected = True

    def calibrate(self) -> None:
        if self.bus is None:
            return

        if not getattr(self, "calibration", None):
            raise RuntimeError(
                "No calibration data loaded. Calibrate motors with your existing calibration workflow, "
                "then rerun with calibration file available."
            )

        self.bus.write_calibration(self.calibration)

    def configure(self) -> None:
        if self.bus is None:
            return

        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor_name in self.bus.motors:
                self.bus.write("Operating_Mode", motor_name, OperatingMode.POSITION.value)
                if self.config.position_p_coefficient is not None:
                    self.bus.write("P_Coefficient", motor_name, self.config.position_p_coefficient)
                if self.config.position_i_coefficient is not None:
                    self.bus.write("I_Coefficient", motor_name, self.config.position_i_coefficient)
                if self.config.position_d_coefficient is not None:
                    self.bus.write("D_Coefficient", motor_name, self.config.position_d_coefficient)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        obs: dict[str, Any] = {}

        if self.bus is not None:
            self._pan_pos = float(self.bus.read("Present_Position", "pan"))
            self._tilt_pos = float(self.bus.read("Present_Position", "tilt"))

        obs["pan.pos"] = float(self._pan_pos)
        obs["tilt.pos"] = float(self._tilt_pos)

        for camera_key, cam in self.cameras.items():
            if hasattr(cam, "async_read"):
                obs[camera_key] = cam.async_read()
            else:
                obs[camera_key] = cam.read()

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        pan_target = float(action.get("pan.pos", self._pan_pos))
        tilt_target = float(action.get("tilt.pos", self._tilt_pos))

        pan_target = clamp(pan_target, self.config.pan_limits_deg[0], self.config.pan_limits_deg[1])
        tilt_target = clamp(tilt_target, self.config.tilt_limits_deg[0], self.config.tilt_limits_deg[1])

        if self.bus is not None:
            self.bus.write("Goal_Position", "pan", pan_target)
            self.bus.write("Goal_Position", "tilt", tilt_target)

        self._pan_pos = pan_target
        self._tilt_pos = tilt_target
        return {"pan.pos": pan_target, "tilt.pos": tilt_target}

    def disconnect(self) -> None:
        if not self._connected:
            return

        for cam in self.cameras.values():
            cam.disconnect()

        if self.bus is not None:
            self.bus.disconnect(self.config.disable_torque_on_disconnect)

        self._connected = False

    def __str__(self) -> str:
        return f"{self.config.id} XLERobotHead"


def build_head_config(
    *,
    robot_id: str,
    calibration_dir: str | None,
    use_mock: bool,
    serial_port: str | None,
    motor_model: str,
    pan_id: int,
    tilt_id: int,
    use_degrees: bool,
    disable_torque_on_disconnect: bool,
    max_relative_target: float | None,
    pan_limits_deg: tuple[float, float],
    tilt_limits_deg: tuple[float, float],
    position_p_coefficient: int | None,
    position_i_coefficient: int | None,
    position_d_coefficient: int | None,
    camera_key: str,
    camera_source: str,
    camera_index: int,
    camera_width: int,
    camera_height: int,
    camera_fps: int,
) -> XLERobotHeadConfig:
    return XLERobotHeadConfig(
        id=robot_id,
        calibration_dir=Path(calibration_dir) if calibration_dir else None,
        use_mock=use_mock,
        port=serial_port,
        motor_model=motor_model,
        pan_id=pan_id,
        tilt_id=tilt_id,
        use_degrees=use_degrees,
        disable_torque_on_disconnect=disable_torque_on_disconnect,
        max_relative_target=max_relative_target,
        pan_limits_deg=pan_limits_deg,
        tilt_limits_deg=tilt_limits_deg,
        position_p_coefficient=position_p_coefficient,
        position_i_coefficient=position_i_coefficient,
        position_d_coefficient=position_d_coefficient,
        camera_key=camera_key,
        camera_source=camera_source,
        camera_index=camera_index,
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fps=camera_fps,
        cameras={},
    )
