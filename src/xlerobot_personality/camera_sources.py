from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


class CameraSourceError(RuntimeError):
    pass


@dataclass
class OpenCVCameraSource:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30

    def __post_init__(self) -> None:
        self.cap: Any | None = None

    @property
    def is_connected(self) -> bool:
        return self.cap is not None

    def connect(self) -> None:
        if cv2 is None:
            raise CameraSourceError("opencv-python is required for OpenCVCameraSource")
        if self.cap is not None:
            return

        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise CameraSourceError(f"Failed to open camera index {self.index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
        cap.set(cv2.CAP_PROP_FPS, int(self.fps))
        self.cap = cap

    def read(self) -> np.ndarray:
        if self.cap is None:
            raise CameraSourceError("Camera not connected")

        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise CameraSourceError("Failed to read frame from camera")
        return frame

    def async_read(self) -> np.ndarray:
        return self.read()

    def disconnect(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


@dataclass
class SyntheticMovingBlobCamera:
    width: int = 640
    height: int = 480
    fps: int = 30
    blob_radius: int = 24

    def __post_init__(self) -> None:
        self._connected = False
        self._phase = 0.0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def read(self) -> np.ndarray:
        if not self._connected:
            raise CameraSourceError("Synthetic camera not connected")

        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :] = (20, 20, 20)

        self._phase += 0.08
        x = int((self.width / 2) + (self.width * 0.35) * math.sin(self._phase))
        y = int((self.height / 2) + (self.height * 0.20) * math.cos(self._phase * 1.4))

        if cv2 is not None:
            cv2.circle(frame, (x, y), self.blob_radius, (0, 220, 255), -1)
            cv2.circle(frame, (self.width // 2, self.height // 2), 5, (255, 255, 255), 1)
        else:
            x0 = max(0, x - self.blob_radius)
            y0 = max(0, y - self.blob_radius)
            x1 = min(self.width, x + self.blob_radius)
            y1 = min(self.height, y + self.blob_radius)
            frame[y0:y1, x0:x1] = (0, 220, 255)

        return frame

    def async_read(self) -> np.ndarray:
        return self.read()

    def disconnect(self) -> None:
        self._connected = False
