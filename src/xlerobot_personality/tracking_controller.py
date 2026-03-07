from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Any

import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class PIDController:
    kp: float
    ki: float
    kd: float
    integral_limit: float

    def __post_init__(self) -> None:
        self._integral = 0.0
        self._prev_error: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def update(self, error: float, dt: float) -> float:
        if dt <= 0:
            return self.kp * error

        self._integral += error * dt
        self._integral = clamp(self._integral, -self.integral_limit, self.integral_limit)

        derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
        self._prev_error = error
        return (self.kp * error) + (self.ki * self._integral) + (self.kd * derivative)


@dataclass
class TrackedTarget:
    cx: float
    cy: float
    area: float
    bbox: tuple[int, int, int, int]
    label: str = "target"
    confidence: float = 1.0
    track_id: int | None = None


@dataclass
class DetectedObject:
    bbox: tuple[int, int, int, int]
    label: str
    confidence: float
    aim_point: tuple[float, float] | None = None

    @property
    def area(self) -> float:
        return float(self.bbox[2] * self.bbox[3])

    @property
    def cx(self) -> float:
        x, _, w, _ = self.bbox
        return x + (w / 2.0)

    @property
    def cy(self) -> float:
        _, y, _, h = self.bbox
        return y + (h / 2.0)


class HeadState(str, Enum):
    IDLE_SCAN = "IDLE_SCAN"
    TRACKING = "TRACKING"
    REACQUIRE = "REACQUIRE"
    INTERACTING = "INTERACTING"
    MANUAL = "MANUAL"


class PersonalityFSM:
    def __init__(self, target_timeout_s: float, reacquire_duration_s: float):
        self.target_timeout_s = target_timeout_s
        self.reacquire_duration_s = reacquire_duration_s
        self.last_target_ts = 0.0
        self.interacting_until = 0.0
        self.state = HeadState.IDLE_SCAN

    def set_interacting(self, hold_s: float) -> None:
        now = time.time()
        self.interacting_until = max(self.interacting_until, now + hold_s)

    def update(self, has_target: bool, now: float) -> HeadState:
        if now < self.interacting_until:
            self.state = HeadState.INTERACTING
            return self.state

        if has_target:
            self.last_target_ts = now
            self.state = HeadState.TRACKING
            return self.state

        since_last_target = now - self.last_target_ts
        if since_last_target <= self.target_timeout_s + self.reacquire_duration_s:
            self.state = HeadState.REACQUIRE
        else:
            self.state = HeadState.IDLE_SCAN

        return self.state


class MotionTracker:
    def __init__(self, min_area_px: int = 700):
        self.min_area_px = min_area_px
        self._bg = None
        if cv2 is not None:
            self._bg = cv2.createBackgroundSubtractorMOG2(history=250, varThreshold=24, detectShadows=False)

    def update(self, frame_bgr: np.ndarray) -> TrackedTarget | None:
        if cv2 is None or self._bg is None:
            return None

        fg = self._bg.apply(frame_bgr)
        _, fg = cv2.threshold(fg, 220, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), dtype=np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, kernel, iterations=2)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < self.min_area_px:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        cx = x + (w / 2.0)
        cy = y + (h / 2.0)
        return TrackedTarget(cx=cx, cy=cy, area=area, bbox=(x, y, w, h), label="motion", confidence=1.0)


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    union = (aw * ah) + (bw * bh) - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def center_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax = a[0] + (a[2] / 2.0)
    ay = a[1] + (a[3] / 2.0)
    bx = b[0] + (b[2] / 2.0)
    by = b[1] + (b[3] / 2.0)
    return math.hypot(ax - bx, ay - by)


def nms_detections(detections: list[DetectedObject], iou_threshold: float = 0.35) -> list[DetectedObject]:
    if len(detections) <= 1:
        return detections

    remaining = sorted(detections, key=lambda det: (det.confidence, det.area), reverse=True)
    selected: list[DetectedObject] = []
    while remaining:
        current = remaining.pop(0)
        selected.append(current)
        remaining = [det for det in remaining if bbox_iou(current.bbox, det.bbox) < iou_threshold]
    return selected


class HOGPersonDetector:
    def __init__(self, confidence_threshold: float = 0.2, max_dim_px: int = 480, min_area_px: int = 700):
        self.confidence_threshold = float(confidence_threshold)
        self.max_dim_px = max(160, int(max_dim_px))
        self.min_area_px = int(min_area_px)
        self._hog = None
        if cv2 is not None:
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedObject]:
        if cv2 is None or self._hog is None:
            return []

        height, width = frame_bgr.shape[:2]
        scale = 1.0
        longest = max(height, width)
        resized = frame_bgr
        if longest > self.max_dim_px:
            scale = self.max_dim_px / float(longest)
            resized = cv2.resize(
                frame_bgr,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_LINEAR,
            )

        rects, weights = self._hog.detectMultiScale(
            resized,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        if len(rects) == 0:
            return []

        raw_weights = np.array(weights).reshape(-1) if len(weights) else np.ones(len(rects), dtype=np.float32)
        detections: list[DetectedObject] = []
        inv_scale = 1.0 / scale
        for rect, weight in zip(rects, raw_weights, strict=False):
            x, y, w, h = [int(v) for v in rect]
            x = int(round(x * inv_scale))
            y = int(round(y * inv_scale))
            w = int(round(w * inv_scale))
            h = int(round(h * inv_scale))
            if (w * h) < self.min_area_px:
                continue
            confidence = float(weight)
            if confidence < self.confidence_threshold:
                continue
            detections.append(DetectedObject(bbox=(x, y, w, h), label="person", confidence=confidence))

        return nms_detections(detections)


class YOLOPersonDetector:
    def __init__(
        self,
        model_name: str,
        confidence_threshold: float = 0.2,
        imgsz: int = 640,
        device: str = "cpu",
        min_area_px: int = 700,
    ):
        self.model_name = model_name
        self.confidence_threshold = float(confidence_threshold)
        self.imgsz = max(160, int(imgsz))
        self.device = device
        self.min_area_px = int(min_area_px)

        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "YOLO tracking requires the ultralytics package. Install it with: pip install -e \".[yolo]\""
            ) from exc

        try:
            self._model = YOLO(model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLO model '{model_name}': {exc}") from exc

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedObject]:
        frame_h, frame_w = frame_bgr.shape[:2]
        try:
            results = self._model.predict(
                source=frame_bgr,
                classes=[0],
                conf=self.confidence_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"YOLO inference failed: {exc}") from exc

        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = getattr(result, "names", None)
        if not isinstance(names, dict):
            names = {}

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        detections: list[DetectedObject] = []
        for coords, confidence, class_id in zip(xyxy, confs, classes, strict=False):
            if int(class_id) != 0:
                continue
            x1, y1, x2, y2 = [int(round(val)) for val in coords]
            x1 = max(0, min(frame_w - 1, x1))
            y1 = max(0, min(frame_h - 1, y1))
            x2 = max(0, min(frame_w, x2))
            y2 = max(0, min(frame_h, y2))
            x = x1
            y = y1
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if (w * h) < self.min_area_px:
                continue
            label = str(names.get(int(class_id), "person"))
            detections.append(
                DetectedObject(
                    bbox=(x, y, w, h),
                    label=label,
                    confidence=float(confidence),
                )
            )

        return nms_detections(detections)


class YOLOPosePersonDetector:
    def __init__(
        self,
        model_name: str,
        confidence_threshold: float = 0.2,
        imgsz: int = 640,
        device: str = "cpu",
        min_area_px: int = 700,
    ):
        self.model_name = model_name
        self.confidence_threshold = float(confidence_threshold)
        self.imgsz = max(160, int(imgsz))
        self.device = device
        self.min_area_px = int(min_area_px)

        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "YOLO pose tracking requires the ultralytics package. Install it with: pip install -e \".[yolo]\""
            ) from exc

        try:
            self._model = YOLO(model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLO pose model '{model_name}': {exc}") from exc

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedObject]:
        frame_h, frame_w = frame_bgr.shape[:2]
        try:
            results = self._model.predict(
                source=frame_bgr,
                conf=self.confidence_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"YOLO pose inference failed: {exc}") from exc

        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)
        if boxes is None or len(boxes) == 0 or keypoints is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        keypoint_data = keypoints.data.cpu().numpy()

        detections: list[DetectedObject] = []
        for coords, confidence, class_id, pose in zip(xyxy, confs, classes, keypoint_data, strict=False):
            if int(class_id) != 0:
                continue

            x1, y1, x2, y2 = [int(round(val)) for val in coords]
            x1 = max(0, min(frame_w - 1, x1))
            y1 = max(0, min(frame_h - 1, y1))
            x2 = max(0, min(frame_w, x2))
            y2 = max(0, min(frame_h, y2))
            x = x1
            y = y1
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if (w * h) < self.min_area_px:
                continue

            aim_point = self._torso_aim_point(pose, bbox=(x, y, w, h), frame_shape=(frame_h, frame_w))
            detections.append(
                DetectedObject(
                    bbox=(x, y, w, h),
                    label="person",
                    confidence=float(confidence),
                    aim_point=aim_point,
                )
            )

        return nms_detections(detections)

    def _torso_aim_point(
        self,
        pose: np.ndarray,
        bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, int],
    ) -> tuple[float, float]:
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape

        def kp(index: int, min_conf: float = 0.25) -> tuple[float, float] | None:
            if index >= pose.shape[0]:
                return None
            px, py, conf = pose[index]
            if float(conf) < min_conf:
                return None
            return float(px), float(py)

        def midpoint(a: tuple[float, float] | None, b: tuple[float, float] | None) -> tuple[float, float] | None:
            if a is None or b is None:
                return None
            return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

        left_shoulder = kp(5)
        right_shoulder = kp(6)
        left_hip = kp(11)
        right_hip = kp(12)

        shoulder_center = midpoint(left_shoulder, right_shoulder)
        hip_center = midpoint(left_hip, right_hip)

        if shoulder_center is not None and hip_center is not None:
            aim_x = (0.35 * shoulder_center[0]) + (0.65 * hip_center[0])
            aim_y = (0.35 * shoulder_center[1]) + (0.65 * hip_center[1])
        elif hip_center is not None:
            aim_x, aim_y = hip_center
        elif shoulder_center is not None:
            aim_x = shoulder_center[0]
            aim_y = shoulder_center[1] + (0.38 * h)
        else:
            aim_x = x + (0.5 * w)
            aim_y = y + (0.58 * h)

        aim_x = clamp(aim_x, 0.0, float(max(0, frame_w - 1)))
        aim_y = clamp(aim_y, 0.0, float(max(0, frame_h - 1)))
        return aim_x, aim_y


class DetectorTracker:
    def __init__(
        self,
        backend: str,
        min_area_px: int,
        detector_confidence: float,
        acquire_confirm_frames: int,
        detector_interval: int,
        detector_max_dim_px: int,
        track_hold_s: float,
        track_match_distance_px: float,
        target_selection_mode: str,
        person_target_x_ratio: float,
        person_target_y_ratio: float,
        person_min_full_body_aspect_ratio: float,
        person_top_frame_ratio: float,
        person_top_framing_gain: float,
        person_closeup_height_ratio: float,
        person_closeup_top_gain: float,
        yolo_model: str,
        yolo_pose_model: str,
        yolo_device: str,
        yolo_imgsz: int,
    ):
        backend = backend.lower().strip()
        if backend not in {"motion", "hog_person", "yolo_person", "yolo_pose_person"}:
            raise ValueError(f"Unsupported tracking backend: {backend}")

        self.backend = backend
        self.track_hold_s = max(0.0, float(track_hold_s))
        self.track_match_distance_px = max(20.0, float(track_match_distance_px))
        self.acquire_confirm_frames = max(1, int(acquire_confirm_frames))
        self.detector_interval = max(1, int(detector_interval))
        self.target_selection_mode = str(target_selection_mode).strip().lower()
        if self.target_selection_mode not in {"sticky", "most_centered"}:
            raise ValueError(f"Unsupported target selection mode: {target_selection_mode}")
        self.person_target_x_ratio = clamp(float(person_target_x_ratio), 0.2, 0.8)
        self.person_target_y_ratio = clamp(float(person_target_y_ratio), 0.25, 0.9)
        self.person_min_full_body_aspect_ratio = clamp(float(person_min_full_body_aspect_ratio), 1.5, 4.0)
        self._frame_index = 0
        self._last_target: TrackedTarget | None = None
        self._last_target_ts = 0.0
        self._candidate_target: TrackedTarget | None = None
        self._candidate_streak = 0
        self._next_track_id = 1

        self._motion_tracker = MotionTracker(min_area_px=min_area_px) if backend == "motion" else None
        self._person_detector = (
            HOGPersonDetector(
                confidence_threshold=detector_confidence,
                max_dim_px=detector_max_dim_px,
                min_area_px=min_area_px,
            )
            if backend == "hog_person"
            else None
        )
        self._yolo_detector = (
            YOLOPersonDetector(
                model_name=yolo_model,
                confidence_threshold=detector_confidence,
                imgsz=yolo_imgsz,
                device=yolo_device,
                min_area_px=min_area_px,
            )
            if backend == "yolo_person"
            else None
        )
        self._yolo_pose_detector = (
            YOLOPosePersonDetector(
                model_name=yolo_pose_model,
                confidence_threshold=detector_confidence,
                imgsz=yolo_imgsz,
                device=yolo_device,
                min_area_px=min_area_px,
            )
            if backend == "yolo_pose_person"
            else None
        )

    def update(self, frame_bgr: np.ndarray, now: float) -> TrackedTarget | None:
        if self.backend == "motion":
            return self._motion_tracker.update(frame_bgr) if self._motion_tracker is not None else None

        self._frame_index += 1
        should_detect = self._last_target is None or (self._frame_index % self.detector_interval == 0)
        detector = self._person_detector
        if detector is None:
            detector = self._yolo_detector
        if detector is None:
            detector = self._yolo_pose_detector
        if should_detect and detector is not None:
            detections = detector.detect(frame_bgr)
            if detections:
                target = self._select_target(detections, frame_bgr.shape[:2])
                if self._last_target is not None and target.track_id == self._last_target.track_id:
                    self._last_target = target
                    self._last_target_ts = now
                    self._candidate_target = None
                    self._candidate_streak = 0
                    return target

                if self._candidate_matches(target):
                    self._candidate_target = target
                    self._candidate_streak += 1
                else:
                    self._candidate_target = target
                    self._candidate_streak = 1

                if self._candidate_streak >= self.acquire_confirm_frames:
                    self._last_target = self._candidate_target
                    self._last_target_ts = now
                    self._candidate_target = None
                    self._candidate_streak = 0
                    return self._last_target
            else:
                self._candidate_target = None
                self._candidate_streak = 0

        if self._last_target is not None and (now - self._last_target_ts) <= self.track_hold_s:
            return self._last_target

        self._last_target = None
        return None

    def _candidate_matches(self, target: TrackedTarget) -> bool:
        if self._candidate_target is None:
            return False

        if self._candidate_target.label != target.label:
            return False

        iou = bbox_iou(self._candidate_target.bbox, target.bbox)
        dist = center_distance(self._candidate_target.bbox, target.bbox)
        area_ratio = target.area / max(1.0, self._candidate_target.area)
        if area_ratio < 0.35 or area_ratio > 2.8:
            return False
        return iou > 0.08 or dist <= (self.track_match_distance_px * 0.45)

    def _select_target(self, detections: list[DetectedObject], frame_shape: tuple[int, int]) -> TrackedTarget:
        height, width = frame_shape
        frame_area = float(max(1, width * height))
        image_center = (width / 2.0, height / 2.0)
        matched = self._match_previous_target(detections)

        if self.target_selection_mode == "sticky":
            if matched is not None:
                detection, track_id = matched
                return self._make_target(detection, track_id, frame_shape)

        best_detection = max(
            detections,
            key=lambda det: self._score_new_target(det, image_center, frame_area),
        )
        if self.target_selection_mode == "most_centered" and matched is not None:
            matched_detection, track_id = matched
            best_score = self._score_new_target(best_detection, image_center, frame_area)
            matched_score = self._score_new_target(matched_detection, image_center, frame_area)
            switch_margin = 0.18
            if matched_score >= (best_score - switch_margin):
                return self._make_target(matched_detection, track_id, frame_shape)

        track_id = self._next_track_id
        self._next_track_id += 1
        return self._make_target(best_detection, track_id, frame_shape)

    def _match_previous_target(self, detections: list[DetectedObject]) -> tuple[DetectedObject, int] | None:
        if self._last_target is None or self._last_target.track_id is None:
            return None

        last_bbox = self._last_target.bbox
        last_area = max(1.0, self._last_target.area)
        candidates: list[tuple[float, DetectedObject]] = []
        for detection in detections:
            iou = bbox_iou(last_bbox, detection.bbox)
            dist = center_distance(last_bbox, detection.bbox)
            area_ratio = detection.area / last_area
            max_match_distance = self.track_match_distance_px * 0.55
            if area_ratio < 0.35 or area_ratio > 2.8:
                continue
            if iou < 0.08 and dist > max_match_distance:
                continue
            score = (3.0 * iou) + max(0.0, 1.0 - (dist / max_match_distance)) + min(detection.confidence, 2.0)
            candidates.append((score, detection))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1], self._last_target.track_id

    def _score_new_target(
        self,
        detection: DetectedObject,
        image_center: tuple[float, float],
        frame_area: float,
    ) -> float:
        dist_to_center = math.hypot(detection.cx - image_center[0], detection.cy - image_center[1])
        center_bonus = max(0.0, 1.0 - (dist_to_center / max(1.0, math.hypot(*image_center))))
        area_bonus = min(1.0, detection.area / (frame_area * 0.22))
        if self.target_selection_mode == "most_centered":
            return (2.2 * center_bonus) + (0.35 * min(detection.confidence, 2.0)) + (0.15 * area_bonus)
        return (1.5 * detection.confidence) + (0.8 * center_bonus) + (0.6 * area_bonus)

    def _make_target(
        self,
        detection: DetectedObject,
        track_id: int,
        frame_shape: tuple[int, int],
    ) -> TrackedTarget:
        x, y, w, h = detection.bbox
        target_x_ratio = 0.5
        target_y_ratio = 0.5
        target_height = float(h)
        if detection.aim_point is not None:
            target_cx, target_cy = detection.aim_point
        else:
            if detection.label == "person":
                target_x_ratio = self.person_target_x_ratio
                target_y_ratio = self.person_target_y_ratio
                inferred_full_body_height = max(float(h), float(w) * self.person_min_full_body_aspect_ratio)
                target_height = inferred_full_body_height

            frame_h, frame_w = frame_shape
            target_cx = clamp(x + (w * target_x_ratio), 0.0, float(max(0, frame_w - 1)))
            target_cy = clamp(y + (target_height * target_y_ratio), 0.0, float(max(0, frame_h - 1)))
        return TrackedTarget(
            cx=target_cx,
            cy=target_cy,
            area=detection.area,
            bbox=(x, y, w, h),
            label=detection.label,
            confidence=detection.confidence,
            track_id=track_id,
        )


@dataclass
class TrackingReport:
    timestamp: float
    state: HeadState
    target: TrackedTarget | None
    pan_obs: float
    tilt_obs: float
    pan_cmd: float
    tilt_cmd: float
    frame: np.ndarray
    target_frame_point: tuple[float, float] | None = None

    @property
    def target_found(self) -> bool:
        return self.target is not None


class PanTiltTrackingController:
    def __init__(
        self,
        robot: Any,
        camera_key: str,
        tracking_backend: str,
        hfov_deg: float,
        vfov_deg: float,
        frame_target_x_ratio: float,
        frame_target_y_ratio: float,
        tilt_tracking_sign: float,
        tilt_error_gain: float,
        pan_limits_deg: tuple[float, float],
        tilt_limits_deg: tuple[float, float],
        max_step_deg: float,
        max_speed_deg_s: float,
        manual_speed_deg_s: float,
        command_smoothing_s: float,
        transition_smoothing_s: float,
        detector_confidence: float,
        acquire_confirm_frames: int,
        detector_interval: int,
        detector_max_dim_px: int,
        track_hold_s: float,
        track_match_distance_px: float,
        target_selection_mode: str,
        person_target_x_ratio: float,
        person_target_y_ratio: float,
        person_min_full_body_aspect_ratio: float,
        person_top_frame_ratio: float,
        person_top_framing_gain: float,
        person_closeup_height_ratio: float,
        person_closeup_top_gain: float,
        yolo_model: str,
        yolo_pose_model: str,
        yolo_device: str,
        yolo_imgsz: int,
        target_timeout_s: float,
        reacquire_duration_s: float,
        scan_speed_deg_s: float,
        scan_tilt_center_deg: float,
        scan_tilt_amplitude_deg: float,
        scan_nod_hz: float,
        pan_pid: PIDController,
        tilt_pid: PIDController,
        min_area_px: int,
    ):
        self.robot = robot
        self.camera_key = camera_key
        self.tracking_backend = tracking_backend
        self.hfov_deg = hfov_deg
        self.vfov_deg = vfov_deg
        self.frame_target_x_ratio = clamp(float(frame_target_x_ratio), 0.2, 0.8)
        self.frame_target_y_ratio = clamp(float(frame_target_y_ratio), 0.25, 0.85)
        self.tilt_tracking_sign = -1.0 if float(tilt_tracking_sign) < 0 else 1.0
        self.tilt_error_gain = clamp(float(tilt_error_gain), 0.1, 2.0)
        self.person_top_frame_ratio = clamp(float(person_top_frame_ratio), 0.05, 0.45)
        self.person_top_framing_gain = clamp(float(person_top_framing_gain), 0.0, 1.5)
        self.person_closeup_height_ratio = clamp(float(person_closeup_height_ratio), 0.2, 0.9)
        self.person_closeup_top_gain = clamp(float(person_closeup_top_gain), 0.0, 3.0)
        self.pan_limits_deg = pan_limits_deg
        self.tilt_limits_deg = tilt_limits_deg
        self.max_step_deg = max_step_deg
        self.max_speed_deg_s = max_speed_deg_s
        self.manual_speed_deg_s = clamp(float(manual_speed_deg_s), 5.0, 180.0)
        self.command_smoothing_s = max(0.0, float(command_smoothing_s))
        self.transition_smoothing_s = max(0.0, float(transition_smoothing_s))
        self.scan_speed_deg_s = scan_speed_deg_s
        self.scan_tilt_center_deg = scan_tilt_center_deg
        self.scan_tilt_amplitude_deg = scan_tilt_amplitude_deg
        self.scan_nod_hz = scan_nod_hz

        self.pan_pid = pan_pid
        self.tilt_pid = tilt_pid
        self.tracker = DetectorTracker(
            backend=tracking_backend,
            min_area_px=min_area_px,
            detector_confidence=detector_confidence,
            acquire_confirm_frames=acquire_confirm_frames,
            detector_interval=detector_interval,
            detector_max_dim_px=detector_max_dim_px,
            track_hold_s=track_hold_s,
            track_match_distance_px=track_match_distance_px,
            target_selection_mode=target_selection_mode,
            person_target_x_ratio=person_target_x_ratio,
            person_target_y_ratio=person_target_y_ratio,
            person_min_full_body_aspect_ratio=person_min_full_body_aspect_ratio,
            person_top_frame_ratio=person_top_frame_ratio,
            person_top_framing_gain=person_top_framing_gain,
            person_closeup_height_ratio=person_closeup_height_ratio,
            person_closeup_top_gain=person_closeup_top_gain,
            yolo_model=yolo_model,
            yolo_pose_model=yolo_pose_model,
            yolo_device=yolo_device,
            yolo_imgsz=yolo_imgsz,
        )
        self.fsm = PersonalityFSM(target_timeout_s=target_timeout_s, reacquire_duration_s=reacquire_duration_s)

        self._scan_direction = 1.0
        self._last_ts = time.perf_counter()
        self.tracking_enabled = True
        self._manual_pan_delta = 0.0
        self._manual_tilt_delta = 0.0
        self._manual_pan_axis = 0.0
        self._manual_tilt_axis = 0.0
        self._scan_pan_target: float | None = None
        self._last_pan_cmd: float | None = None
        self._last_tilt_cmd: float | None = None
        self._last_state: HeadState | None = None
        self._transition_until = 0.0

    def set_tracking_enabled(self, enabled: bool) -> bool:
        enabled = bool(enabled)
        if self.tracking_enabled != enabled:
            self.pan_pid.reset()
            self.tilt_pid.reset()
        self.tracking_enabled = enabled
        return self.tracking_enabled

    def set_manual_drive(self, pan_axis: float = 0.0, tilt_axis: float = 0.0) -> None:
        self._manual_pan_axis = clamp(float(pan_axis), -1.0, 1.0)
        self._manual_tilt_axis = clamp(float(tilt_axis), -1.0, 1.0)

    def stop_manual_drive(self) -> None:
        self._manual_pan_axis = 0.0
        self._manual_tilt_axis = 0.0

    def queue_manual_delta(self, pan_delta: float = 0.0, tilt_delta: float = 0.0) -> None:
        self._manual_pan_delta += float(pan_delta)
        self._manual_tilt_delta += float(tilt_delta)

    def set_manual_speed_deg_s(self, speed_deg_s: float) -> float:
        self.manual_speed_deg_s = clamp(float(speed_deg_s), 5.0, 180.0)
        return self.manual_speed_deg_s

    def clear_manual_delta(self) -> tuple[float, float]:
        pan_delta = self._manual_pan_delta
        tilt_delta = self._manual_tilt_delta
        self._manual_pan_delta = 0.0
        self._manual_tilt_delta = 0.0
        return pan_delta, tilt_delta

    def set_interacting(self, hold_s: float) -> None:
        self.fsm.set_interacting(hold_s)

    def _apply_step_limits(self, obs_val: float, target_val: float, dt: float) -> float:
        speed_limited_step = self.max_speed_deg_s * max(1e-3, dt)
        step_limit = min(self.max_step_deg, speed_limited_step)
        delta = clamp(target_val - obs_val, -step_limit, step_limit)
        return obs_val + delta

    def _command_anchor(self, obs_val: float, last_cmd: float | None) -> float:
        if last_cmd is None:
            return obs_val

        resync_threshold = max(8.0, self.max_step_deg * 4.0)
        if abs(last_cmd - obs_val) > resync_threshold:
            return obs_val

        return last_cmd

    def _smooth_target(
        self,
        current_cmd: float,
        target_val: float,
        dt: float,
        smoothing_s: float | None = None,
    ) -> float:
        smoothing_s = self.command_smoothing_s if smoothing_s is None else max(0.0, float(smoothing_s))
        if smoothing_s <= 1e-6:
            return target_val

        alpha = 1.0 - math.exp(-max(1e-3, dt) / smoothing_s)
        return current_cmd + ((target_val - current_cmd) * alpha)

    def _compute_scan_target(self, pan_anchor: float, now: float, dt: float, speed_scale: float) -> tuple[float, float]:
        if self._scan_pan_target is None:
            self._scan_pan_target = clamp(pan_anchor, self.pan_limits_deg[0], self.pan_limits_deg[1])

        pan_target = self._scan_pan_target + (self._scan_direction * self.scan_speed_deg_s * speed_scale * dt)
        if pan_target <= self.pan_limits_deg[0]:
            self._scan_direction = 1.0
            pan_target = self.pan_limits_deg[0]
        elif pan_target >= self.pan_limits_deg[1]:
            self._scan_direction = -1.0
            pan_target = self.pan_limits_deg[1]

        self._scan_pan_target = pan_target

        tilt_target = self.scan_tilt_center_deg + (
            self.scan_tilt_amplitude_deg * math.sin((2.0 * math.pi * self.scan_nod_hz) * now)
        )
        return pan_target, tilt_target

    def step(self, observation: dict[str, Any]) -> TrackingReport:
        now = time.time()
        t_now = time.perf_counter()
        dt = max(1e-3, t_now - self._last_ts)
        self._last_ts = t_now

        frame = observation[self.camera_key]
        pan_obs = float(observation.get("pan.pos", 0.0))
        tilt_obs = float(observation.get("tilt.pos", 0.0))
        pan_anchor = self._command_anchor(pan_obs, self._last_pan_cmd)
        tilt_anchor = self._command_anchor(tilt_obs, self._last_tilt_cmd)

        target = self.tracker.update(frame, now=now)
        manual_pan_delta, manual_tilt_delta = self.clear_manual_delta()
        manual_pan_delta += self._manual_pan_axis * self.manual_speed_deg_s * dt
        manual_tilt_delta += self._manual_tilt_axis * self.manual_speed_deg_s * dt
        has_manual_command = abs(manual_pan_delta) > 1e-6 or abs(manual_tilt_delta) > 1e-6

        pan_cmd = pan_anchor
        tilt_cmd = tilt_anchor

        if has_manual_command:
            state = HeadState.MANUAL
            self.pan_pid.reset()
            self.tilt_pid.reset()
            pan_cmd = pan_anchor + manual_pan_delta
            tilt_cmd = tilt_anchor + manual_tilt_delta

        elif not self.tracking_enabled:
            state = HeadState.MANUAL
            self.pan_pid.reset()
            self.tilt_pid.reset()

        else:
            state = self.fsm.update(has_target=target is not None, now=now)

        if state != self._last_state:
            self.pan_pid.reset()
            self.tilt_pid.reset()
            self._transition_until = now + self.transition_smoothing_s
            self._last_state = state

        if state not in {HeadState.REACQUIRE, HeadState.IDLE_SCAN}:
            self._scan_pan_target = pan_anchor

        if state == HeadState.TRACKING and target is not None:
            h, w = frame.shape[:2]
            target_frame_x = w * self.frame_target_x_ratio
            target_frame_y = h * self.frame_target_y_ratio
            ex = (target.cx - target_frame_x) / (w / 2.0)
            ey = (target.cy - target_frame_y) / (h / 2.0)
            if target.label == "person":
                bbox_top = float(target.bbox[1])
                bbox_height_ratio = float(target.bbox[3]) / max(1.0, float(h))
                desired_top = h * self.person_top_frame_ratio
                top_ey = (bbox_top - desired_top) / (h / 2.0)
                closeup_ratio = clamp(
                    (bbox_height_ratio - self.person_closeup_height_ratio)
                    / max(1e-3, 1.0 - self.person_closeup_height_ratio),
                    0.0,
                    1.0,
                )
                top_gain = self.person_top_framing_gain * (1.0 + (self.person_closeup_top_gain * closeup_ratio))
                ey += top_gain * top_ey

            yaw_error_deg = ex * (self.hfov_deg / 2.0)
            pitch_error_deg = self.tilt_tracking_sign * ey * (self.vfov_deg / 2.0) * self.tilt_error_gain

            pan_delta = self.pan_pid.update(yaw_error_deg, dt)
            tilt_delta = self.tilt_pid.update(pitch_error_deg, dt)

            pan_cmd = pan_obs + pan_delta
            tilt_cmd = tilt_obs + tilt_delta

        elif state == HeadState.REACQUIRE:
            pan_cmd, tilt_cmd = self._compute_scan_target(pan_anchor, now, dt, speed_scale=1.15)

        elif state == HeadState.IDLE_SCAN:
            pan_cmd, tilt_cmd = self._compute_scan_target(pan_anchor, now, dt, speed_scale=1.0)

        # INTERACTING and MANUAL keep pose stable unless teleop is active.
        smoothing_s = self.command_smoothing_s
        if now < self._transition_until:
            smoothing_s = max(smoothing_s, self.transition_smoothing_s)

        pan_cmd = self._smooth_target(pan_anchor, pan_cmd, dt, smoothing_s=smoothing_s)
        tilt_cmd = self._smooth_target(tilt_anchor, tilt_cmd, dt, smoothing_s=smoothing_s)
        pan_cmd = self._apply_step_limits(pan_anchor, pan_cmd, dt)
        tilt_cmd = self._apply_step_limits(tilt_anchor, tilt_cmd, dt)

        pan_cmd = clamp(pan_cmd, self.pan_limits_deg[0], self.pan_limits_deg[1])
        tilt_cmd = clamp(tilt_cmd, self.tilt_limits_deg[0], self.tilt_limits_deg[1])

        sent = self.robot.send_action({"pan.pos": pan_cmd, "tilt.pos": tilt_cmd})
        self._last_pan_cmd = float(sent["pan.pos"])
        self._last_tilt_cmd = float(sent["tilt.pos"])

        return TrackingReport(
            timestamp=now,
            state=state,
            target=target,
            pan_obs=pan_obs,
            tilt_obs=tilt_obs,
            pan_cmd=float(sent["pan.pos"]),
            tilt_cmd=float(sent["tilt.pos"]),
            frame=frame,
            target_frame_point=(frame.shape[1] * self.frame_target_x_ratio, frame.shape[0] * self.frame_target_y_ratio),
        )


def annotate_tracking_frame(report: TrackingReport, camera_key: str = "head_cam") -> np.ndarray:
    frame = report.frame.copy()
    if cv2 is None:
        return frame

    def draw_panel(
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        lines: list[tuple[str, tuple[int, int, int], float, int]],
    ) -> None:
        row_height = 28
        panel_height = 14 + (row_height * len(lines))
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + panel_height), (18, 26, 40), -1)
        cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
        cv2.rectangle(image, (x, y), (x + width, y + panel_height), (75, 104, 132), 1)

        text_y = y + 24
        for text, color, font_scale, thickness in lines:
            cv2.putText(
                image,
                text,
                (x + 12, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
            )
            text_y += row_height

    h, w = frame.shape[:2]
    cv2.line(frame, (w // 2, 0), (w // 2, h), (80, 80, 80), 1)
    cv2.line(frame, (0, h // 2), (w, h // 2), (80, 80, 80), 1)
    if report.target_frame_point is not None:
        tx = int(round(report.target_frame_point[0]))
        ty = int(round(report.target_frame_point[1]))
        cv2.circle(frame, (tx, ty), 6, (64, 210, 120), 2)
        cv2.line(frame, (tx - 10, ty), (tx + 10, ty), (64, 210, 120), 1)
        cv2.line(frame, (tx, ty - 10), (tx, ty + 10), (64, 210, 120), 1)

    if report.target is not None:
        x, y, bw, bh = report.target.bbox
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 220, 255), 2)
        cv2.circle(frame, (int(report.target.cx), int(report.target.cy)), 5, (0, 220, 255), -1)
        confidence = f" {report.target.confidence:.2f}" if report.target.confidence > 0 else ""
        label = f"{report.target.label}{confidence}".strip()
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_x = max(10, min(x, w - label_w - 18))
        label_y = max(28, y - 10)
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (label_x - 8, label_y - label_h - 8),
            (label_x + label_w + 8, label_y + 6),
            (0, 68, 86),
            -1,
        )
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.putText(
            frame,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
        )

    if report.target is not None:
        target_text = f"target: {report.target.label}"
    else:
        target_text = "target: scanning"
    draw_panel(
        frame,
        x=12,
        y=12,
        width=240,
        lines=[
            (f"state: {report.state.value}", (255, 255, 255), 0.58, 2),
            (target_text, (185, 213, 235), 0.55, 1),
        ],
    )

    draw_panel(
        frame,
        x=max(12, w - 220),
        y=12,
        width=208,
        lines=[
            (f"pan:  {report.pan_cmd:6.1f} deg", (255, 255, 255), 0.56, 2),
            (f"tilt: {report.tilt_cmd:6.1f} deg", (255, 255, 255), 0.56, 2),
        ],
    )

    source_label = f"cam: {camera_key}"
    (source_w, source_h), _ = cv2.getTextSize(source_label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    source_x = 12
    source_y = h - 14
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (source_x - 8, source_y - source_h - 10),
        (source_x + source_w + 8, source_y + 8),
        (22, 30, 44),
        -1,
    )
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.putText(
        frame,
        source_label,
        (source_x, source_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (195, 210, 222),
        1,
    )
    return frame
