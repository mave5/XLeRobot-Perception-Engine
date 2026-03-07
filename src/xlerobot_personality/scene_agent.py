from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass
import json
import math
import os
import threading
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

from .tracking_controller import HeadState, TrackingReport


@dataclass
class SceneMemoryItem:
    timestamp: float
    state: HeadState
    target_found: bool
    pan_cmd: float
    tilt_cmd: float
    target_center: tuple[float, float] | None


@dataclass
class ChatTurn:
    timestamp: float
    user_message: str
    robot_reply: str


class SceneAgent:
    """
    Scene summarization + question answering from recent tracking memory.

    Default behavior is local/rule-based. It can optionally:
    - query an Ollama or OpenAI VLM for scene descriptions
    - use a lightweight Ollama chat model as a "robot brain" for witty dialogue
      grounded in the latest scene + tracking context.
    """

    def __init__(
        self,
        memory_seconds: float,
        use_ollama: bool = False,
        vlm_model: str = "qwen2.5vl:3b",
        ollama_model: str | None = None,
        ollama_url: str = "http://127.0.0.1:11434",
        ollama_timeout_s: float = 20.0,
        ollama_keep_alive: str = "10m",
        max_image_dim_px: int = 672,
        ollama_max_image_dim_px: int | None = None,
        vlm_only_on_significant_change: bool = True,
        vlm_change_threshold: float = 0.08,
        vlm_change_sample_dim_px: int = 96,
        vlm_target_center_change_ratio: float = 0.12,
        vlm_target_area_change_ratio: float = 0.28,
        vlm_force_refresh_s: float = 20.0,
        use_brain: bool = False,
        use_ollama_brain: bool | None = None,
        brain_model: str = "qwen2.5:1.5b",
        ollama_brain_model: str | None = None,
        brain_temperature: float = 0.5,
        ollama_brain_temperature: float | None = None,
        brain_max_tokens: int = 96,
        ollama_brain_max_tokens: int | None = None,
        include_chat_history: bool = True,
        chat_history_max_turns: int = 10,
        use_openai: bool = False,
        openai_model: str = "gpt-4.1-mini",
        openai_api_key_env: str = "OPENAI_API_KEY",
    ):
        self.memory_seconds = memory_seconds
        self.use_ollama = use_ollama
        if ollama_model is not None:
            vlm_model = ollama_model
        self.vlm_model = vlm_model
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_timeout_s = max(1.0, float(ollama_timeout_s))
        self.ollama_keep_alive = ollama_keep_alive
        if ollama_max_image_dim_px is not None:
            max_image_dim_px = ollama_max_image_dim_px
        self.max_image_dim_px = max(0, int(max_image_dim_px))
        self.vlm_only_on_significant_change = bool(vlm_only_on_significant_change)
        self.vlm_change_threshold = max(0.0, float(vlm_change_threshold))
        self.vlm_change_sample_dim_px = max(16, int(vlm_change_sample_dim_px))
        self.vlm_target_center_change_ratio = max(0.0, float(vlm_target_center_change_ratio))
        self.vlm_target_area_change_ratio = max(0.0, float(vlm_target_area_change_ratio))
        self.vlm_force_refresh_s = max(0.0, float(vlm_force_refresh_s))
        if use_ollama_brain is not None:
            use_brain = use_ollama_brain
        self.use_brain = bool(use_brain)
        if ollama_brain_model is not None:
            brain_model = ollama_brain_model
        self.brain_model = brain_model
        if ollama_brain_temperature is not None:
            brain_temperature = ollama_brain_temperature
        self.brain_temperature = max(0.0, min(2.0, float(brain_temperature)))
        if ollama_brain_max_tokens is not None:
            brain_max_tokens = ollama_brain_max_tokens
        self.brain_max_tokens = max(16, int(brain_max_tokens))
        self.include_chat_history = bool(include_chat_history)
        self.chat_history_max_turns = max(0, int(chat_history_max_turns))
        self.use_openai = use_openai
        self.openai_model = openai_model
        self.openai_api_key_env = openai_api_key_env

        self._memory: deque[SceneMemoryItem] = deque()
        self._chat_history: deque[ChatTurn] = deque()
        self._last_report: TrackingReport | None = None
        self._last_scene_description = ""
        self._last_scene_source = "rule"
        self._last_scene_timestamp = 0.0
        self._last_vlm_attempt_ts = 0.0
        self._last_vlm_state: HeadState | None = None
        self._last_vlm_target_snapshot: tuple[str, float, float, float] | None = None
        self._last_vlm_frame_thumbnail: np.ndarray | None = None
        self._lock = threading.RLock()

    def ingest(self, report: TrackingReport) -> None:
        with self._lock:
            self._last_report = report

            center = None
            if report.target is not None:
                center = (report.target.cx, report.target.cy)

            self._memory.append(
                SceneMemoryItem(
                    timestamp=report.timestamp,
                    state=report.state,
                    target_found=report.target_found,
                    pan_cmd=report.pan_cmd,
                    tilt_cmd=report.tilt_cmd,
                    target_center=center,
                )
            )
            self._trim_memory(report.timestamp)

    def _trim_memory(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        while self._memory and (now - self._memory[0].timestamp > self.memory_seconds):
            self._memory.popleft()

    def describe_current_scene(self) -> str:
        with self._lock:
            cached_description = self._last_scene_description
        if cached_description:
            return cached_description
        return self._describe_tracking_state()

    def describe_tracking_state(self) -> str:
        return self._describe_tracking_state()

    def refresh_scene_description(self, frame: Any | None = None) -> str:
        fallback = self._describe_tracking_state()
        description = None
        source = "rule"
        has_vlm = self.use_ollama or self.use_openai
        should_query_vlm = has_vlm and self._should_query_vlm(frame)

        if should_query_vlm:
            self._record_vlm_attempt(frame)

            if self.use_ollama:
                description = self._describe_with_ollama(frame)
                if description is not None:
                    source = "ollama"

            if description is None and self.use_openai:
                description = self._describe_with_openai(frame)
                if description is not None:
                    source = "openai"

        if has_vlm and not should_query_vlm:
            with self._lock:
                cached_description = self._last_scene_description
                cached_source = self._last_scene_source
            summary = cached_description or fallback
            with self._lock:
                self._last_scene_description = summary
                self._last_scene_source = cached_source if cached_description else "rule"
                self._last_scene_timestamp = time.time()
            return summary

        summary = description or fallback
        with self._lock:
            self._last_scene_description = summary
            self._last_scene_source = source
            self._last_scene_timestamp = time.time()
        return summary

    def _should_query_vlm(self, frame: Any | None) -> bool:
        if not self.vlm_only_on_significant_change:
            return True

        now = time.time()
        with self._lock:
            report = self._last_report
            last_attempt_ts = self._last_vlm_attempt_ts
            last_state = self._last_vlm_state
            last_target = self._last_vlm_target_snapshot
            last_thumbnail = self._last_vlm_frame_thumbnail

        if report is None:
            return False

        if last_attempt_ts <= 0.0:
            return True

        if self.vlm_force_refresh_s > 0.0 and (now - last_attempt_ts) >= self.vlm_force_refresh_s:
            return True

        if report.state != last_state:
            return True

        current_target = self._target_snapshot(report, frame)
        if self._target_changed(last_target, current_target):
            return True

        current_thumbnail = self._frame_change_thumbnail(frame)
        if self._frame_changed(last_thumbnail, current_thumbnail):
            return True

        return False

    def _record_vlm_attempt(self, frame: Any | None) -> None:
        now = time.time()
        with self._lock:
            report = self._last_report
            self._last_vlm_attempt_ts = now
            self._last_vlm_state = report.state if report is not None else None
            self._last_vlm_target_snapshot = self._target_snapshot(report, frame)
            self._last_vlm_frame_thumbnail = self._frame_change_thumbnail(frame)

    def _frame_change_thumbnail(self, frame: Any | None) -> np.ndarray | None:
        if frame is None or cv2 is None:
            return None

        try:
            if len(frame.shape) == 2:
                gray = frame
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dim = self.vlm_change_sample_dim_px
            return cv2.resize(gray, (dim, dim), interpolation=cv2.INTER_AREA)
        except Exception:
            return None

    def _frame_changed(self, previous: np.ndarray | None, current: np.ndarray | None) -> bool:
        if previous is None or current is None:
            return False

        try:
            diff = np.mean(np.abs(current.astype(np.float32) - previous.astype(np.float32))) / 255.0
        except Exception:
            return False

        return float(diff) >= self.vlm_change_threshold

    def _target_snapshot(
        self,
        report: TrackingReport | None,
        frame: Any | None,
    ) -> tuple[str, float, float, float] | None:
        if report is None or report.target is None:
            return None

        label = str(report.target.label).strip().lower()
        cx = float(report.target.cx)
        cy = float(report.target.cy)
        area = float(report.target.area)

        if frame is not None:
            try:
                frame_h, frame_w = frame.shape[:2]
                frame_w = max(1, int(frame_w))
                frame_h = max(1, int(frame_h))
                cx /= float(frame_w)
                cy /= float(frame_h)
                area /= float(frame_w * frame_h)
            except Exception:
                pass

        return (label, cx, cy, area)

    def _target_changed(
        self,
        previous: tuple[str, float, float, float] | None,
        current: tuple[str, float, float, float] | None,
    ) -> bool:
        if previous is None and current is None:
            return False
        if previous is None or current is None:
            return True

        prev_label, prev_cx, prev_cy, prev_area = previous
        cur_label, cur_cx, cur_cy, cur_area = current

        if prev_label != cur_label:
            return True

        center_delta = math.hypot(cur_cx - prev_cx, cur_cy - prev_cy)
        if center_delta >= self.vlm_target_center_change_ratio:
            return True

        if prev_area <= 0.0:
            return cur_area > 0.0

        area_delta_ratio = abs(cur_area - prev_area) / prev_area
        return area_delta_ratio >= self.vlm_target_area_change_ratio

    def _describe_tracking_state(self) -> str:
        with self._lock:
            report = self._last_report
        if report is None:
            return "I do not have camera data yet."

        if report.state == HeadState.MANUAL:
            if report.target is None:
                return (
                    f"Tracking is paused for manual control. "
                    f"Pan is {report.pan_cmd:.1f} deg and tilt is {report.tilt_cmd:.1f} deg."
                )

            _, _, w, h = report.target.bbox
            return (
                f"Tracking is paused for manual control. I can still see one {report.target.label} at "
                f"pixel center ({report.target.cx:.0f}, {report.target.cy:.0f}) with bounding box {w}x{h}. "
                f"Pan {report.pan_cmd:.1f} deg, tilt {report.tilt_cmd:.1f} deg."
            )

        if report.target is None:
            return (
                f"I am scanning the scene. Head state is {report.state.value}. "
                f"Pan is {report.pan_cmd:.1f} deg, tilt is {report.tilt_cmd:.1f} deg."
            )

        x, y, w, h = report.target.bbox
        return (
            f"I am tracking one {report.target.label} at pixel center ({report.target.cx:.0f}, {report.target.cy:.0f}) "
            f"with bounding box {w}x{h}. Head state is {report.state.value}. "
            f"Pan {report.pan_cmd:.1f} deg, tilt {report.tilt_cmd:.1f} deg."
        )

    def answer(self, question: str, frame: Any | None = None) -> str:
        clean_question = question.strip()
        q = clean_question.lower()
        if not clean_question:
            return "Please ask a question."

        answer: str | None = None

        if self.use_brain:
            answer = self._answer_with_brain(question=clean_question, frame=frame)

        if answer is None and any(token in q for token in ["what do you see", "describe", "scene"]):
            answer = self.describe_current_scene()

        if answer is None and ("state" in q or "tracking" in q or "follow" in q):
            with self._lock:
                report = self._last_report
            if report is None:
                answer = "I do not have state yet."
            else:
                answer = f"Current behavior state: {report.state.value}."

        if answer is None and "where" in q and "look" in q:
            with self._lock:
                report = self._last_report
            if report is None:
                answer = "I do not have pose data yet."
            else:
                answer = f"I am looking at pan {report.pan_cmd:.1f} deg and tilt {report.tilt_cmd:.1f} deg."

        if answer is None and "how many" in q and "target" in q:
            with self._lock:
                report = self._last_report
            if report is None or not report.target_found:
                answer = "I currently detect 0 tracked targets."
            else:
                label = report.target.label if report.target is not None else "target"
                answer = f"I currently detect 1 tracked {label}."

        if answer is None and self.use_openai:
            answer = self._answer_with_openai(question=clean_question, frame=frame)

        if answer is None:
            answer = (
                "I can answer scene status questions now (state, tracking, where I look, description). "
                "For richer chat, enable the Ollama brain or OpenAI VLM in config."
            )

        self._record_chat_turn(clean_question, answer)
        return answer

    def _answer_with_brain(self, question: str, frame: Any | None) -> str | None:
        if not self.brain_model:
            return None

        scene_summary = self.describe_current_scene()
        tracking_summary = self._describe_tracking_context_for_brain()
        chat_history = self._format_chat_history_for_brain()
        system_prompt = (
            "You are a robot assistant. You can see through the robot's camera and have access to its recent tracking memory."
            "Personality: cute, concise, witty, and friendly. "
            "Always stay grounded in the provided context. "
            "If information is missing or uncertain, say so plainly. "
            "Reply in 1-2 short sentences, max 35 words."
        )
        user_prompt = (
            "User query:\n"
            f"{question}\n\n"
            "Scene description context:\n"
            f"{scene_summary}\n\n"
            "Tracking context from detector/tracker:\n"
            f"{tracking_summary}\n\n"
            "Recent chat history (oldest to newest):\n"
            f"{chat_history}\n\n"
            "Answer as the robot."
        )
        payload = {
            "model": self.brain_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "keep_alive": self.ollama_keep_alive,
            "options": {
                "temperature": self.brain_temperature,
                "num_predict": self.brain_max_tokens,
            },
        }

        body = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            url=f"{self.ollama_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib_request.urlopen(req, timeout=self.ollama_timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (TimeoutError, OSError, ValueError, urllib_error.URLError):
            return None

        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return self._normalize_brain_answer(content)

        text = data.get("response")
        if isinstance(text, str) and text.strip():
            return self._normalize_brain_answer(text)

        return None

    def _answer_with_ollama_brain(self, question: str, frame: Any | None) -> str | None:
        return self._answer_with_brain(question, frame)

    def _normalize_brain_answer(self, text: str) -> str:
        compact = " ".join(text.strip().split())
        if len(compact) <= 220:
            return compact
        shortened = compact[:220].rstrip()
        last_space = shortened.rfind(" ")
        if last_space > 80:
            shortened = shortened[:last_space]
        return shortened.rstrip(". ") + "..."

    def _record_chat_turn(self, question: str, answer: str) -> None:
        if not self.include_chat_history or self.chat_history_max_turns <= 0:
            return
        compact_question = self._normalize_history_text(question, max_chars=180)
        compact_answer = self._normalize_history_text(answer, max_chars=220)
        if not compact_question or not compact_answer:
            return
        with self._lock:
            self._chat_history.append(
                ChatTurn(
                    timestamp=time.time(),
                    user_message=compact_question,
                    robot_reply=compact_answer,
                )
            )
            while len(self._chat_history) > self.chat_history_max_turns:
                self._chat_history.popleft()

    def _format_chat_history_for_brain(self) -> str:
        if not self.include_chat_history or self.chat_history_max_turns <= 0:
            return "History disabled."
        with self._lock:
            turns = list(self._chat_history)[-self.chat_history_max_turns :]
        if not turns:
            return "No prior turns."
        lines: list[str] = []
        for idx, turn in enumerate(turns, start=1):
            lines.append(f"{idx}. User: {turn.user_message}")
            lines.append(f"   Robot: {turn.robot_reply}")
        return "\n".join(lines)

    def _normalize_history_text(self, text: str, max_chars: int) -> str:
        compact = " ".join(str(text).strip().split())
        if not compact:
            return ""
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3].rstrip() + "..."

    def _describe_tracking_context_for_brain(self) -> str:
        with self._lock:
            report = self._last_report
            memory = list(self._memory)
        if report is None:
            return "No tracking report available yet."

        parts = [
            (
                f"State={report.state.value}; "
                f"target_found={str(report.target_found).lower()}; "
                f"head_pose=pan {report.pan_cmd:.1f} deg, tilt {report.tilt_cmd:.1f} deg."
            )
        ]

        if report.target is None:
            parts.append("Current target: none.")
        else:
            x, y, w, h = report.target.bbox
            parts.append(
                "Current target: "
                f"label={report.target.label}, confidence={report.target.confidence:.2f}, "
                f"center=({report.target.cx:.0f},{report.target.cy:.0f}), "
                f"bbox=({x},{y},{w},{h})."
            )

        if memory:
            tracked_count = sum(1 for item in memory if item.target_found)
            tracked_ratio = tracked_count / float(len(memory))
            parts.append(
                "Recent memory: "
                f"target seen in {tracked_count}/{len(memory)} samples ({tracked_ratio:.0%}) "
                f"within ~{self.memory_seconds:.0f}s."
            )

        return " ".join(parts)

    def _scene_description_prompt(self) -> str:
        return (
            "Describe what the robot camera sees right now in one short paragraph. "
            "Use at most two sentences. Mention the main people, objects, and actions, "
            "stay grounded in the image, and say when something is uncertain."
        )

    def _describe_with_ollama(self, frame: Any | None) -> str | None:
        return self._query_with_ollama_prompt(frame=frame, prompt=self._scene_description_prompt())

    def _query_with_ollama_prompt(self, frame: Any | None, prompt: str) -> str | None:
        b64_img = self._encode_frame_b64(frame)
        if b64_img is None:
            return None

        payload = {
            "model": self.vlm_model,
            "prompt": prompt,
            "images": [b64_img],
            "stream": False,
            "keep_alive": self.ollama_keep_alive,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            url=f"{self.ollama_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib_request.urlopen(req, timeout=self.ollama_timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (TimeoutError, OSError, ValueError, urllib_error.URLError):
            return None

        text = data.get("response")
        if isinstance(text, str) and text.strip():
            return text.strip()
        return None

    def _describe_with_openai(self, frame: Any | None) -> str | None:
        return self._answer_with_openai_prompt(
            question="Describe what the camera sees right now in one short paragraph.",
            frame=frame,
            tracking_summary=None,
        )

    def _answer_with_openai(self, question: str, frame: Any | None) -> str | None:
        return self._answer_with_openai_prompt(
            question=question,
            frame=frame,
            tracking_summary=self._describe_tracking_state(),
        )

    def _answer_with_openai_prompt(
        self,
        question: str,
        frame: Any | None,
        tracking_summary: str | None,
    ) -> str | None:
        api_key = os.getenv(self.openai_api_key_env)
        if not api_key:
            return None
        b64_img = self._encode_frame_b64(frame)
        if b64_img is None:
            return None

        try:
            from openai import OpenAI
        except ModuleNotFoundError:
            return None

        system_text = (
            "You are the scene intelligence module for a pan/tilt robot head. "
            "Answer in 1-3 concise sentences using only what is visible and known. "
            "If uncertain, say so clearly."
        )

        prompt = f"Question: {question}"
        if tracking_summary:
            prompt = f"{prompt}\nTracking summary: {tracking_summary}"

        try:
            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=self.openai_model,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": system_text}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64_img}"},
                        ],
                    },
                ],
                max_output_tokens=180,
            )
        except Exception:
            return None

        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        return None

    def _encode_frame_b64(self, frame: Any | None) -> str | None:
        if frame is None or cv2 is None:
            return None

        prepared = frame
        max_dim = self.max_image_dim_px
        if max_dim > 0:
            frame_h, frame_w = frame.shape[:2]
            longest_side = max(frame_h, frame_w)
            if longest_side > max_dim:
                scale = max_dim / float(longest_side)
                resized_w = max(1, int(round(frame_w * scale)))
                resized_h = max(1, int(round(frame_h * scale)))
                prepared = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

        ok, jpeg = cv2.imencode(".jpg", prepared, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return None
        return base64.b64encode(jpeg.tobytes()).decode("ascii")
