from __future__ import annotations

import asyncio
import contextlib
import select
import sys
import time
from typing import Any

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

from .scene_agent import SceneAgent
from .speech import TextToSpeechPlayer
from .tracking_controller import PanTiltTrackingController, TrackingReport, annotate_tracking_frame
from .web_preview import WebPreviewServer


class SystemOrchestrator:
    def __init__(
        self,
        robot: Any,
        controller: PanTiltTrackingController,
        scene_agent: SceneAgent,
        tracking_hz: float,
        scene_summary_hz: float,
        interactive_hold_s: float,
        enable_dialog: bool,
        speaker: TextToSpeechPlayer | None,
        visualize: bool,
        web_preview: bool,
        web_host: str,
        web_port: int,
        camera_key: str,
        run_duration_s: float,
    ):
        self.robot = robot
        self.controller = controller
        self.scene_agent = scene_agent
        self.tracking_hz = tracking_hz
        self.scene_summary_hz = max(0.1, scene_summary_hz)
        self.interactive_hold_s = interactive_hold_s
        self.enable_dialog = enable_dialog
        self.speaker = speaker
        self.visualize = visualize and (cv2 is not None)
        self.web_preview = web_preview and (cv2 is not None)
        self.camera_key = camera_key
        self.run_duration_s = run_duration_s
        self.window_name = "xlerobot-head"
        self.web_server = WebPreviewServer(web_host, web_port) if self.web_preview else None

        self._latest_report: TrackingReport | None = None
        self._stop_event = asyncio.Event()
        self._start_ts = 0.0

    @property
    def latest_frame(self):
        if self._latest_report is None:
            return None
        return self._latest_report.frame

    async def run(self) -> None:
        self._start_ts = time.time()
        loop = asyncio.get_running_loop()
        self.robot.connect(calibrate=False)
        if self.web_server is not None:
            self.web_server.set_stop_callback(lambda: loop.call_soon_threadsafe(self._stop_event.set))
            self.web_server.set_control_callback(
                lambda command: loop.call_soon_threadsafe(self._handle_web_command, command)
            )
            self.web_server.start()
            print(f"[web] Preview available at {self.web_server.url}", flush=True)

        tasks = [
            asyncio.create_task(self._tracking_loop(), name="tracking_loop"),
            asyncio.create_task(self._scene_loop(), name="scene_loop"),
        ]

        if self.enable_dialog:
            tasks.append(asyncio.create_task(self._dialog_loop(), name="dialog_loop"))

        if self.run_duration_s > 0:
            tasks.append(asyncio.create_task(self._duration_guard(), name="duration_guard"))

        try:
            await self._stop_event.wait()
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            if self.web_server is not None:
                self.web_server.stop()
            if self.speaker is not None:
                self.speaker.close()
            self.robot.disconnect()
            if self.visualize and cv2 is not None:
                cv2.destroyWindow(self.window_name)
                cv2.destroyAllWindows()
                cv2.waitKey(1)

    async def _duration_guard(self) -> None:
        await asyncio.sleep(self.run_duration_s)
        self._stop_event.set()

    async def _tracking_loop(self) -> None:
        period = 1.0 / max(1.0, self.tracking_hz)

        while not self._stop_event.is_set():
            loop_start = time.perf_counter()

            observation = self.robot.get_observation()
            report = self.controller.step(observation)
            self.scene_agent.ingest(report)
            self._latest_report = report

            if self.visualize or self.web_server is not None:
                annotated = annotate_tracking_frame(report, camera_key=self.camera_key)

            if self.web_server is not None:
                self.web_server.update(annotated, self._preview_status(report))

            if self.visualize:
                try:
                    cv2.imshow(self.window_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self._stop_event.set()
                        break
                    if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                        self._stop_event.set()
                        break
                except cv2.error:
                    self._stop_event.set()
                    break

            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0.0, period - elapsed))

    async def _scene_loop(self) -> None:
        period = 1.0 / self.scene_summary_hz
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()
            if self._latest_report is not None:
                frame = self.latest_frame
                if frame is not None:
                    frame = frame.copy()
                summary = await asyncio.to_thread(self.scene_agent.refresh_scene_description, frame)
                # Keep the terminal prompt stable while interactive dialog is enabled.
                if not self.enable_dialog:
                    print(f"[scene] {summary}", flush=True)
            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0.0, period - elapsed))

    async def _dialog_loop(self) -> None:
        print("[dialog] Ask questions. Type 'quit' to stop.", flush=True)
        prompt_visible = False
        line_queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        use_reader = hasattr(loop, "add_reader")

        def _on_stdin_ready() -> None:
            line = sys.stdin.readline()
            if line == "":
                line_queue.put_nowait(None)
                return
            line_queue.put_nowait(line)

        if use_reader:
            try:
                loop.add_reader(sys.stdin.fileno(), _on_stdin_ready)
            except (NotImplementedError, PermissionError, OSError):
                use_reader = False

        try:
            while not self._stop_event.is_set():
                if not prompt_visible:
                    print("ask> ", end="", flush=True)
                    prompt_visible = True

                if use_reader:
                    try:
                        question = await asyncio.wait_for(line_queue.get(), timeout=0.25)
                    except asyncio.TimeoutError:
                        continue
                else:
                    try:
                        question = await asyncio.to_thread(_read_stdin_line, 0.25)
                    except EOFError:
                        print("", flush=True)
                        self._stop_event.set()
                        break

                    if question is None:
                        continue

                if question is None:
                    print("", flush=True)
                    self._stop_event.set()
                    break

                prompt_visible = False
                q = question.strip()
                if not q:
                    continue

                if q.lower() in {"quit", "exit", "q"}:
                    self._stop_event.set()
                    break

                await self._answer_dialog_question(q)
        finally:
            try:
                if use_reader:
                    loop.remove_reader(sys.stdin.fileno())
            except Exception:
                pass

    async def _answer_dialog_question(self, question: str) -> str:
        return await asyncio.to_thread(self._answer_dialog_question_sync, question)

    def _answer_dialog_question_sync(self, question: str) -> str:
        self.controller.set_interacting(self.interactive_hold_s)
        answer = self.scene_agent.answer(question, self.latest_frame)
        print(f"[agent] {answer}", flush=True)
        if self.speaker is not None:
            self.speaker.speak(answer)
        return answer

    def _preview_status(self, report: TrackingReport) -> dict[str, Any]:
        target_center = None
        target_bbox = None
        if report.target is not None:
            target_center = [round(report.target.cx, 1), round(report.target.cy, 1)]
            target_bbox = list(report.target.bbox)

        return {
            "timestamp": report.timestamp,
            "state": report.state.value,
            "target_found": report.target_found,
            "target_center": target_center,
            "target_bbox": target_bbox,
            "target_label": report.target.label if report.target is not None else None,
            "target_confidence": report.target.confidence if report.target is not None else None,
            "pan_cmd_deg": report.pan_cmd,
            "tilt_cmd_deg": report.tilt_cmd,
            "tracking_enabled": self.controller.tracking_enabled,
            "teleop_speed_deg_s": self.controller.manual_speed_deg_s,
            "summary": self.scene_agent.describe_current_scene(),
        }

    def _handle_web_command(self, command: dict[str, Any]) -> None:
        kind = str(command.get("kind", "")).strip().lower()
        if not kind:
            return

        if kind == "tracking":
            enabled = bool(command.get("enabled", True))
            self.controller.set_tracking_enabled(enabled)
            return

        if kind == "nudge":
            pan_delta = float(command.get("pan_delta", 0.0))
            tilt_delta = float(command.get("tilt_delta", 0.0))
            self.controller.queue_manual_delta(pan_delta=pan_delta, tilt_delta=tilt_delta)
            return

        if kind == "manual_drive":
            pan_axis = float(command.get("pan_axis", 0.0))
            tilt_axis = float(command.get("tilt_axis", 0.0))
            self.controller.set_manual_drive(pan_axis=pan_axis, tilt_axis=tilt_axis)
            return

        if kind == "manual_drive_stop":
            self.controller.stop_manual_drive()
            return

        if kind == "teleop_speed":
            speed_deg_s = float(
                command.get("speed_deg_s", command.get("step_deg", self.controller.manual_speed_deg_s))
            )
            self.controller.set_manual_speed_deg_s(speed_deg_s)
            return


def _read_stdin_line(timeout_s: float) -> str | None:
    readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not readable:
        return None

    line = sys.stdin.readline()
    if line == "":
        raise EOFError

    return line
