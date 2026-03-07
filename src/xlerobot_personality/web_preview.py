from __future__ import annotations

from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any, Callable

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>XLeRobot Web Preview</title>
  <style>
    :root {
      --bg: #08111b;
      --panel: #102133;
      --panel-2: #15304a;
      --text: #ebf3fb;
      --muted: #9fb7cb;
      --accent: #39d0a9;
      --danger: #ff6b57;
      --border: rgba(255, 255, 255, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at top, #17314a 0%, var(--bg) 48%);
      color: var(--text);
      font-family: "Segoe UI", Tahoma, sans-serif;
    }
    .wrap {
      max-width: 1120px;
      margin: 0 auto;
      padding: 20px;
    }
    .hero {
      display: flex;
      gap: 18px;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 18px;
      flex-wrap: wrap;
    }
    h1 {
      margin: 0;
      font-size: 28px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .sub {
      color: var(--muted);
      margin-top: 4px;
      font-size: 14px;
    }
    .panel {
      background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
      border: 1px solid var(--border);
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 16px 48px rgba(0, 0, 0, 0.28);
    }
    .main {
      display: grid;
      grid-template-columns: minmax(0, 1.5fr) minmax(280px, 0.75fr);
      gap: 18px;
    }
    .video-shell {
      position: relative;
      background: #000;
      aspect-ratio: 4 / 3;
    }
    #frame {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      background: #000;
    }
    .status {
      padding: 16px;
      display: grid;
      gap: 12px;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .value {
      margin-top: 6px;
      font-size: 20px;
      font-weight: 700;
    }
    .summary {
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      line-height: 1.4;
      min-height: 96px;
    }
    .control-block {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      display: grid;
      gap: 10px;
    }
    .speed-row {
      display: grid;
      gap: 8px;
    }
    .speed-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .speed-value {
      color: var(--accent);
      font-weight: 700;
      font-size: 14px;
    }
    .speed-slider {
      width: 100%;
      accent-color: var(--accent);
    }
    .teleop-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      align-items: center;
    }
    .teleop-btn {
      min-height: 52px;
      background: #214567;
      color: var(--text);
      font-size: 22px;
      font-weight: 700;
      border-radius: 16px;
    }
    .teleop-btn.secondary {
      background: #1a3248;
      font-size: 14px;
    }
    .teleop-spacer {
      visibility: hidden;
    }
    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.72;
      cursor: default;
    }
    .stop {
      background: var(--danger);
      color: #fff;
    }
    .refresh {
      background: var(--accent);
      color: #04131a;
    }
    .toggle {
      background: #ffd166;
      color: #231800;
    }
    .hint {
      color: var(--muted);
      font-size: 13px;
    }
    @media (max-width: 900px) {
      .main { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div>
        <h1>XLeRobot Preview</h1>
        <div class="sub">Browser-based tracking monitor for SSH/headless runs</div>
      </div>
      <div class="hint">Open this page through SSH port forwarding.</div>
    </div>

    <div class="main">
      <div class="panel">
        <div class="video-shell">
          <img id="frame" alt="Robot preview frame">
        </div>
      </div>

      <div class="panel">
        <div class="status">
          <div class="metric-grid">
            <div class="metric">
              <div class="label">State</div>
              <div class="value" id="state">-</div>
            </div>
            <div class="metric">
              <div class="label">Target</div>
              <div class="value" id="target">-</div>
            </div>
            <div class="metric">
              <div class="label">Pan</div>
              <div class="value" id="pan">-</div>
            </div>
            <div class="metric">
              <div class="label">Tilt</div>
              <div class="value" id="tilt">-</div>
            </div>
          </div>

          <div class="summary" id="summary">Waiting for frames...</div>

          <div class="control-block">
            <div class="label">Teleop</div>
            <div class="speed-row">
              <div class="speed-head">
                <div class="hint">Manual speed</div>
                <div class="speed-value" id="speed-value">35 deg/s</div>
              </div>
              <input class="speed-slider" id="speed-slider" type="range" min="5" max="120" step="5" value="35">
            </div>
            <div class="teleop-grid">
              <button class="teleop-btn teleop-spacer" type="button" tabindex="-1">.</button>
              <button class="teleop-btn" type="button" data-pan="0" data-tilt="1" aria-label="Tilt up">↑</button>
              <button class="teleop-btn teleop-spacer" type="button" tabindex="-1">.</button>
              <button class="teleop-btn" type="button" data-pan="-1" data-tilt="0" aria-label="Pan left">←</button>
              <button class="teleop-btn secondary" type="button" id="toggle-hint" disabled>WASD / Arrows</button>
              <button class="teleop-btn" type="button" data-pan="1" data-tilt="0" aria-label="Pan right">→</button>
              <button class="teleop-btn teleop-spacer" type="button" tabindex="-1">.</button>
              <button class="teleop-btn" type="button" data-pan="0" data-tilt="-1" aria-label="Tilt down">↓</button>
              <button class="teleop-btn teleop-spacer" type="button" tabindex="-1">.</button>
            </div>
            <div class="hint" id="teleop-note">Hold arrow keys or WASD to move the head camera. Press T to toggle tracking.</div>
          </div>

          <div class="actions">
            <button class="toggle" type="button" id="tracking-toggle">Pause Tracking</button>
            <button class="refresh" type="button" id="refresh">Refresh</button>
            <button class="stop" type="button" id="stop">Stop Robot Loop</button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const frameEl = document.getElementById("frame");
    const summaryEl = document.getElementById("summary");
    const stateEl = document.getElementById("state");
    const targetEl = document.getElementById("target");
    const panEl = document.getElementById("pan");
    const tiltEl = document.getElementById("tilt");
    const trackingToggleEl = document.getElementById("tracking-toggle");
    const teleopNoteEl = document.getElementById("teleop-note");
    const speedSliderEl = document.getElementById("speed-slider");
    const speedValueEl = document.getElementById("speed-value");
    let latestStatus = {};
    let activeDrive = null;
    let speedUpdateTimer = null;

    function refreshFrame() {
      frameEl.src = "/frame.jpg?t=" + Date.now();
    }

    function updateSpeedLabel(step) {
      speedValueEl.textContent = Number(step).toFixed(0) + " deg/s";
    }

    function updateControls() {
      const trackingEnabled = latestStatus.tracking_enabled !== false;
      trackingToggleEl.textContent = trackingEnabled ? "Pause Tracking" : "Resume Tracking";
      teleopNoteEl.textContent = trackingEnabled
        ? "Auto-tracking is active. Pause tracking for stable manual teleop. Arrow keys or WASD also work. Press T to toggle tracking."
        : "Manual teleop is active. Use arrow keys or WASD to move the head camera. Press T to resume tracking.";
      const teleopStep = Number(latestStatus.teleop_speed_deg_s ?? 35.0);
      if (document.activeElement !== speedSliderEl) {
        speedSliderEl.value = String(teleopStep);
      }
      updateSpeedLabel(teleopStep);
    }

    async function refreshStatus() {
      try {
        const res = await fetch("/api/status?t=" + Date.now());
        if (!res.ok) {
          throw new Error("status not ready");
        }
        const data = await res.json();
        latestStatus = data;
        stateEl.textContent = data.state ?? "-";
        targetEl.textContent = data.target_found ? "Locked" : "None";
        panEl.textContent = data.pan_cmd_deg == null ? "-" : data.pan_cmd_deg.toFixed(1) + " deg";
        tiltEl.textContent = data.tilt_cmd_deg == null ? "-" : data.tilt_cmd_deg.toFixed(1) + " deg";
        summaryEl.textContent = data.summary ?? "No summary available.";
        updateControls();
      } catch (err) {
        summaryEl.textContent = "Preview server is waiting for fresh robot data.";
      }
    }

    async function sendControl(payload) {
      await fetch("/api/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      refreshFrame();
      setTimeout(refreshStatus, 80);
    }

    async function setTeleopSpeed(stepDeg) {
      const step = Number(stepDeg);
      latestStatus.teleop_speed_deg_s = step;
      updateSpeedLabel(step);
      await sendControl({
        kind: "teleop_speed",
        speed_deg_s: step,
      });
    }

    async function stopRobotLoop() {
      await fetch("/api/stop", { method: "POST" });
    }

    async function toggleTracking() {
      const enabled = latestStatus.tracking_enabled === false;
      await sendControl({ kind: "tracking", enabled });
    }

    async function stopHoldControl() {
      if (!activeDrive) {
        return;
      }
      activeDrive = null;
      await sendControl({ kind: "manual_drive_stop" });
    }

    async function startHoldControl(pan, tilt) {
      if (activeDrive && activeDrive.pan === pan && activeDrive.tilt === tilt) {
        return;
      }
      activeDrive = { pan, tilt };
      await sendControl({
        kind: "manual_drive",
        pan_axis: pan,
        tilt_axis: tilt,
      });
    }

    document.getElementById("refresh").addEventListener("click", () => {
      refreshFrame();
      refreshStatus();
    });
    document.getElementById("stop").addEventListener("click", stopRobotLoop);
    document.getElementById("tracking-toggle").addEventListener("click", toggleTracking);
    document.querySelectorAll("[data-pan][data-tilt]").forEach((button) => {
      const pan = Number(button.dataset.pan || "0");
      const tilt = Number(button.dataset.tilt || "0");
      button.addEventListener("pointerdown", async (event) => {
        event.preventDefault();
        await startHoldControl(pan, tilt);
      });
      button.addEventListener("pointerup", stopHoldControl);
      button.addEventListener("pointerleave", stopHoldControl);
      button.addEventListener("pointercancel", stopHoldControl);
      button.addEventListener("click", (event) => {
        event.preventDefault();
      });
    });
    document.addEventListener("pointerup", stopHoldControl);
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        stopHoldControl();
      }
    });
    speedSliderEl.addEventListener("input", () => {
      const step = Number(speedSliderEl.value);
      latestStatus.teleop_speed_deg_s = step;
      updateSpeedLabel(step);
      if (speedUpdateTimer !== null) {
        clearTimeout(speedUpdateTimer);
      }
      speedUpdateTimer = setTimeout(() => {
        setTeleopSpeed(step);
      }, 120);
    });
    speedSliderEl.addEventListener("change", () => {
      const step = Number(speedSliderEl.value);
      setTeleopSpeed(step);
    });

    document.addEventListener("keyup", (event) => {
      const key = event.key.toLowerCase();
      if (["arrowup", "arrowdown", "arrowleft", "arrowright", "w", "a", "s", "d"].includes(key)) {
        stopHoldControl();
      }
    });

    window.addEventListener("blur", stopHoldControl);

    document.addEventListener("keydown", (event) => {
      if (event.ctrlKey || event.metaKey || event.altKey) {
        return;
      }

      const target = event.target;
      if (target && ["INPUT", "TEXTAREA"].includes(target.tagName)) {
        return;
      }

      const key = event.key.toLowerCase();
      if (key === "arrowup" || key === "w") {
        event.preventDefault();
        if (!event.repeat) {
          startHoldControl(0, 1);
        }
      } else if (key === "arrowdown" || key === "s") {
        event.preventDefault();
        if (!event.repeat) {
          startHoldControl(0, -1);
        }
      } else if (key === "arrowleft" || key === "a") {
        event.preventDefault();
        if (!event.repeat) {
          startHoldControl(-1, 0);
        }
      } else if (key === "arrowright" || key === "d") {
        event.preventDefault();
        if (!event.repeat) {
          startHoldControl(1, 0);
        }
      } else if (key === "t") {
        event.preventDefault();
        toggleTracking();
      }
    });

    refreshFrame();
    refreshStatus();
    setInterval(refreshFrame, 140);
    setInterval(refreshStatus, 500);
  </script>
</body>
</html>
"""


@dataclass
class WebPreviewState:
    jpeg_bytes: bytes | None = None
    status: dict[str, Any] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_callback: Callable[[], None] | None = None
    control_callback: Callable[[dict[str, Any]], None] | None = None


class WebPreviewServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.state = WebPreviewState()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for web preview")
        if self._httpd is not None:
            return

        try:
            server = ThreadingHTTPServer((self.host, self.port), _make_handler(self.state))
        except OSError as exc:
            raise RuntimeError(f"Failed to bind web preview on {self.host}:{self.port}: {exc}") from exc
        server.daemon_threads = True
        self._httpd = server
        self._thread = threading.Thread(target=server.serve_forever, name="web_preview_server", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._httpd = None
        self._thread = None

    def set_stop_callback(self, callback: Callable[[], None]) -> None:
        with self.state.lock:
            self.state.stop_callback = callback

    def set_control_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        with self.state.lock:
            self.state.control_callback = callback

    def update(self, frame_bgr: Any, status: dict[str, Any]) -> None:
        if cv2 is None:
            return
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return

        with self.state.lock:
            self.state.jpeg_bytes = encoded.tobytes()
            self.state.status = dict(status)


def _make_handler(state: WebPreviewState) -> type[BaseHTTPRequestHandler]:
    class PreviewHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/frame.jpg"):
                self._serve_frame()
                return

            if self.path.startswith("/api/status"):
                self._serve_status()
                return

            if self.path == "/" or self.path.startswith("/index.html"):
                self._serve_html()
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:  # noqa: N802
            if self.path.startswith("/api/stop"):
                callback = None
                with state.lock:
                    callback = state.stop_callback

                if callback is not None:
                    callback()

                payload = json.dumps({"ok": True}).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            if self.path.startswith("/api/control"):
                command = self._read_json_body()
                callback = None
                with state.lock:
                    callback = state.control_callback

                if callback is not None:
                    callback(command)

                payload = json.dumps({"ok": True}).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def _serve_html(self) -> None:
            payload = HTML_PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _serve_frame(self) -> None:
            with state.lock:
                jpeg_bytes = state.jpeg_bytes

            if jpeg_bytes is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Frame not ready")
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Content-Length", str(len(jpeg_bytes)))
            self.end_headers()
            self.wfile.write(jpeg_bytes)

        def _serve_status(self) -> None:
            with state.lock:
                payload = json.dumps(state.status).encode("utf-8")

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                payload = {}
            return payload if isinstance(payload, dict) else {}

        def log_message(self, format: str, *args: Any) -> None:
            return

    return PreviewHandler
