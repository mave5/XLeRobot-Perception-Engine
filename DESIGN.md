# System Design

## Goals

- Keep head tracking control loop real-time and deterministic.
- Keep language/scene reasoning decoupled so API latency never blocks servo control.
- Keep implementation LeRobot-compatible and hardware-portable.

## Pipeline Diagram

```mermaid
flowchart TD
    Cam[Real head camera<br/>or synthetic camera]
    Head[XLERobotHead<br/>xlerobot_head.py]
    Ctrl[PanTiltTrackingController<br/>tracking_controller.py]
    Detector[Detector / Tracker backend<br/>motion | hog_person | yolo_person | yolo_pose_person]
    Select[Target selection<br/>sticky | most_centered]
    FSM[Behavior FSM<br/>MANUAL | TRACKING | REACQUIRE | IDLE_SCAN | INTERACTING]
    Control[Control law<br/>image error + torso/hip aim point<br/>top-of-person framing<br/>PID + smoothing + limits]
    Servos[Pan / tilt servos]
    Report[TrackingReport<br/>annotated frame + state + target + commands]
    Web[WebPreviewServer<br/>browser UI + teleop + tracking toggle]
    Scene[SceneAgent<br/>scene summaries + Ollama brain chat + OpenAI fallback QA]
    Orch[SystemOrchestrator<br/>tracking loop + scene loop + dialog loop]

    Cam --> Head
    Head -->|frame, pan.pos, tilt.pos| Ctrl
    Ctrl --> Detector
    Detector --> Select
    Select --> FSM
    FSM --> Control
    Control -->|pan.pos, tilt.pos| Head
    Head --> Servos

    Ctrl --> Report
    Report --> Web
    Report --> Scene
    Web --> Orch
    Orch --> Ctrl
    Scene --> Orch
```

## Runtime loops

1. Tracking loop (`20-50 Hz`)
- Reads latest robot observation (`pan.pos`, `tilt.pos`, `head_cam`)
- Runs detector/tracker backend and target selection
- Computes pan/tilt commands (PID + framing logic + safety clamps)
- Sends servo action

2. Scene loop (`0.5-1 Hz`)
- Summarizes latest state + short memory window
- Optionally sends the latest frame to a local Ollama vision model for a short paragraph caption
- Emits textual scene status

3. Dialog loop (event-driven)
- Accepts user questions
- Switches FSM to `INTERACTING` for stable gaze hold
- If enabled, calls Ollama lightweight chat brain with:
  - latest VLM scene summary
  - current tracking/person-detection context
  - recent chat turns (bounded history, default 10 turns)
  - user query
- Falls back to rule-based/OpenAI responses if Ollama chat is unavailable

4. Web preview loop (threaded HTTP server)
- Serves annotated latest frame to a browser
- Exposes lightweight status JSON, teleop, and tracking controls
- Avoids X11/OpenCV GUI dependency for SSH workflows

## Core modules

- `xlerobot_head.py`
  - LeRobot-style robot wrapper and camera/motor adapters
- `tracking_controller.py`
  - Detector backends, target selector, PID controller, head behavior FSM
- `scene_agent.py`
  - Memory-backed scene description and Q/A
  - Cached VLM captions refreshed off the control path
- `orchestrator.py`
  - Async coordination and lifecycle
- `web_preview.py`
  - Browser preview, keyboard teleop, and operator controls

## Behavior FSM

- `MANUAL`: operator teleop or tracking paused.
- `IDLE_SCAN`: no target, left-right sweep search.
- `TRACKING`: target visible, smooth pursuit.
- `REACQUIRE`: target just lost, faster sweep.
- `INTERACTING`: hold gaze while answering.

## Safety

- Hard pan/tilt limits in degrees.
- Per-step and per-second command limiting.
- Transition smoothing across scan/track/manual state changes.
- Mock mode default for safe development.

## Upgrade path

1. Add browser-side runtime tuning for tracking/framing gains.
2. Add explicit multi-person target handoff controls in the web UI.
3. Add voice I/O and intent-conditioned gestures.
4. Add semantic memory store for longer conversations.
