import json
import numpy as np

from xlerobot_personality.scene_agent import SceneAgent
from xlerobot_personality.tracking_controller import HeadState, TrackingReport, TrackedTarget


def _make_report(state: HeadState = HeadState.IDLE_SCAN) -> TrackingReport:
    return TrackingReport(
        timestamp=1.0,
        state=state,
        target=None,
        pan_obs=0.0,
        tilt_obs=0.0,
        pan_cmd=1.5,
        tilt_cmd=-2.0,
        frame=np.zeros((24, 24, 3), dtype=np.uint8),
    )


def _make_person_report(state: HeadState = HeadState.TRACKING) -> TrackingReport:
    report = _make_report(state=state)
    report.target = TrackedTarget(
        cx=12.0,
        cy=10.0,
        area=220.0,
        bbox=(3, 1, 12, 18),
        label="person",
        confidence=0.88,
        track_id=4,
    )
    return report


def test_scene_agent_refresh_falls_back_to_tracking_state():
    agent = SceneAgent(memory_seconds=30.0)
    report = _make_report()
    agent.ingest(report)

    summary = agent.refresh_scene_description(frame=report.frame)

    assert "I am scanning the scene." in summary
    assert agent.describe_current_scene() == summary


def test_scene_agent_refresh_uses_ollama_caption(monkeypatch):
    agent = SceneAgent(memory_seconds=30.0, use_ollama=True)
    report = _make_report(state=HeadState.TRACKING)
    agent.ingest(report)

    monkeypatch.setattr(
        agent,
        "_describe_with_ollama",
        lambda frame: "A person stands near the middle of the frame with a plain background behind them.",
    )

    summary = agent.refresh_scene_description(frame=report.frame)

    assert summary == "A person stands near the middle of the frame with a plain background behind them."
    assert agent.describe_current_scene() == summary
    assert agent.answer("what do you see?") == summary


def test_scene_agent_skips_vlm_when_scene_is_unchanged(monkeypatch):
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    agent = SceneAgent(
        memory_seconds=30.0,
        use_ollama=True,
        vlm_only_on_significant_change=True,
        vlm_force_refresh_s=120.0,
    )
    report = _make_report()
    report.frame = frame
    agent.ingest(report)

    calls = {"count": 0}

    def _fake_ollama(_: np.ndarray) -> str:
        calls["count"] += 1
        return f"caption-{calls['count']}"

    monkeypatch.setattr(agent, "_describe_with_ollama", _fake_ollama)

    first = agent.refresh_scene_description(frame=frame.copy())
    second = agent.refresh_scene_description(frame=frame.copy())

    assert first == "caption-1"
    assert second == "caption-1"
    assert calls["count"] == 1


def test_scene_agent_runs_vlm_after_significant_frame_change(monkeypatch):
    frame_a = np.zeros((32, 32, 3), dtype=np.uint8)
    frame_b = np.full((32, 32, 3), 255, dtype=np.uint8)
    agent = SceneAgent(
        memory_seconds=30.0,
        use_ollama=True,
        vlm_only_on_significant_change=True,
        vlm_change_threshold=0.05,
        vlm_force_refresh_s=120.0,
    )
    report = _make_report()
    report.frame = frame_a
    agent.ingest(report)

    calls = {"count": 0}

    def _fake_ollama(_: np.ndarray) -> str:
        calls["count"] += 1
        return f"caption-{calls['count']}"

    monkeypatch.setattr(agent, "_describe_with_ollama", _fake_ollama)

    first = agent.refresh_scene_description(frame=frame_a.copy())
    second = agent.refresh_scene_description(frame=frame_b.copy())

    assert first == "caption-1"
    assert second == "caption-2"
    assert calls["count"] == 2


def test_scene_agent_answer_uses_brain(monkeypatch):
    agent = SceneAgent(
        memory_seconds=30.0,
        use_brain=True,
        use_ollama=False,
    )
    report = _make_person_report()
    agent.ingest(report)
    agent.refresh_scene_description(frame=report.frame)

    captured: dict[str, str] = {}

    def _fake_ollama(question: str, frame: np.ndarray) -> str:
        captured["question"] = question
        assert frame is report.frame
        return "Hi bestie, I spot one person and I'm locked in."

    monkeypatch.setattr(agent, "_answer_with_brain", _fake_ollama)

    answer = agent.answer("Who are you watching?", frame=report.frame)

    assert "bestie" in answer
    assert captured["question"] == "Who are you watching?"


def test_scene_agent_brain_payload_contains_scene_and_tracking(monkeypatch):
    agent = SceneAgent(
        memory_seconds=30.0,
        use_brain=True,
        use_ollama=False,
        brain_model="qwen2.5:1.5b",
    )
    report = _make_person_report()
    agent.ingest(report)
    agent.refresh_scene_description(frame=report.frame)

    captured: dict[str, object] = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"message": {"content": "Tiny but mighty."}}).encode("utf-8")

    def _fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setattr("xlerobot_personality.scene_agent.urllib_request.urlopen", _fake_urlopen)

    answer = agent.answer("What are you focused on?", frame=report.frame)

    assert answer == "Tiny but mighty."
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "qwen2.5:1.5b"
    assert "Scene description context:" in payload["messages"][1]["content"]
    assert "Tracking context from detector/tracker:" in payload["messages"][1]["content"]
    assert "Recent chat history (oldest to newest):" in payload["messages"][1]["content"]
    assert "label=person" in payload["messages"][1]["content"]


def test_scene_agent_brain_tracking_context_includes_target_details():
    agent = SceneAgent(memory_seconds=30.0, use_brain=True)
    report = _make_person_report()
    agent.ingest(report)

    context = agent._describe_tracking_context_for_brain()

    assert "State=TRACKING" in context
    assert "label=person" in context
    assert "target seen in 1/1 samples" in context


def test_scene_agent_brain_includes_recent_chat_history(monkeypatch):
    agent = SceneAgent(
        memory_seconds=30.0,
        use_brain=True,
        use_ollama=False,
        include_chat_history=True,
        chat_history_max_turns=2,
    )
    report = _make_person_report()
    agent.ingest(report)
    agent.refresh_scene_description(frame=report.frame)
    agent._record_chat_turn("turn one question", "turn one answer")
    agent._record_chat_turn("turn two question", "turn two answer")
    agent._record_chat_turn("turn three question", "turn three answer")

    captured: dict[str, object] = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"message": {"content": "History acknowledged."}}).encode("utf-8")

    def _fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setattr("xlerobot_personality.scene_agent.urllib_request.urlopen", _fake_urlopen)

    agent.answer("What did we discuss?", frame=report.frame)

    payload = captured["payload"]
    assert isinstance(payload, dict)
    content = payload["messages"][1]["content"]
    assert "turn one question" not in content
    assert "turn two question" in content
    assert "turn three question" in content


def test_scene_agent_brain_answer_does_not_trigger_vlm_generate(monkeypatch):
    agent = SceneAgent(memory_seconds=30.0, use_brain=True, use_ollama=True, use_openai=False)
    report = _make_person_report()
    agent.ingest(report)
    agent.refresh_scene_description(frame=report.frame)

    captured: dict[str, object] = {"urls": []}

    class _FakeResponse:
        def __init__(self, data: dict[str, object]):
            self._data = data

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(self._data).encode("utf-8")

    def _fake_urlopen(req, timeout):
        captured["urls"].append(req.full_url)
        if req.full_url.endswith("/api/generate"):
            return _FakeResponse({"response": "Scene summary from VLM."})
        if req.full_url.endswith("/api/chat"):
            return _FakeResponse({"message": {"content": "Using cached scene context."}})
        raise AssertionError(f"Unexpected URL: {req.full_url}")

    monkeypatch.setattr("xlerobot_personality.scene_agent.urllib_request.urlopen", _fake_urlopen)

    # Reset call history so only the query-time behavior is measured.
    captured["urls"] = []
    answer = agent.answer("What is the person wearing?", frame=report.frame)

    assert answer == "Using cached scene context."
    assert captured["urls"] == ["http://127.0.0.1:11434/api/chat"]


def test_scene_agent_brain_payload_includes_scene_description_context(monkeypatch):
    agent = SceneAgent(
        memory_seconds=30.0,
        use_brain=True,
        use_ollama=False,
        brain_model="qwen2.5:1.5b",
    )
    report = _make_person_report()
    agent.ingest(report)
    agent.refresh_scene_description(frame=report.frame)

    captured: dict[str, object] = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"message": {"content": "Blue hoodie spotted."}}).encode("utf-8")

    def _fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setattr("xlerobot_personality.scene_agent.urllib_request.urlopen", _fake_urlopen)

    answer = agent.answer("What is the person wearing?", frame=report.frame)

    assert answer == "Blue hoodie spotted."
    payload = captured["payload"]
    assert isinstance(payload, dict)
    content = payload["messages"][1]["content"]
    assert "Scene description context:" in content
    assert "I am tracking one person at pixel center" in content
