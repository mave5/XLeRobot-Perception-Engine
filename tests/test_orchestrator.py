from types import SimpleNamespace

from xlerobot_personality.orchestrator import SystemOrchestrator


class _DummyController:
    def __init__(self):
        self.holds: list[float] = []

    def set_interacting(self, hold_s: float) -> None:
        self.holds.append(hold_s)


class _DummySceneAgent:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def answer(self, question: str, frame: object) -> str:
        self.calls.append((question, frame))
        return "Speech me."


class _DummySpeaker:
    def __init__(self):
        self.messages: list[str] = []

    def speak(self, text: str) -> bool:
        self.messages.append(text)
        return True


def test_dialog_answer_is_forwarded_to_speech():
    controller = _DummyController()
    scene_agent = _DummySceneAgent()
    speaker = _DummySpeaker()
    orchestrator = SystemOrchestrator(
        robot=SimpleNamespace(),
        controller=controller,
        scene_agent=scene_agent,
        tracking_hz=10.0,
        scene_summary_hz=1.0,
        interactive_hold_s=2.5,
        enable_dialog=True,
        speaker=speaker,
        visualize=False,
        web_preview=False,
        web_host="127.0.0.1",
        web_port=8765,
        camera_key="head_cam",
        run_duration_s=0.0,
    )
    orchestrator._latest_report = SimpleNamespace(frame="frame-1")

    answer = orchestrator._answer_dialog_question_sync("hello there")

    assert answer == "Speech me."
    assert controller.holds == [2.5]
    assert scene_agent.calls == [("hello there", "frame-1")]
    assert speaker.messages == ["Speech me."]
