from pathlib import Path

from xlerobot_personality.speech import TextToSpeechPlayer


class _FakeStdin:
    def __init__(self):
        self.writes: list[bytes] = []
        self.flush_count = 0
        self.closed = False

    def write(self, data):
        self.writes.append(data)
        return len(data)

    def flush(self):
        self.flush_count += 1

    def close(self):
        self.closed = True


class _FakeProcess:
    def __init__(self):
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.stdin = _FakeStdin()

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


class _FakeSynthesisConfig:
    def __init__(
        self,
        speaker_id=None,
        length_scale=None,
        noise_scale=None,
        noise_w_scale=None,
        volume=1.0,
        normalize_audio=True,
    ):
        self.speaker_id = speaker_id
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w_scale = noise_w_scale
        self.volume = volume
        self.normalize_audio = normalize_audio


class _FakeVoice:
    def __init__(self):
        self.stream_calls: list[tuple[str, _FakeSynthesisConfig]] = []
        self.wav_calls: list[tuple[str, _FakeSynthesisConfig, bool]] = []
        self.config = type("Config", (), {"sample_rate": 22050})()

    def synthesize_wav(self, text, wav_file, syn_config=None, set_wav_format=True):
        self.wav_calls.append((text, syn_config, set_wav_format))
        wav_file.writeframes(b"\0\0" * 32)

    def synthesize(self, text, syn_config=None):
        self.stream_calls.append((text, syn_config))
        yield type("Chunk", (), {"audio_int16_bytes": b"\x01\x02" * 32})()


def test_tts_player_uses_piper_and_aplay(tmp_path, monkeypatch):
    model_path = tmp_path / "voice.onnx"
    config_path = tmp_path / "voice.onnx.json"
    model_path.write_bytes(b"fake-model")
    config_path.write_text("{}")

    fake_voice = _FakeVoice()
    captured: dict[str, object] = {}

    class _FakePiperVoice:
        @staticmethod
        def load(model_path, config_path, use_cuda=False, download_dir=None):
            captured["model_path"] = model_path
            captured["config_path"] = config_path
            captured["use_cuda"] = use_cuda
            captured["download_dir"] = download_dir
            return fake_voice

    processes: list[_FakeProcess] = []

    def _fake_popen(command, stdin=None, stdout=None, stderr=None, bufsize=None):
        captured["command"] = command
        captured["stdin"] = stdin
        captured["bufsize"] = bufsize
        process = _FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr("xlerobot_personality.speech.PiperVoice", _FakePiperVoice)
    monkeypatch.setattr("xlerobot_personality.speech.SynthesisConfig", _FakeSynthesisConfig)
    monkeypatch.setattr("xlerobot_personality.speech.shutil.which", lambda cmd: "/usr/bin/aplay" if cmd == "aplay" else None)
    monkeypatch.setattr("xlerobot_personality.speech.subprocess.Popen", _fake_popen)

    player = TextToSpeechPlayer(
        model_path=str(model_path),
        audio_player="auto",
        speaker_id=7,
        length_scale=0.85,
        noise_scale=0.4,
        noise_w_scale=0.9,
        volume=1.2,
        lead_in_ms=120,
    )

    assert player.available is True
    assert player.backend_name == "piper"
    assert player.player_name == "aplay"
    assert captured["model_path"] == str(model_path)
    assert captured["config_path"] == str(config_path)
    assert captured["download_dir"] == str(tmp_path)

    assert player.speak("  Hello\nrobot   ") is True
    assert fake_voice.stream_calls[0][0] == "Hello robot"
    synthesis_config = fake_voice.stream_calls[0][1]
    assert synthesis_config.speaker_id == 7
    assert synthesis_config.length_scale == 0.85
    assert synthesis_config.noise_scale == 0.4
    assert synthesis_config.noise_w_scale == 0.9
    assert synthesis_config.volume == 1.2
    assert captured["command"] == [
        "/usr/bin/aplay",
        "-q",
        "-t",
        "raw",
        "-f",
        "S16_LE",
        "-r",
        "22050",
        "-c",
        "1",
        "/dev/stdin",
    ]
    assert captured["stdin"] is not None
    assert captured["bufsize"] == 0
    lead_in_bytes = b"\0\0" * int(22050 * 0.12)
    assert processes[0].stdin.writes == [lead_in_bytes + (b"\x01\x02" * 32)]
    player.close()


def test_tts_player_reuses_stream_process(tmp_path, monkeypatch):
    model_path = tmp_path / "voice.onnx"
    config_path = tmp_path / "voice.onnx.json"
    model_path.write_bytes(b"fake-model")
    config_path.write_text("{}")

    class _FakePiperVoice:
        @staticmethod
        def load(model_path, config_path, use_cuda=False, download_dir=None):
            return _FakeVoice()

    commands: list[list[str]] = []
    processes: list[_FakeProcess] = []

    def _fake_popen(command, stdin=None, stdout=None, stderr=None, bufsize=None):
        process = _FakeProcess()
        commands.append(command)
        processes.append(process)
        return process

    monkeypatch.setattr("xlerobot_personality.speech.PiperVoice", _FakePiperVoice)
    monkeypatch.setattr("xlerobot_personality.speech.SynthesisConfig", _FakeSynthesisConfig)
    monkeypatch.setattr("xlerobot_personality.speech.shutil.which", lambda cmd: "/usr/bin/aplay" if cmd == "aplay" else None)
    monkeypatch.setattr("xlerobot_personality.speech.subprocess.Popen", _fake_popen)

    player = TextToSpeechPlayer(model_path=str(model_path), audio_player="aplay")

    assert player.speak("first") is True
    assert player.speak("second") is True
    assert len(processes) == 1
    assert len(processes[0].stdin.writes) == 2
    player.close()
    assert processes[0].terminated is True
    assert processes[0].stdin.closed is True
