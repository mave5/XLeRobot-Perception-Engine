from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import wave

try:
    from piper import PiperVoice
    try:
        from piper import SynthesisConfig
    except ImportError:
        from piper.config import SynthesisConfig
except ModuleNotFoundError:
    PiperVoice = None
    SynthesisConfig = None


@dataclass(frozen=True)
class _AudioPlayer:
    name: str
    executable: str


class TextToSpeechPlayer:
    _KNOWN_PLAYERS: tuple[tuple[str, str], ...] = (
        ("paplay", "paplay"),
        ("ffplay", "ffplay"),
        ("aplay", "aplay"),
        ("afplay", "afplay"),
    )

    def __init__(
        self,
        backend: str = "piper",
        model_path: str | None = None,
        config_path: str | None = None,
        audio_player: str = "auto",
        speaker_id: int | None = None,
        length_scale: float | None = None,
        noise_scale: float | None = None,
        noise_w_scale: float | None = None,
        volume: float = 1.0,
        use_cuda: bool = False,
        lead_in_ms: int = 250,
    ):
        self.backend = (backend or "piper").strip().lower()
        self.model_path = Path(model_path).expanduser() if isinstance(model_path, str) and model_path else None
        self.config_path = Path(config_path).expanduser() if isinstance(config_path, str) and config_path else None
        self.speaker_id = speaker_id
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w_scale = noise_w_scale
        self.volume = volume
        self.use_cuda = use_cuda
        self.lead_in_ms = max(0, int(lead_in_ms))

        self.error: str | None = None
        self._audio_player = self._resolve_audio_player(audio_player)
        self._voice: PiperVoice | None = None
        self._lock = threading.Lock()
        self._playback_process: subprocess.Popen[bytes] | None = None
        self._playback_path: Path | None = None

        self._initialize()

    @property
    def available(self) -> bool:
        return self._voice is not None and self._audio_player is not None and self.error is None

    @property
    def backend_name(self) -> str | None:
        if not self.available:
            return None
        return "piper"

    @property
    def player_name(self) -> str | None:
        if self._audio_player is None:
            return None
        return self._audio_player.name

    def speak(self, text: str) -> bool:
        if not self.available:
            return False

        normalized_text = self._normalize_text(text)
        if not normalized_text:
            return False

        wav_path: Path | None = None
        with self._lock:
            try:
                if self._uses_streaming_player():
                    self._ensure_stream_process()
                    self._write_stream_audio(normalized_text)
                    return True

                self._stop_locked()
                wav_path = self._synthesize_to_file(normalized_text)
                self._playback_process = subprocess.Popen(
                    self._build_player_command(wav_path),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._playback_path = wav_path
            except Exception:
                self._playback_process = None
                if wav_path is not None:
                    with contextlib.suppress(FileNotFoundError):
                        wav_path.unlink()
                return False
        return True

    def close(self) -> None:
        with self._lock:
            self._stop_locked()

    def _initialize(self) -> None:
        if self.backend not in {"auto", "piper"}:
            self.error = f"Unsupported speech backend: {self.backend}"
            return

        if PiperVoice is None or SynthesisConfig is None:
            self.error = "Install `piper-tts` to enable Piper speech."
            return

        if self.model_path is None:
            self.error = "Set `runtime.speech.model_path` to a Piper .onnx file."
            return
        if not self.model_path.exists():
            self.error = f"Piper model file not found: {self.model_path}"
            return

        if self.config_path is None:
            self.config_path = Path(f"{self.model_path}.json")
        if not self.config_path.exists():
            self.error = f"Piper config file not found: {self.config_path}"
            return

        if self._audio_player is None:
            self.error = "No supported audio player found. Install `aplay`, `paplay`, or `ffplay`."
            return

        try:
            self._voice = PiperVoice.load(
                model_path=str(self.model_path),
                config_path=str(self.config_path),
                use_cuda=self.use_cuda,
                download_dir=str(self.model_path.parent),
            )
        except Exception as exc:
            self.error = f"Failed to load Piper voice: {exc}"
            self._voice = None
            return

        self.error = None

    def _uses_streaming_player(self) -> bool:
        return self._audio_player is not None and self._audio_player.name in {"aplay", "paplay"}

    def _ensure_stream_process(self) -> None:
        if self._playback_process is not None and self._playback_process.poll() is None:
            return

        self._stop_locked()
        self._playback_process = subprocess.Popen(
            self._build_stream_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def _write_stream_audio(self, text: str) -> None:
        assert self._playback_process is not None
        assert self._playback_process.stdin is not None

        audio_bytes = self._build_stream_bytes(text)
        try:
            self._playback_process.stdin.write(audio_bytes)
            self._playback_process.stdin.flush()
        except (BrokenPipeError, OSError):
            self._stop_locked()
            self._ensure_stream_process()
            assert self._playback_process is not None
            assert self._playback_process.stdin is not None
            self._playback_process.stdin.write(audio_bytes)
            self._playback_process.stdin.flush()

    def _build_stream_bytes(self, text: str) -> bytes:
        assert self._voice is not None

        audio_parts = [self._build_lead_in_silence()]
        synthesis_config = self._build_synthesis_config()
        for audio_chunk in self._voice.synthesize(text, syn_config=synthesis_config):
            audio_parts.append(audio_chunk.audio_int16_bytes)
        return b"".join(audio_parts)

    def _synthesize_to_file(self, text: str) -> Path:
        assert self._voice is not None

        with tempfile.NamedTemporaryFile(prefix="xlerobot_tts_", suffix=".wav", delete=False) as temp_file:
            wav_path = Path(temp_file.name)

        try:
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setframerate(self._voice.config.sample_rate)
                wav_file.setsampwidth(2)
                wav_file.setnchannels(1)
                self._write_lead_in_silence(wav_file)
                self._voice.synthesize_wav(
                    text,
                    wav_file,
                    syn_config=self._build_synthesis_config(),
                    set_wav_format=False,
                )
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                wav_path.unlink()
            raise

        return wav_path

    def _build_synthesis_config(self):
        assert SynthesisConfig is not None
        return SynthesisConfig(
            speaker_id=self.speaker_id,
            length_scale=self.length_scale,
            noise_scale=self.noise_scale,
            noise_w_scale=self.noise_w_scale,
            volume=self.volume,
        )

    def _build_player_command(self, wav_path: Path) -> list[str]:
        assert self._audio_player is not None

        if self._audio_player.name == "aplay":
            return [self._audio_player.executable, "-q", str(wav_path)]
        if self._audio_player.name == "paplay":
            return [self._audio_player.executable, str(wav_path)]
        if self._audio_player.name == "ffplay":
            return [self._audio_player.executable, "-loglevel", "quiet", "-autoexit", "-nodisp", str(wav_path)]
        return [self._audio_player.executable, str(wav_path)]

    def _build_stream_command(self) -> list[str]:
        assert self._audio_player is not None
        assert self._voice is not None

        sample_rate = str(self._voice.config.sample_rate)
        if self._audio_player.name == "paplay":
            return [
                self._audio_player.executable,
                "--raw",
                "--rate",
                sample_rate,
                "--format",
                "s16le",
                "--channels",
                "1",
                "--client-name",
                "xlerobot-personality",
                "--stream-name",
                "xlerobot-speech",
                "/dev/stdin",
            ]

        return [
            self._audio_player.executable,
            "-q",
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-r",
            sample_rate,
            "-c",
            "1",
            "/dev/stdin",
        ]

    def _write_lead_in_silence(self, wav_file: wave.Wave_write) -> None:
        silence_bytes = self._build_lead_in_silence()
        if not silence_bytes:
            return
        wav_file.writeframes(silence_bytes)

    def _build_lead_in_silence(self) -> bytes:
        assert self._voice is not None
        if self.lead_in_ms <= 0:
            return b""

        sample_rate = self._voice.config.sample_rate
        num_frames = int(sample_rate * self.lead_in_ms / 1000.0)
        if num_frames <= 0:
            return b""
        return b"\0\0" * num_frames

    def _resolve_audio_player(self, requested_player: str) -> _AudioPlayer | None:
        normalized = (requested_player or "auto").strip().lower()
        if normalized == "auto":
            for player_name, executable in self._KNOWN_PLAYERS:
                resolved = shutil.which(executable)
                if resolved is not None:
                    return _AudioPlayer(name=player_name, executable=resolved)
            return None

        for player_name, executable in self._KNOWN_PLAYERS:
            if normalized != player_name:
                continue
            resolved = shutil.which(executable)
            if resolved is not None:
                return _AudioPlayer(name=player_name, executable=resolved)
            return None
        return None

    def _stop_locked(self) -> None:
        if self._playback_process is not None:
            if self._playback_process.stdin is not None:
                with contextlib.suppress(Exception):
                    self._playback_process.stdin.close()
            if self._playback_process.poll() is None:
                self._playback_process.terminate()
                try:
                    self._playback_process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    self._playback_process.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        self._playback_process.wait(timeout=0.2)

        self._playback_process = None
        if self._playback_path is not None:
            with contextlib.suppress(FileNotFoundError):
                self._playback_path.unlink()
            self._playback_path = None

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(str(text).split())
