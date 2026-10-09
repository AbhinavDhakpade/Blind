"""
app/alerts/speech.py
---------------------
Text-to-speech alert interface and implementations.

Supported backends
------------------
espeak : system espeak-ng (recommended for Pi, no internet required)
gtts   : Google TTS (requires internet, higher quality)
mock   : logs message without audio
"""

from __future__ import annotations

import abc
import logging
import subprocess
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class SpeechAlertInterface(abc.ABC):
    @abc.abstractmethod
    def initialize(self) -> None: ...

    @abc.abstractmethod
    def speak(self, text: str, blocking: bool = False) -> None:
        """Speak *text*.  When blocking=False, audio runs in a daemon thread."""

    @abc.abstractmethod
    def cleanup(self) -> None: ...


# ---------------------------------------------------------------------------
# eSpeak-NG implementation (Raspberry Pi native, offline)
# ---------------------------------------------------------------------------

class EspeakSpeechAlert(SpeechAlertInterface):
    """Uses system espeak-ng via subprocess.  Zero Python deps beyond stdlib."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: Optional[subprocess.Popen] = None

    def initialize(self) -> None:
        # Verify espeak-ng is installed
        result = subprocess.run(
            ["espeak-ng", "--version"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "espeak-ng not found.  Install: sudo apt install -y espeak-ng"
            )
        logger.info("EspeakSpeechAlert initialised.")

    def speak(self, text: str, blocking: bool = False) -> None:
        if blocking:
            self._run(text)
        else:
            t = threading.Thread(target=self._run, args=(text,), daemon=True)
            t.start()

    def _run(self, text: str) -> None:
        with self._lock:
            if self._current is not None:
                try:
                    self._current.terminate()
                except ProcessLookupError:
                    pass
            self._current = subprocess.Popen(
                ["espeak-ng", "-s", "140", "-p", "50", text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._current.wait()

    def cleanup(self) -> None:
        if self._current is not None:
            try:
                self._current.terminate()
            except ProcessLookupError:
                pass


# ---------------------------------------------------------------------------
# gTTS implementation (requires internet)
# ---------------------------------------------------------------------------

class GTTSSpeechAlert(SpeechAlertInterface):
    """Google TTS — requires internet and gtts + pygame (or playsound)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def initialize(self) -> None:
        try:
            import gtts  # noqa: F401  type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "gtts not installed.  Run: pip install gtts playsound"
            ) from exc
        logger.info("GTTSSpeechAlert initialised.")

    def speak(self, text: str, blocking: bool = False) -> None:
        if blocking:
            self._run(text)
        else:
            t = threading.Thread(target=self._run, args=(text,), daemon=True)
            t.start()

    def _run(self, text: str) -> None:
        import io
        from gtts import gTTS  # type: ignore
        import pygame  # type: ignore
        with self._lock:
            tts = gTTS(text=text, lang="en", slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            pygame.mixer.init()
            pygame.mixer.music.load(buf)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                import time; time.sleep(0.05)

    def cleanup(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Mock implementation
# ---------------------------------------------------------------------------

class MockSpeechAlert(SpeechAlertInterface):
    """Logs speech events without audio output."""

    def __init__(self) -> None:
        self.last_message: str = ""

    def initialize(self) -> None:
        logger.info("MockSpeechAlert initialised (simulation).")

    def speak(self, text: str, blocking: bool = False) -> None:
        self.last_message = text
        logger.info("SPEECH [MOCK]: %s", text)

    def cleanup(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_speech(cfg_alerts) -> SpeechAlertInterface:
    backend = str(getattr(cfg_alerts, "speech_backend", "mock")).lower()
    if backend == "espeak":
        return EspeakSpeechAlert()
    if backend == "gtts":
        return GTTSSpeechAlert()
    return MockSpeechAlert()
