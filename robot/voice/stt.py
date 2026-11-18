"""
robot/voice/stt.py
------------------
تحويل الكلام إلى نص (الروبوت يسمع المريض). تجريد `SpeechToText` قابل للتبديل،
وتطبيق `WhisperSTT` يسجّل من المايك (sounddevice) ويفرّغ بـfaster-whisper (يدعم
العربية والإنجليزية تلقائياً، محلياً).

التسجيل ينتهي تلقائياً بعد صمتٍ قصير عقب الكلام (endpointing بسيط بالطاقة)، بحدّ
أقصى للمدة كي لا يعلق.

التثبيت:  pip install faster-whisper sounddevice
أول تشغيل ينزّل نموذج Whisper مرّة واحدة (يحتاج إنترنت مرّة).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

SAMPLE_RATE = 16000


class SpeechToText(ABC):
    @abstractmethod
    def listen(self) -> str:
        """سجّل من المايك وأرجع النص المفرّغ (قد يكون فارغاً إن لم يُسمع كلام)."""

    def close(self) -> None:
        ...


class WhisperSTT(SpeechToText):
    """تسجيل من المايك + تفريغ بـfaster-whisper (لغة تلقائية: عربي/إنجليزي)."""

    def __init__(self, model_size: str = "small", max_seconds: float = 12.0,
                 silence_tail_s: float = 1.2, energy_threshold: float = 0.012) -> None:
        from faster_whisper import WhisperModel
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self._max_seconds = max_seconds
        self._silence_tail = silence_tail_s
        self._threshold = energy_threshold

    def _record(self) -> np.ndarray:
        """يسجّل حتى صمتٍ تالٍ للكلام، أو حتى الحدّ الأقصى."""
        import sounddevice as sd
        block = int(SAMPLE_RATE * 0.1)            # كتل 100ms
        frames: list[np.ndarray] = []
        speech_started = False
        silent_blocks = 0
        max_blocks = int(self._max_seconds / 0.1)
        tail_blocks = int(self._silence_tail / 0.1)

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
            for _ in range(max_blocks):
                data, _ = stream.read(block)
                mono = data[:, 0]
                frames.append(mono)
                rms = float(np.sqrt(np.mean(mono ** 2)) + 1e-9)
                if rms >= self._threshold:
                    speech_started = True
                    silent_blocks = 0
                elif speech_started:
                    silent_blocks += 1
                    if silent_blocks >= tail_blocks:
                        break
        return np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)

    def listen(self) -> str:
        audio = self._record()
        if audio.size == 0:
            return ""
        segments, _info = self._model.transcribe(audio, beam_size=1)
        return " ".join(seg.text for seg in segments).strip()


class TypedSTT(SpeechToText):
    """بديل بلا مايك (للتصحيح): يقرأ ردّ المريض من لوحة المفاتيح."""

    def listen(self) -> str:
        try:
            return input("⌨️  أنت (المريض): ").strip()
        except (EOFError, KeyboardInterrupt):
            return ""
