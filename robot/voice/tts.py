"""
robot/voice/tts.py
------------------
تحويل النص إلى كلام (الروبوت يتكلّم). تجريد `TextToSpeech` قابل للتبديل، وتطبيقان:
  - `OpenAITTS` (الافتراضي): أصوات OpenAI — صوت واحد متعدد اللغات ينطق العربية والإنجليزية
    بنفس الشخصية الصوتية. يحتاج OPENAI_API_KEY (نفس مفتاح النماذج).
  - `EdgeTTS`: أصوات Microsoft العصبية (صوت لكل لغة) عبر edge-tts.
كلاهما يشغّل الصوت بـpygame.

التثبيت:  pip install openai pygame        (و edge-tts لبديل EdgeTTS)
الضبط  :  set OPENAI_TTS_MODEL=gpt-4o-mini-tts   (اختياري)
          set OPENAI_TTS_VOICE=alloy             (اختياري: alloy/nova/onyx/shimmer/...)
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import threading
from abc import ABC, abstractmethod

# صوت عربي ذكوري وآخر إنجليزي. غيّرها كما تحب (مثلاً ar-EG-SalmaNeural أنثوي).
VOICE_AR = "ar-SA-HamedNeural"
VOICE_EN = "en-US-GuyNeural"


def _run_async(coro) -> None:
    """شغّل coroutine سواء وُجدت حلقة async قيد التشغيل أم لا. لو كانت هناك حلقة
    (نُودينا من داخل async)، نشغّلها في خيط منفصل بحلقته الخاصة لتجنّب
    'asyncio.run() cannot be called from a running event loop'."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return
    err: list[BaseException] = []

    def _worker() -> None:
        try:
            asyncio.run(coro)
        except BaseException as e:  # noqa: BLE001
            err.append(e)

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    if err:
        raise err[0]


class TextToSpeech(ABC):
    @abstractmethod
    def speak(self, text: str, lang: str = "ar") -> None:
        """انطق النص (lang: 'ar' أو 'en'). محجوب حتى ينتهي النطق."""

    def close(self) -> None:
        ...


def _init_pygame():
    import pygame  # تشغيل mp3 موثوق على Windows/الـPi
    pygame.mixer.init()
    return pygame


def _play_mp3_blocking(pygame_mod, path: str) -> None:
    """شغّل ملف mp3 وانتظر انتهاءه (مشترك بين كل تطبيقات TTS)."""
    pygame_mod.mixer.music.load(path)
    pygame_mod.mixer.music.play()
    while pygame_mod.mixer.music.get_busy():
        pygame_mod.time.wait(80)
    pygame_mod.mixer.music.unload()


class OpenAITTS(TextToSpeech):
    """أصوات OpenAI: نموذج TTS متعدد اللغات — الصوت نفسه ينطق العربية والإنجليزية، فلا يتغيّر
    طابع الروبوت الصوتي بين اللغتين. يفشل مبكراً (في البناء) إن غاب OPENAI_API_KEY، حتى
    يستطيع المستدعي السقوط إلى بديل آخر قبل بدء المقابلة."""

    def __init__(self, model: str | None = None, voice: str | None = None) -> None:
        from openai import OpenAI  # يرفع OpenAIError مبكراً إن غاب المفتاح
        self._client = OpenAI()
        self._model = model or os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
        self._voice = voice or os.environ.get("OPENAI_TTS_VOICE", "alloy")
        self._pygame = _init_pygame()

    def speak(self, text: str, lang: str = "ar") -> None:
        text = (text or "").strip()
        if not text:
            return
        path = tempfile.mktemp(suffix=".mp3")
        # الصوت متعدد اللغات؛ `lang` غير مطلوب للتوليد لكنه جزء من عقد TextToSpeech.
        with self._client.audio.speech.with_streaming_response.create(
                model=self._model, voice=self._voice, input=text) as resp:
            resp.stream_to_file(path)
        _play_mp3_blocking(self._pygame, path)


class EdgeTTS(TextToSpeech):
    """أصوات عصبية عبر edge-tts + تشغيل بـpygame. يحتاج إنترنت."""

    def __init__(self, voice_ar: str = VOICE_AR, voice_en: str = VOICE_EN) -> None:
        self._voice_ar = voice_ar
        self._voice_en = voice_en
        self._pygame = _init_pygame()

    def speak(self, text: str, lang: str = "ar") -> None:
        text = (text or "").strip()
        if not text:
            return
        import edge_tts
        voice = self._voice_ar if lang == "ar" else self._voice_en
        path = tempfile.mktemp(suffix=".mp3")

        async def _synth() -> None:
            await edge_tts.Communicate(text, voice).save(path)

        _run_async(_synth())
        _play_mp3_blocking(self._pygame, path)


class PrintTTS(TextToSpeech):
    """بديل بلا صوت (للتصحيح): يطبع ما كان الروبوت سيقوله."""

    def speak(self, text: str, lang: str = "ar") -> None:
        print(f"🔊 الروبوت: {text}")
