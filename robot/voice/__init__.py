"""
robot/voice/
------------
طبقة الصوت لروبوت المساعد الطبي (المرحلة 2: الصوت).
TTS (الروبوت يتكلّم) + STT (الروبوت يسمع) خلف تجريدين قابلين للتبديل،
مع بدائل نصّية (PrintTTS/TypedSTT) للتصحيح بلا أجهزة صوت.
"""
from robot.voice.stt import SpeechToText, TypedSTT, WhisperSTT
from robot.voice.tts import EdgeTTS, OpenAITTS, PrintTTS, TextToSpeech

__all__ = [
    "TextToSpeech", "OpenAITTS", "EdgeTTS", "PrintTTS",
    "SpeechToText", "WhisperSTT", "TypedSTT",
]
