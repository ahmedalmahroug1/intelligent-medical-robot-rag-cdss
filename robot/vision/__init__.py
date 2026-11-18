"""
robot/vision/
-------------
طبقة الرؤية لروبوت المساعد الطبي (المرحلة 1: التعرّف على وجوه المرضى).
الكاميرا تُنتج patient_id يدخل نفس مسار استعادة تاريخ المريض في الـConcierge.
"""
from robot.vision.detector import (
    FaceDetector,
    HaarFaceDetector,
    InsightFaceDetector,
    make_default_detector,
)
from robot.vision.face import (
    DeepFaceRecognizer,
    DetectedFace,
    FaceMatch,
    FaceRecognizer,
    FaceRegistry,
    InsightFaceRecognizer,
    cosine_distance,
    make_default_recognizer,
)

__all__ = [
    # التعرّف على الهوية
    "FaceRecognizer",
    "InsightFaceRecognizer",
    "DeepFaceRecognizer",
    "FaceRegistry",
    "DetectedFace",
    "FaceMatch",
    "cosine_distance",
    "make_default_recognizer",
    # كشف الوجه للتتبّع
    "FaceDetector",
    "InsightFaceDetector",
    "HaarFaceDetector",
    "make_default_detector",
]
