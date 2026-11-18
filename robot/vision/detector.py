"""
robot/vision/detector.py
------------------------
كشف موضع الوجه للتتبّع الحيّ (مقابل التعرّف على الهوية في face.py).

المشكلة القديمة: كان التتبّع يعتمد كاشف Haar الأمامي (haarcascade_frontalface_default.xml)
الذي يكتشف الوجه فقط قريباً من الوضع الأمامي المستقيم؛ فعند إمالة الرأس أو انحرافه (زاوية
جانبية) لا يضع إطاراً ولا يتتبّع. الحلّ: كاشف قويّ للزوايا.

التصميم (مثل بقية المشروع): تجريد `FaceDetector` قابل للتبديل:
  - `InsightFaceDetector` — كاشف SCRFD من InsightFace (نفس عائلة RetinaFace): قويّ للزوايا
    والانحراف والوجوه الصغيرة. وضع الكشف فقط (بلا embeddings) وبحجم إدخال أصغر (سرعة أعلى).
  - `HaarFaceDetector` — احتياطي بلا اعتماديات جديدة: أمامي + جانبي + قلب أفقي لالتقاط
    الوضعية الجانبية المعاكسة أيضاً (أوسع تغطية من الأمامي وحده).

التتبّع لا يحتاج هوية، بل صندوق الوجه فقط — لذلك `detect_boxes` تُرجع صناديق (x1,y1,x2,y2).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FaceDetector(ABC):
    """يكتشف مواضع الوجوه في إطار (للتتبّع). لا يعنيه من صاحب الوجه."""

    @abstractmethod
    def detect_boxes(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """أرجع صناديق الوجوه (x1, y1, x2, y2). فارغة إن لا وجه."""


class InsightFaceDetector(FaceDetector):
    """كاشف SCRFD من InsightFace — قويّ للزوايا/الانحراف. كشف فقط (أخفّ من buffalo_l كاملاً)."""

    def __init__(self, det_size: tuple[int, int] = (320, 320), ctx_id: int = -1) -> None:
        self._det_size = det_size          # أصغر = أسرع (يكفي للتتبّع)
        self._ctx = ctx_id                 # -1 = CPU (مناسب للـPi)
        self._app = None

    def _get_app(self):
        if self._app is None:
            from insightface.app import FaceAnalysis  # استيراد كسول
            app = FaceAnalysis(allowed_modules=["detection"])   # كشف فقط
            app.prepare(ctx_id=self._ctx, det_size=self._det_size)
            self._app = app
        return self._app

    def detect_boxes(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        faces = self._get_app().get(image)
        return [tuple(int(v) for v in f.bbox[:4]) for f in faces]


class HaarFaceDetector(FaceDetector):
    """احتياطي بلا اعتماديات جديدة: كاسكيد أمامي + جانبي + قلب أفقي للوضعية المعاكسة.

    أوسع تغطيةً للزوايا من الأمامي وحده، لكنه يبقى أضعف من InsightFace."""

    def __init__(self, min_size: int = 70) -> None:
        import cv2
        base = cv2.data.haarcascades
        self._front = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        self._profile = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")
        self._min = min_size

    def detect_boxes(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes: list[tuple[int, int, int, int]] = []

        def _run(cascade, on_flipped=False):
            if cascade.empty():
                return
            src = cv2.flip(gray, 1) if on_flipped else gray
            for (x, y, w, h) in cascade.detectMultiScale(
                    src, scaleFactor=1.2, minNeighbors=5, minSize=(self._min, self._min)):
                x1 = (gray.shape[1] - (x + w)) if on_flipped else x   # أعِد الإحداثي بعد القلب
                boxes.append((int(x1), int(y), int(x1 + w), int(y + h)))

        _run(self._front)                       # وجه أمامي
        _run(self._profile)                     # وجه جانبي (يواجه جهةً)
        _run(self._profile, on_flipped=True)    # الجهة المعاكسة (بقلب الصورة)
        return boxes


def make_default_detector(prefer_insightface: bool = True,
                          det_size: tuple[int, int] = (320, 320)) -> FaceDetector:
    """اختر أقوى كاشف متاح للتتبّع: InsightFace إن ثُبِّت، وإلا Haar (أمامي+جانبي)."""
    if prefer_insightface:
        try:
            import insightface  # noqa: F401 — فحص توفّر فقط
            return InsightFaceDetector(det_size=det_size)
        except Exception as exc:  # noqa: BLE001
            print(f"[vision] كاشف InsightFace غير متاح ({exc}) — سأستخدم Haar (أمامي+جانبي).")
    return HaarFaceDetector()
