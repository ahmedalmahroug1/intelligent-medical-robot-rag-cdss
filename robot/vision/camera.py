"""
robot/vision/camera.py
----------------------
التقاط صورة من كاميرا (اللابتوب/USB الآن، كاميرا الـPi لاحقاً) عبر OpenCV.
الصورة تُعاد كمصفوفة numpy بصيغة BGR — وهي ما يتوقّعه FaceRecognizer.

التثبيت:  pip install opencv-python
"""
from __future__ import annotations

import numpy as np


from typing import Union
from robot.camera_config import CAMERA_SOURCE
from robot.vision.camera_manager import CameraManager

def capture_image(camera_source: Union[int, str] = CAMERA_SOURCE, warmup_frames: int = 5) -> np.ndarray:
    """
    التقط إطاراً واحداً من الكاميرا وأرجعه (BGR). 
    هذه الدالة الآن تعتبر طبقة توافق (Compatibility Layer) وتعتمد على CameraManager 
    خلف الكواليس لضمان استقرار الاتصال.
    """
    import time
    manager = CameraManager(camera_source)
    manager.start()
    
    try:
        # الانتظار حتى يتم التقاط إطار
        frame = None
        for _ in range(max(1, warmup_frames * 2)): # محاولات قصيرة
            frame = manager.read()
            if frame is not None:
                break
            time.sleep(0.1)
            
        if frame is None:
            raise RuntimeError(f"تعذّرت قراءة إطار من الكاميرا source={camera_source}")
        return frame
    finally:
        manager.stop()


def load_image(path: str) -> np.ndarray:
    """اقرأ صورة من ملف (BGR). مفيد للاختبار بصور محفوظة بدل الكاميرا."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"تعذّر قراءة الصورة: {path}")
    return img
