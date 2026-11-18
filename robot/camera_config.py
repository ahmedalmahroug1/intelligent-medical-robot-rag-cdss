"""
robot/camera_config.py
----------------------
إعدادات الكاميرا المركزية للمشروع.
تعتمد على المتغيرات البيئية لسهولة النشر دون تعديل الكود.
"""
import os

# إعدادات عنوان Raspberry Pi
PI_HOST = os.environ.get("PI_HOST", "192.168.1.50")
PI_PORT = os.environ.get("PI_PORT", "8080")

# تم تفعيل كاميرا الـ Raspberry Pi افتراضياً بناءً على طلبك
USE_PI_CAMERA = os.environ.get("USE_PI_CAMERA", "true").lower() == "true"

if USE_PI_CAMERA:
    CAMERA_SOURCE = f"http://{PI_HOST}:{PI_PORT}/stream"
else:
    # الرجوع للكاميرا المحلية (0, 1, إلخ)
    raw_source = os.environ.get("CAMERA_SOURCE", "0")
    # تحويل القيمة إلى int إذا كانت تحوي أرقاماً فقط لأن OpenCV يتطلب ذلك للكاميرات المحلية
    CAMERA_SOURCE = int(raw_source) if raw_source.isdigit() else raw_source

# الوقت بالثواني للمحاولة مرة أخرى عند انقطاع الاتصال
CAMERA_RECONNECT_DELAY = 2.0

