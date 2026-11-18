"""
robot/vision/camera_manager.py
------------------------------
إدارة الكاميرا بشكل احترافي مع دعم خيوط المعالجة (Threading)،
تفادي تراكم الإطارات (Zero-latency buffer)،
إعادة الاتصال التلقائي (Auto Reconnect)،
وتوفير إحصائيات الأداء (FPS, Drop count).
"""
import time
import threading
import logging
import cv2
import numpy as np
from typing import Optional, Union

from robot.camera_config import CAMERA_RECONNECT_DELAY

logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self, source: Union[int, str]):
        # إذا تم تمرير الكاميرا كنص "0" من سطر الأوامر، يجب تحويلها إلى رقم
        # لأن OpenCV لا يدعم فتح كاميرا محلية برقم نصي
        if isinstance(source, str) and source.isdigit():
            self.source = int(source)
        else:
            self.source = source
            
        self.cap = None
        self.latest_frame: Optional[np.ndarray] = None
        self.running = False
        
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
        # إحصائيات الأداء
        self._frames_received = 0
        self._frames_dropped = 0
        self._reconnect_count = 0
        self._last_reconnect_time = 0.0
        self._start_time = 0.0
        self._fps_current = 0.0
        self._last_frame_time = 0.0
        self._is_connected = False

    def start(self):
        """بدء تشغيل الكاميرا في Thread خلفي."""
        if self.running:
            return
        
        self.running = True
        self._stop_event.clear()
        self._start_time = time.time()
        
        self._thread = threading.Thread(target=self._update, daemon=True, name="CameraManagerThread")
        self._thread.start()
        logger.info(f"Started CameraManager on source: {self.source}")

        # انتظار أولي بسيط للتأكد من جلب أول إطار (Warmup)
        # لا نجمد البرنامج تماماً، لكن نمنحه فرصة قصيرة للتحميل
        start_wait = time.time()
        while self.latest_frame is None and (time.time() - start_wait) < 2.0 and not self._stop_event.is_set():
            time.sleep(0.05)

    def stop(self):
        """إيقاف الـ Thread وتنظيف الموارد بشكل سليم."""
        if not self.running:
            return
        
        logger.info("Stopping CameraManager...")
        self.running = False
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            
        with self._lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self._is_connected = False
        logger.info("CameraManager stopped and resources released.")

    def read(self) -> Optional[np.ndarray]:
        """
        إرجاع أحدث إطار متاح في لحظة الاستدعاء (Zero-latency).
        يتم استدعاء هذا من الـ Thread الرئيسي للبرنامج.
        """
        with self._lock:
            if self.latest_frame is not None:
                # بمجرد قراءة الإطار من التطبيق الرئيسي، نعتبر أنه لم يعد "مفقوداً"
                # (سيقوم الـ Thread الخلفي بزيادة الـ dropped count إذا جلب إطاراً جديداً قبل قراءة هذا الإطار)
                pass
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_stats(self) -> dict:
        """إرجاع إحصائيات الأداء وحالة الكاميرا."""
        uptime = time.time() - self._start_time if self._start_time > 0 else 0
        avg_fps = self._frames_received / uptime if uptime > 0 else 0.0
        
        return {
            "connected": self._is_connected,
            "fps_current": round(self._fps_current, 1),
            "fps_average": round(avg_fps, 1),
            "frames_received": self._frames_received,
            "frames_dropped": self._frames_dropped,
            "reconnect_count": self._reconnect_count,
            "last_reconnect_time": self._last_reconnect_time
        }

    def _open_camera(self) -> bool:
        """فتح الاتصال بـ OpenCV بصمت."""
        try:
            import platform
            is_windows = platform.system().lower() == "windows"

            attempts = []
            if is_windows and isinstance(self.source, int):
                attempts = [
                    ("DSHOW", lambda: cv2.VideoCapture(self.source, cv2.CAP_DSHOW)),
                    ("MSMF", lambda: cv2.VideoCapture(self.source, cv2.CAP_MSMF)),
                    ("ANY", lambda: cv2.VideoCapture(self.source)),
                ]
            else:
                attempts = [("ANY", lambda: cv2.VideoCapture(self.source))]

            for backend_name, factory in attempts:
                if self.cap is not None:
                    self.cap.release()
                self.cap = factory()
                if self.cap.isOpened():
                    self._is_connected = True
                    logger.info(f"Camera successfully connected via {backend_name}.")
                    return True

            self._is_connected = False
            return False
        except Exception as e:
            logger.error(f"Error opening camera {self.source}: {e}")
            self._is_connected = False
            return False

    def _update(self):
        """الحلقة الخلفية (الـ Loop) التي تعمل بأقصى سرعة ممكنة."""
        frame_count = 0
        fps_timer = time.time()
        
        while not self._stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                self._is_connected = False
                self._reconnect_count += 1
                self._last_reconnect_time = time.time()
                # أول محاولة يجب أن تكون فورية. التأخير يخص إعادة المحاولة بعد
                # فشل حقيقي، وإلا يتسابق start() مع مهلة فحص الجاهزية القصيرة.
                if self._open_camera():
                    continue
                logger.warning(
                    f"Camera disconnected. Attempting to reconnect in "
                    f"{CAMERA_RECONNECT_DELAY}s..."
                )
                self._stop_event.wait(CAMERA_RECONNECT_DELAY)
                continue

            try:
                ok, frame = self.cap.read()
            except Exception as e:
                logger.error(f"Exception during cap.read(): {e}")
                ok = False

            if not ok:
                logger.warning("Failed to read frame. Dropping connection.")
                with self._lock:
                    if self.cap is not None:
                        self.cap.release()
                        self.cap = None
                    self._is_connected = False
                continue

            # الإطار نجح
            now = time.time()
            with self._lock:
                # إذا كان هناك إطار في latest_frame لم يقرأه التطبيق بعد، فهو "مفقود" Drop
                if self.latest_frame is not None and self._last_frame_time != now:
                     self._frames_dropped += 1
                     
                self.latest_frame = frame
                self._last_frame_time = now
                self._frames_received += 1
                self._is_connected = True

            # حساب الـ FPS الحالي (كل ثانية)
            frame_count += 1
            if now - fps_timer >= 1.0:
                self._fps_current = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer = now
