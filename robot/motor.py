"""
robot/motor.py
--------------
طبقة التحكّم بالسيرفوات من بايثون (العقل على الحاسوب).

التصميم يعكس فلسفة المشروع: الاعتماد على تجريد `Transport` لا على وسيلة نقل محدّدة،
فالانتقال من USB إلى WiFi لاحقاً = كتابة `WifiTransport` بنفس الواجهة، دون تغيير
`MotorController` ولا منطق الحركة الذي يبرمجه/يستدعيه الـLLM.

الاستخدام:
    from robot.motor import MotorController, SerialTransport
    mc = MotorController(SerialTransport("COM5"))   # Windows: COMx | Linux/Pi: /dev/ttyUSB0
    assert mc.ping()
    mc.set(0, 90)                     # سيرفو واحد (id 0 -> القناة 1 على PCA9685)
    mc.set_all([90, 90, 45, 135, 90, 0])
    mc.close()
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod


class Transport(ABC):
    """قناة نقل مجرّدة إلى الـESP32. SerialTransport الآن، WifiTransport لاحقاً."""

    @abstractmethod
    def send(self, line: str) -> None:
        """أرسل سطر أمر واحد (تُضاف \\n تلقائياً)."""

    @abstractmethod
    def readline(self) -> str:
        """اقرأ سطر ردّ واحداً (منزوع المسافات)، أو سلسلة فارغة عند انتهاء المهلة."""

    def close(self) -> None:
        """أغلق القناة. تجاوزها إن لزم."""


class SerialTransport(Transport):
    """نقل عبر USB Serial باستخدام pyserial (pip install pyserial)."""

    def __init__(self, port: str, baud: int = 115200, boot_wait: float = 2.0,
                 timeout: float = 1.0) -> None:
        import serial  # pyserial
        self._ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(boot_wait)            # انتظار إقلاع ESP32 بعد فتح المنفذ
        self._ser.reset_input_buffer()   # تجاهل رسالة READY الأولية

    def send(self, line: str) -> None:
        self._ser.write((line.rstrip("\n") + "\n").encode("ascii"))

    def readline(self) -> str:
        return self._ser.readline().decode("ascii", errors="ignore").strip()

    def close(self) -> None:
        self._ser.close()


class WifiTransport(Transport):
    """نقل عبر WiFi (خادم TCP على الـESP32). نفس واجهة SerialTransport تماماً."""

    def __init__(self, host: str, port: int = 8080, timeout: float = 2.0) -> None:
        import socket
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.settimeout(timeout)
        self._buf = b""

    def send(self, line: str) -> None:
        self._sock.sendall((line.rstrip("\n") + "\n").encode("ascii"))

    def readline(self) -> str:
        import socket
        while b"\n" not in self._buf:
            try:
                chunk = self._sock.recv(256)
            except socket.timeout:
                break
            if not chunk:
                break
            self._buf += chunk
        if b"\n" in self._buf:
            raw, self._buf = self._buf.split(b"\n", 1)
        else:
            raw, self._buf = self._buf, b""
        return raw.decode("ascii", errors="ignore").strip()

    def close(self) -> None:
        self._sock.close()


class MotorController:
    """تحكّم عالي المستوى بالسيرفوات الستة، مستقل عن وسيلة النقل."""

    NUM_SERVOS = 6

    def __init__(self, transport: Transport) -> None:
        self._t = transport
        self._lock = threading.Lock()    # أمر/ردّ واحد في كل مرة

    @staticmethod
    def _clamp(angle: int) -> int:
        return max(0, min(180, int(angle)))

    def set(self, servo_id: int, angle: int) -> str:
        """حرّك سيرفو واحداً (0..5) إلى زاوية 0..180. يُرجع ردّ الـESP32."""
        if not 0 <= servo_id < self.NUM_SERVOS:
            raise ValueError(f"servo_id يجب أن يكون 0..{self.NUM_SERVOS - 1}")
        with self._lock:
            self._t.send(f"S{servo_id} {self._clamp(angle)}")
            return self._t.readline()

    def set_all(self, angles: list[int]) -> str:
        """اضبط السيرفوات الستة دفعةً واحدة. `angles` قائمة من 6 زوايا."""
        if len(angles) != self.NUM_SERVOS:
            raise ValueError(f"angles يجب أن تكون {self.NUM_SERVOS} قيم")
        vals = " ".join(str(self._clamp(a)) for a in angles)
        with self._lock:
            self._t.send("A " + vals)
            return self._t.readline()

    def ping(self) -> bool:
        """فحص الاتصال: True إذا ردّ الـESP32 بـPONG."""
        with self._lock:
            self._t.send("PING")
            return self._t.readline() == "PONG"

    def close(self) -> None:
        self._t.close()
