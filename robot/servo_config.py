"""
robot/servo_config.py
---------------------
معايرة كل سيرفو على حدة + غلاف CalibratedMotor فوق MotorController.

الفكرة: motor.py يبقى عاماً (0..180 لكل سيرفو). هذا الملف يضيف فوقه طبقة معايرة:
لكل سيرفو حدّ أدنى/أعلى آمن، وزاوية منتصف (home)، واتّجاه (invert) لو كان مركّباً معكوساً.
هكذا يحترم بقية النظام (تتبّع الرأس مثلاً) حدودَ كل سيرفو ولا يجبره على زاوية تؤذيه.

⚠️ القيم أدناه افتراضية مبدئياً (0..180، منتصف 90). بعد التجربة بـ servo_console.py:
  - عدّل min_angle/max_angle لكل سيرفو إلى المدى الآمن الذي لاحظته،
  - عيّن PAN_SERVO (الأفقي: يسار/يمين) و TILT_SERVO (العمودي: أعلى/أسفل)،
  - لو تحرّك السيرفو عكس المتوقّع اجعل invert=True (أو اقلب الاتجاه في تتبّع الرأس).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ServoCal:
    """معايرة سيرفو واحد. الزاوية المنطقية تُقصّ إلى [min_angle, max_angle] ثم تُعكس إن لزم."""
    name: str
    channel: int                 # قناة PCA9685 (مرجع، مطابق لـ test_sweep.py)
    min_angle: int = 0
    max_angle: int = 180
    home: int = 90               # زاوية الراحة/المنتصف
    invert: bool = False         # True لو كان اتّجاه السيرفو معكوساً ميكانيكياً

    def clamp(self, angle: float) -> int:
        return int(max(self.min_angle, min(self.max_angle, round(angle))))

    def to_physical(self, logical: float) -> int:
        """زاوية منطقية → زاوية فيزيائية تُرسَل للـESP32 (تقصّ ثم تعكس داخل المدى نفسه)."""
        a = self.clamp(logical)
        if self.invert:
            a = self.min_angle + self.max_angle - a
        return int(max(0, min(180, a)))


# ── معايرة السيرفوات الستة (id → ServoCal) — القنوات مطابقة لـ test_sweep.py ──
SERVOS: dict[int, ServoCal] = {
    0: ServoCal("servo0", channel=1,  min_angle=0, max_angle=180, home=90),
    1: ServoCal("servo1", channel=3,  min_angle=0, max_angle=180, home=90),
    2: ServoCal("servo2", channel=9,  min_angle=0, max_angle=180, home=90),
    3: ServoCal("servo3", channel=11, min_angle=0, max_angle=180, home=90),
    4: ServoCal("servo4", channel=13, min_angle=0, max_angle=180, home=90),
    5: ServoCal("servo5", channel=15, min_angle=30, max_angle=110, home=70),
}

# ── سيرفوات تتبّع الرأس (املأها بعد التجربة) ──
PAN_SERVO = 0      # الأفقي: يحرّك الرأس يسار/يمين
TILT_SERVO = 5     # العمودي: يحرّك الرأس أعلى/أسفل

# اتّجاه التتبّع: لو تحرّك الرأس عكس المريض، اقلب الإشارة (+1 ↔ -1).
PAN_DIR = -1
TILT_DIR = +1


class CalibratedMotor:
    """غلاف فوق MotorController يطبّق المعايرة لكل سيرفو (قصّ + عكس + home)."""

    def __init__(self, motor, servos: dict[int, ServoCal] | None = None) -> None:
        self._mc = motor
        self._servos = servos if servos is not None else SERVOS

    def cal(self, sid: int) -> ServoCal:
        return self._servos[sid]

    def move(self, sid: int, logical_angle: float) -> int:
        """حرّك السيرفو إلى زاوية منطقية (تُعايَر تلقائياً). يُرجع الزاوية الفيزيائية المُرسَلة."""
        phys = self._servos[sid].to_physical(logical_angle)
        self._mc.set(sid, phys)
        return phys

    def home_one(self, sid: int) -> None:
        c = self._servos[sid]
        self._mc.set(sid, c.to_physical(c.home))

    def home(self) -> None:
        """أرجِع كل السيرفوات إلى زاوية الراحة (home) دفعةً واحدة."""
        angles = []
        for sid in range(self._mc.NUM_SERVOS):
            c = self._servos.get(sid)
            angles.append(c.to_physical(c.home) if c else 90)
        self._mc.set_all(angles)

    def close(self) -> None:
        self._mc.close()
