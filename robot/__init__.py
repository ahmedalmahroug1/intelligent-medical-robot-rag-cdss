"""
robot/
------
الطبقة الحركية لروبوت المساعد الطبي (المرحلة 3: التجسيد).
العقل بايثون على الحاسوب يتحكّم بـ6 سيرفوات عبر ESP32 + PCA9685.
"""
from robot.motor import MotorController, SerialTransport, Transport, WifiTransport

__all__ = ["MotorController", "SerialTransport", "WifiTransport", "Transport"]
