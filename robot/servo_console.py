"""
robot/servo_console.py
----------------------
أداة تفاعلية لتجربة زوايا السيرفوات الستة مباشرةً — لاكتشاف ماذا يحرّك كل سيرفو،
ومداه الآمن (min/max)، وأيّها أفقي/عمودي للرأس، قبل ضبط المعايرة (servo_config.py)
وتتبّع الرأس (head_tracking.py).

لا تلمس motor.py — تبني فوق MotorController الموجود فقط.

تشغيل (WiFi، الافتراضي):
    python robot/servo_console.py                  # 192.168.1.50:8080
    python robot/servo_console.py 192.168.1.50    # IP آخر
تشغيل (USB Serial):
    python robot/servo_console.py --serial COM5     # ويندوز، أو /dev/ttyUSB0 على لينكس/Pi

الأوامر داخل الأداة:
    <id> <angle>     حرّك السيرفو id (0..5) إلى زاوية 0..180        مثال:  0 120
    n <id> <delta>   زِد/أنقص زاوية السيرفو تدريجياً (لاكتشاف الحدود)  مثال:  n 0 -5
    all <angle>      اضبط كل السيرفوات الستة على زاوية واحدة
    sweep <id>       مسح بطيء للسيرفو عبر مداه لرؤية اتّجاه حركته
    sweep            مسح كل السيرفوات واحداً تلو الآخر
    home             إرجاع الكل إلى 90 (المنتصف)
    show             اطبع آخر زاوية مُرسَلة لكل سيرفو
    ping             فحص الاتصال (PONG)
    help             إظهار هذه القائمة
    q  /  quit       خروج (يُرجع الكل للمنتصف أولاً)

أثناء التجربة دوّن لكل سيرفو: ماذا يحرّك فيزيائياً، وأصغر/أكبر زاوية آمنة، وأي سيرفو
هو الأفقي (يسار/يمين) وأيّها العمودي (أعلى/أسفل) للرأس — سنضعها في servo_config.py.
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, ".")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from robot.motor import MotorController, SerialTransport, WifiTransport

NUM = MotorController.NUM_SERVOS
CHANNELS = [1, 3, 9, 11, 13, 15]      # خريطة PCA9685 للسيرفوات 0..5 (مرجع للعرض فقط)
HOME = 90


def build_motor(args) -> MotorController:
    if args.serial:
        print(f"الاتصال عبر Serial بـ {args.serial} ...")
        return MotorController(SerialTransport(args.serial))
    print(f"الاتصال عبر WiFi بـ {args.ip}:8080 ...")
    return MotorController(WifiTransport(args.ip))


def sweep_one(mc: MotorController, sid: int, last: dict[int, int]) -> None:
    """مسح بطيء 90→150→30→90 لرؤية اتّجاه ومدى السيرفو."""
    print(f"  مسح السيرفو {sid} (القناة {CHANNELS[sid]}) ...")
    for ang in [*range(90, 151, 10), *range(150, 29, -10), *range(30, 91, 10)]:
        mc.set(sid, ang)
        last[sid] = ang
        time.sleep(0.18)
    print(f"  انتهى مسح السيرفو {sid}.")


def print_help() -> None:
    print(__doc__.split("الأوامر داخل الأداة:")[1].split("أثناء التجربة")[0].rstrip())


def main() -> None:
    ap = argparse.ArgumentParser(description="أداة تفاعلية لتجربة زوايا السيرفوات")
    ap.add_argument("ip", nargs="?", default="192.168.1.50", help="عنوان ESP32 (WiFi)")
    ap.add_argument("--serial", help="منفذ USB Serial بدل WiFi (مثل COM5)")
    args = ap.parse_args()

    mc = build_motor(args)
    print("PING ->", "PONG ✓" if mc.ping() else "لا ردّ ✗ (تحقّق من الاتصال/التغذية)")

    last: dict[int, int] = {i: HOME for i in range(NUM)}
    print(f"تصفير: كل السيرفوات إلى {HOME}° (المنتصف).")
    try:
        mc.set_all([HOME] * NUM)
    except Exception as exc:  # noqa: BLE001
        print("تعذّر التصفير:", exc)

    print("\nاكتب الأوامر (help للقائمة، q للخروج). مثال:  0 120")
    print(f"خريطة القنوات: " + ", ".join(f"S{i}→ch{CHANNELS[i]}" for i in range(NUM)))

    try:
        while True:
            try:
                line = input(">> ").strip()
            except EOFError:
                break
            if not line:
                continue
            parts = line.split()
            cmd = parts[0].lower()

            if cmd in ("q", "quit", "exit", "خروج"):
                break
            if cmd in ("help", "h", "?"):
                print_help(); continue
            if cmd == "ping":
                print("PONG ✓" if mc.ping() else "لا ردّ ✗"); continue
            if cmd == "show":
                print("  " + " | ".join(f"S{i}={last[i]}°" for i in range(NUM))); continue
            if cmd == "home":
                mc.set_all([HOME] * NUM)
                for i in range(NUM):
                    last[i] = HOME
                print(f"  الكل → {HOME}°"); continue
            if cmd == "all" and len(parts) == 2:
                ang = int(parts[1])
                mc.set_all([ang] * NUM)
                for i in range(NUM):
                    last[i] = max(0, min(180, ang))
                print(f"  الكل → {ang}°"); continue
            if cmd == "sweep":
                if len(parts) == 2:
                    sid = int(parts[1])
                    if 0 <= sid < NUM:
                        sweep_one(mc, sid, last)
                    else:
                        print("  id خارج المدى 0..5")
                else:
                    for sid in range(NUM):
                        sweep_one(mc, sid, last)
                continue
            if cmd == "n" and len(parts) == 3:        # nudge
                sid, delta = int(parts[1]), int(parts[2])
                if not 0 <= sid < NUM:
                    print("  id خارج المدى 0..5"); continue
                ang = max(0, min(180, last[sid] + delta))
                resp = mc.set(sid, ang); last[sid] = ang
                print(f"  S{sid} {ang}°  (ESP32: {resp})"); continue

            # الصيغة الأساسية:  <id> <angle>
            if len(parts) == 2 and parts[0].isdigit():
                sid, ang = int(parts[0]), int(parts[1])
                if not 0 <= sid < NUM:
                    print("  id خارج المدى 0..5"); continue
                if not 0 <= ang <= 180:
                    print("  الزاوية يجب أن تكون 0..180"); continue
                resp = mc.set(sid, ang); last[sid] = ang
                print(f"  S{sid} {ang}°  (ESP32: {resp})"); continue

            print("  أمر غير مفهوم. اكتب help.")
    except KeyboardInterrupt:
        print("\nمقاطعة.")
    finally:
        print(f"إرجاع الكل إلى {HOME}° وإغلاق الاتصال ...")
        try:
            mc.set_all([HOME] * NUM)
        except Exception:  # noqa: BLE001
            pass
        mc.close()
        print("تمّ.")


if __name__ == "__main__":
    main()


