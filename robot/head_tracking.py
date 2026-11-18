"""
robot/head_tracking.py
----------------------
تتبّع رأس المريض بسيرفوين: أفقي (pan) وعمودي (tilt).

يكتشف الوجه من الكاميرا بكاشف قويّ للزوايا (InsightFace/SCRFD إن توفّر، وإلا Haar أمامي+
جانبي) — لا embeddings ثقيلة، فقط موضع الوجه. ثم يحسب انزياح مركزه عن مركز الإطار ويحرّك
السيرفوين بتحكّم تناسبي (P-control) تدريجي لإبقاء الوجه في المنتصف. الكاشف القويّ يعالج
المشكلة القديمة: الوجه المائل/الجانبي كان لا يُكتشف فلا يُتتبَّع.

يعتمد على المعايرة في servo_config.py: PAN_SERVO / TILT_SERVO / PAN_DIR / TILT_DIR
وحدود كل سيرفو. عدّلها هناك بعد التجربة بـ servo_console.py.

المتطلّبات:  pip install opencv-python numpy   (والاتصال نفسه لا يحتاج pyserial مع WiFi)

تشغيل (WiFi، الافتراضي):
    python robot/head_tracking.py                  # 192.168.1.50 ، كاميرا 0
    python robot/head_tracking.py 192.168.1.50 --camera 1
تشغيل (USB Serial):
    python robot/head_tracking.py --serial COM5
بلا نافذة عرض (Pi بلا شاشة):
    python robot/head_tracking.py --no-display

داخل النافذة: اضغط q للخروج. عند الخروج تعود السيرفوات إلى وضع الراحة.
لو تحرّك الرأس عكس المريض: اقلب PAN_DIR و/أو TILT_DIR في servo_config.py.
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

from robot import servo_config as cfg
from robot.motor import MotorController, SerialTransport, WifiTransport
from robot.servo_config import CalibratedMotor

# ── معاملات التحكّم (اضبطها لو كانت الحركة عصبية أو بطيئة) ──
KP_PAN = 6.0        # درجة لكل وحدة خطأ أفقي (تقليل القيمة يمنع التخطي/overshooting)
KP_TILT = 5.0       # درجة لكل وحدة خطأ عمودي
DEADZONE = 0.1      # منطقة التجاهل بالمنتصف (تمنع الاهتزاز المستمر)
MAX_STEP = 10.0     # أقصى تغيّر زاوية في الإطار الواحد
EMA = 0.6          # تنعيم مركز الوجه لتجنب الحركات المفاجئة
FRAME_W, FRAME_H = 640, 480
MIN_FACE = 70       # أصغر وجه يُكتشف (بكسل)


def build_motor(args) -> MotorController:
    if args.serial:
        print(f"الاتصال عبر Serial بـ {args.serial} ...")
        return MotorController(SerialTransport(args.serial))
    print(f"الاتصال عبر WiFi بـ {args.ip}:8080 ...")
    return MotorController(WifiTransport(args.ip))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def run_tracker(camera_manager, ip: str, serial_port: str, no_display: bool, stop_event=None) -> None:
    import cv2

    class Args:
        pass
    args = Args()
    args.ip = ip
    args.serial = serial_port
    args.no_display = no_display

    # كاشف قويّ للزوايا (يعالج فشل التتبّع عند إمالة/انحراف الوجه) — كشف فقط، بلا تعرّف.
    from robot.vision.detector import make_default_detector
    detector = make_default_detector()

    # لا ننشئ الكاميرا هنا بل نستخدم الـ CameraManager الممرر لنا
    # cap = cv2.VideoCapture(args.camera)

    motor = CalibratedMotor(build_motor(args))
    print("PING ->", "PONG ✓" if motor._mc.ping() else "لا ردّ ✗")

    pan_cal, tilt_cal = cfg.SERVOS[cfg.PAN_SERVO], cfg.SERVOS[cfg.TILT_SERVO]
    pan, tilt = float(pan_cal.home), float(tilt_cal.home)
    motor.home()
    print(f"بدء التتبّع: PAN=S{cfg.PAN_SERVO}  TILT=S{cfg.TILT_SERVO}.  q للخروج.")

    ema_cx = ema_cy = None
    last_sent = (None, None)
    
    try:
        while True:
            if stop_event and stop_event.is_set():
                break

            frame = camera_manager.read()
            if frame is None:
                continue
            h, w = frame.shape[:2]
            boxes = detector.detect_boxes(frame)

            if boxes:
                # أكبر وجه = الأقرب/الأبرز
                x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                x, y, fw, fh = x1, y1, x2 - x1, y2 - y1
                cx, cy = x + fw / 2.0, y + fh / 2.0
                # تنعيم مركز الوجه (EMA)
                ema_cx = cx if ema_cx is None else EMA * ema_cx + (1 - EMA) * cx
                ema_cy = cy if ema_cy is None else EMA * ema_cy + (1 - EMA) * cy

                ex = (ema_cx - w / 2.0) / (w / 2.0)   # [-1,1] موجب = الوجه يمين المركز
                ey = (ema_cy - h / 2.0) / (h / 2.0)   # موجب = الوجه أسفل المركز

                if abs(ex) > DEADZONE:
                    pan += cfg.PAN_DIR * clamp(KP_PAN * ex, -MAX_STEP, MAX_STEP)
                if abs(ey) > DEADZONE:
                    tilt += cfg.TILT_DIR * clamp(KP_TILT * ey, -MAX_STEP, MAX_STEP)

                pan = clamp(pan, pan_cal.min_angle, pan_cal.max_angle)
                tilt = clamp(tilt, tilt_cal.min_angle, tilt_cal.max_angle)

                # أرسِل فقط عند تغيّر ملموس (يخفّف ازدحام الاتصال)
                tgt = (round(pan), round(tilt))
                if tgt != last_sent:
                    motor.move(cfg.PAN_SERVO, pan)
                    motor.move(cfg.TILT_SERVO, tilt)
                    last_sent = tgt

                if not args.no_display:
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 180, 160), 2)
                    cv2.circle(frame, (int(ema_cx), int(ema_cy)), 4, (0, 180, 160), -1)
                    # رسم خط من مركز الصورة إلى مركز الوجه يمثل قوة واتجاه السحب
                    cv2.line(frame, (w // 2, h // 2), (int(ema_cx), int(ema_cy)), (0, 0, 255), 2)
                    # عرض قيم الخطأ
                    cv2.putText(frame, f"Error X: {ex:.2f} | Error Y: {ey:.2f}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if not args.no_display:
                cv2.line(frame, (w // 2, 0), (w // 2, h), (90, 90, 90), 1)
                cv2.line(frame, (0, h // 2), (w, h // 2), (90, 90, 90), 1)
                cv2.putText(frame, f"pan {pan:.0f}  tilt {tilt:.0f}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 160), 2)
                
                # إظهار رسالة إذا كان السيرفو وصل للحد الأقصى
                if pan <= pan_cal.min_angle or pan >= pan_cal.max_angle:
                    cv2.putText(frame, "PAN LIMIT REACHED!", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if tilt <= tilt_cal.min_angle or tilt >= tilt_cal.max_angle:
                    cv2.putText(frame, "TILT LIMIT REACHED!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                cv2.imshow("head tracking (q=quit)", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nمقاطعة.")
    finally:
        print("إرجاع السيرفوات إلى الراحة وإغلاق ...")
        try:
            motor.home()
        except Exception:  # noqa: BLE001
            pass
        motor.close()
        # لم نعد مسؤولين عن تحرير الكاميرا هنا، يتم ذلك في المستدعي
        # cap.release()
        if not args.no_display:
            try:
                import cv2 as _c; _c.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass
        print("تمّ.")

def main() -> None:
    from robot.camera_config import CAMERA_SOURCE
    from robot.vision.camera_manager import CameraManager

    ap = argparse.ArgumentParser(description="تتبّع رأس المريض بسيرفوين (أفقي/عمودي)")
    ap.add_argument("ip", nargs="?", default="192.168.1.50", help="عنوان ESP32 (WiFi)")
    ap.add_argument("--serial", help="منفذ USB Serial بدل WiFi (مثل COM5)")
    ap.add_argument("--camera", default=CAMERA_SOURCE, help="فهرس أو رابط الكاميرا")
    ap.add_argument("--no-display", action="store_true", help="بلا نافذة عرض")
    args = ap.parse_args()
    
    # تشغيل الكاميرا من الطبقة المركزية
    cam_manager = CameraManager(args.camera)
    cam_manager.start()
    
    try:
        run_tracker(cam_manager, args.ip, args.serial, args.no_display)
    finally:
        cam_manager.stop()

if __name__ == "__main__":
    main()


