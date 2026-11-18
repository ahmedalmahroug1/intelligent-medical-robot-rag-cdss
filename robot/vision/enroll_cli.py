"""
robot/vision/enroll_cli.py
--------------------------
تسجيل وجه مريض من الكاميرا (المرحلة 1). يلتقط عدّة صور لزيادة المتانة ويخزّنها
تحت اسم المريض، فيُرجع patient_id الذي سيُستعاد في الزيارة القادمة.

تشغيل:
    python robot/vision/enroll_cli.py "Patient"
    python robot/vision/enroll_cli.py "Patient" --shots 5 --camera 0
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

from robot.vision.camera import capture_image
from robot.vision.face import FaceRegistry, make_default_recognizer


def main() -> None:
    from robot.camera_config import CAMERA_SOURCE
    ap = argparse.ArgumentParser(description="تسجيل وجه مريض")
    ap.add_argument("name", help="اسم المريض")
    ap.add_argument("--shots", type=int, default=3, help="عدد الصور الملتقطة")
    ap.add_argument("--camera", default=CAMERA_SOURCE, help="فهرس أو رابط الكاميرا")
    args = ap.parse_args()

    reg = FaceRegistry(make_default_recognizer())
    pid = None
    captured = 0
    for i in range(args.shots):
        print(f"التقاط الصورة {i + 1}/{args.shots} ... انظر للكاميرا")
        img = capture_image(args.camera)
        try:
            pid = reg.enroll(img, name=args.name, patient_id=pid)
            captured += 1
            print(f"  ✔ سُجّلت ({captured})")
        except ValueError:
            print("  ✗ لا وجه واضح في هذه اللقطة، أعِد المحاولة")
        time.sleep(1.0)

    if pid is None:
        print("لم تُلتقط أي صورة صالحة. حسّن الإضاءة وأعِد المحاولة.")
        sys.exit(1)
    print(f"\nتم تسجيل «{args.name}»  patient_id = {pid}  (لقطات: {captured})")


if __name__ == "__main__":
    main()

