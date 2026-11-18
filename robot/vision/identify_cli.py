"""
robot/vision/identify_cli.py
----------------------------
التعرّف على مريض من الكاميرا. يلتقط صورة ويبحث في السجلّ: إن طابق وجهاً مسجّلاً
يطبع patient_id واسمه (وهو ما يُمرَّر للـConcierge لاستعادة التاريخ)، وإلا «مريض جديد».

تشغيل:
    python robot/vision/identify_cli.py
    python robot/vision/identify_cli.py --camera 0
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from robot.vision.camera import capture_image
from robot.vision.face import FaceRegistry, make_default_recognizer


def main() -> None:
    from robot.camera_config import CAMERA_SOURCE
    ap = argparse.ArgumentParser(description="التعرّف على مريض")
    ap.add_argument("--camera", default=CAMERA_SOURCE, help="فهرس أو رابط الكاميرا")
    args = ap.parse_args()

    reg = FaceRegistry(make_default_recognizer())
    print(f"عدد المرضى المسجّلين: {len(reg)}")
    print("التقاط صورة ... انظر للكاميرا")
    img = capture_image(args.camera)

    # اعرض كل من في الإطار (تعدّد الأشخاص)، ثم الوجه الأبرز/الأقرب بوصفه المريض الحالي.
    everyone = reg.identify_all(img)
    print(f"وجوه في الإطار: {len(everyone)}")
    for i, m in enumerate(everyone, 1):
        who = m.name or "(مجهول)"
        dist = f"{m.distance:.3f}" if m.distance is not None else "—"
        print(f"  {i}. {who}  bbox={m.bbox}  مسافة={dist}")

    match = reg.identify(img)
    if match is None:
        print("→ الوجه الأقرب: مريض جديد (لا تطابق). يمكن تسجيله بـ enroll_cli.py")
    else:
        pid, dist = match
        name = reg.name_of(pid) or "(بلا اسم)"
        print(f"→ تعرّفتُ على الأقرب: {name}")
        print(f"   patient_id = {pid}")
        print(f"   مسافة التطابق = {dist:.3f}  (أقل = أوثق)")


if __name__ == "__main__":
    main()
