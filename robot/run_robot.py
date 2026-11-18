"""
robot/run_robot.py
------------------
المشروع كاملاً كروبوت: صوت + تعرّف على الوجه + ذاكرة مريض في قاعدة البيانات + الدائرة الطبية.

التدفّق لكل زيارة:
  1) الكاميرا تلتقط وجه المريض → FaceRegistry.identify.
       - معروف   : نُحمّل سجلّه من SQLite ونرحّب به باسمه (التاريخ مُستعاد).
       - جديد    : نسأله اسمه صوتاً، نُسجّل وجهه، ننشئ له سجلّاً في قاعدة البيانات.
  2) الـConcierge يجري المقابلة صوتاً (المريض يتكلّم → Whisper، والروبوت يردّ → edge-tts)،
     يستشير محرّك الـCDSS، ويعرض التشخيص المبدئي حين تكتمل المعلومات.
  3) نكتب البيانات المُستخرَجة (عمر/جنس/أدوية/حساسية + التشخيص) في سجلّ المريض —
     كتابة *منظَّمة ومُتحقَّقة* عبر PatientRepository، لا SQL حرّ.

أوضاع التشغيل:
    python robot/run_robot.py                            # كامل: وجه + صوت
    python robot/run_robot.py --mode text                # بلا صوت (كتابة) للتصحيح
    python robot/run_robot.py --no-face                  # بلا كاميرا (اسم بالكتابة/الصوت)
    python robot/run_robot.py --camera 0 --serial COM5   # كمبيوتر + كاميرا USB + ESP32 عبر USB

الإعداد:
    set OPENAI_API_KEY=...        (مطلوب لمزوّد OpenAI)
    set CONCIERGE_MODEL=gpt-4o (اختياري، الاستدلال) · set CONCIERGE_MODEL_FAST=gpt-4o-mini (المحادثة)
    set CDSS_MODEL=gpt-4o (اختياري) · set CDSS_MODEL_FAST=gpt-4o-mini (اختياري)
    pip install -r robot/requirements.txt
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

sys.path.insert(0, ".")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.concierge import ConciergeAgent, ConciergeSafety, prompts as concierge_prompts
from app.concierge.identity import IdentityRegistry, detect_language, parse_name
from app.concierge.visit_writer import persist_concierge_visit
from app.concierge.schemas import ClinicalPicture
from app.core.enums import Sex
from app.core.state import Allergy, Medication, PastVisitSummary, PatientRecord
from app.llm.base import ModelTier
from app.llm.factory import make_gateway
from app.knowledge import build_knowledge_agent, build_reasoning_facade
from app.persistence.database import Database
from app.persistence.store import ConsultationStore
from app.safety.guard import SafetyGuard

# النماذج بطبقتين: سريعة للمحادثة (استخراج/صياغة سؤال) وقوية للاستدلال الطبي.
# gpt-5.4 بطيء (~3.5ث/نداء) فلا يُستخدم افتراضياً؛ gpt-4o-mini (~1ث) للطبقة السريعة و gpt-4o للاستدلال.
CONCIERGE_FAST = os.environ.get("CONCIERGE_MODEL_FAST", "gpt-4o-mini")
CONCIERGE_REASONING = os.environ.get("CONCIERGE_MODEL", "gpt-4o")
CDSS_FAST = os.environ.get("CDSS_MODEL_FAST", "gpt-4o-mini")
CDSS_REASONING = os.environ.get("CDSS_MODEL", "gpt-4o")
EXIT_WORDS = {"exit", "quit", "خروج", "انهاء", "إنهاء"}


def _now():
    return datetime.now(timezone.utc)


@dataclass
class Identification:
    """نتيجة تحديد هوية المريض في بداية الزيارة. `sex`/`age` تقديران من الوجه (قد يكونا None)
    يُملآن الفراغ في سجلّ المريض فلا يُسأل عنهما — لا يُلغيان قيمةً محفوظة مسبقاً."""
    patient_id: str
    name: str
    is_new: bool
    sex: str | None = None          # "male" | "female" | None (تقدير بصري)
    age: int | None = None          # تقدير العمر أو None


# ─────────────────────────────────────────────────────────────────────────────
# بناء طبقتي الصوت والرؤية حسب الوضع (استيراد كسول للاعتماديات الثقيلة)
# ─────────────────────────────────────────────────────────────────────────────

def build_voice(mode: str):
    """يُرجع (tts, stt). 'voice' = صوت حقيقي، 'text' = بدائل نصّية للتصحيح.
    مزوّد النطق: OpenAI TTS افتراضياً (صوت واحد متعدد اللغات)؛ `set TTS_PROVIDER=edge`
    للعودة إلى edge-tts. لو تعذّر تجهيز OpenAI TTS (مثلاً بلا مفتاح) نسقط إلى Edge."""
    if mode == "text":
        from robot.voice import PrintTTS, TypedSTT
        return PrintTTS(), TypedSTT()
    from robot.voice import EdgeTTS, OpenAITTS, WhisperSTT
    print("تحميل نموذج Whisper (قد يأخذ لحظات أول مرّة) ...")
    stt = WhisperSTT()
    if os.environ.get("TTS_PROVIDER", "openai").lower() == "openai":
        try:
            return OpenAITTS(), stt
        except Exception as exc:  # noqa: BLE001 - بلا مفتاح/شبكة: أكمل بصوت Edge بدل الانهيار
            print(f"تعذّر تجهيز OpenAI TTS ({exc}) — سنستخدم EdgeTTS.")
    return EdgeTTS(), stt


def autonomous_enroll(face, grab_frame, tts, name: str, patient_id: str,
                      shots: int = 5, max_tries: int = 18) -> int:
    """يسجّل الروبوت وجه مريض جديد *ذاتياً* أثناء التشغيل (بلا أدوات سطر أوامر).

    يلتقط عدّة لقطات موجّهة صوتياً (أمام/يمين/يسار) فيخزّن عدّة embeddings للشخص نفسه —
    وهذا يرفع متانة التعرّف لاحقاً كثيراً مقابل لقطة واحدة. يُرجع عدد اللقطات المحفوظة
    (0 = تعذّرت رؤية الوجه). كل لقطة تختار الوجه الأكبر/الأقرب، فلا تُسجَّل وجوه المارّة."""
    import time
    guidance = [
        "انظر إلى الكاميرا مباشرةً من فضلك.",
        "أدِر رأسك قليلاً إلى اليمين.",
        "والآن قليلاً إلى اليسار.",
        "أحسنت، انظر أمامك.",
        "لقطة أخيرة… ثبات من فضلك.",
    ]
    captured, tries = 0, 0
    while captured < shots and tries < max_tries:
        if captured < len(guidance):
            tts.speak(guidance[captured], "ar")
        img = grab_frame()
        tries += 1
        if img is None:
            continue
        try:
            face.enroll(img, name=name, patient_id=patient_id)
            captured += 1
            time.sleep(0.5)      # فارق زمني ليختلف الإطار التالي (زاوية/تعبير)
        except ValueError:
            time.sleep(0.3)      # لا وجه واضح — أعِد المحاولة
    return captured


def identify_patient(use_face: bool, camera_manager, tts, stt):
    """يُرجع (patient_id, name, is_new). الوجه إن توفّر، وإلا اسم عبر الصوت/الكتابة.

    مع الكاميرا: يتعرّف على *كل* من في الإطار، فإن كان أكثر من شخص يعلن من يراهم ويطلب
    ممّن يريد التشخيص أن يقترب، ثم يختار الأقرب (الوجه الأكبر) بوصفه المريض الحالي.
    المريض الجديد يُسجّله الروبوت ذاتياً بعدّة لقطات (autonomous_enroll)."""
    if use_face:
        from robot.vision import FaceRegistry, make_default_recognizer
        import time
        # أقوى محرّك متاح (InsightFace إن ثُبِّت، وإلا DeepFace) + عتبته الموصى بها.
        face = FaceRegistry(make_default_recognizer())

        def grab_frame():
            """أعِد أحدث إطار صالح من الكاميرا (بانتظار قصير إن لزم)، أو None."""
            for _ in range(20):
                img = camera_manager.read()
                if img is not None:
                    return img
                time.sleep(0.1)
            return None

        tts.speak("مرحباً، انظر إلى الكاميرا من فضلك.", "ar")
        image = grab_frame()
        if image is None:
            tts.speak("عذراً، لم أتمكن من التقاط صورة. سأكمل بدون حفظ الوجه.", "ar")
            return Identification(uuid.uuid4().hex, "ضيف", True)

        # 1) من في الإطار؟ (كل الوجوه، مع رفض المجهول)
        matches = face.identify_all(image)
        if not matches:
            tts.speak("لم أرَ أي وجه بوضوح. سأكمل بدون حفظ الوجه.", "ar")
            return Identification(uuid.uuid4().hex, "ضيف", True)

        # 2) عدّة أشخاص: اعلن من نعرفهم واطلب من صاحب التشخيص الاقتراب، ثم أعِد الالتقاط
        if len(matches) == 1:
            primary = matches[0]              # شخص واحد ⇒ هو المريض الأقرب مباشرةً
        else:
            known_names = [m.name for m in matches if m.known and m.name]
            greet = f"أرى {len(matches)} أشخاص أمامي"
            if known_names:
                greet += "، منهم " + "، ".join(known_names)
            tts.speak(greet + ". من يريد التشخيص؟ ليقترب من الكاميرا من فضله.", "ar")
            time.sleep(1.5)
            image = grab_frame() or image     # الأقرب سيصبح الوجه الأكبر في اللقطة الجديدة
            primary = face.identify_primary(image)

        # الجنس/العمر المُقدَّران من الوجه — يُملآن الفراغ فلا يُسأل عن الجنس
        pred_sex = primary.sex if primary else None
        pred_age = primary.age if primary else None

        # 3) المريض الحالي = الوجه الأبرز/الأقرب في الإطار
        if primary is not None and primary.known:
            return Identification(primary.patient_id, primary.name or "", False,
                                  pred_sex, pred_age)

        # مريض جديد: خذ اسمه ثم سجّل الروبوت وجهه ذاتياً بعدّة لقطات (متانة أعلى)
        tts.speak("لا أعرفك بعد. ما اسمك من فضلك؟", "ar")
        name = parse_name(stt.listen()) or "ضيف"
        pid = uuid.uuid4().hex
        captured = autonomous_enroll(face, grab_frame, tts, name, pid)
        if captured == 0:
            tts.speak("لم أتمكّن من رؤية وجهك بوضوح، سأكمل بدون حفظ الصورة.", "ar")
        else:
            tts.speak(f"سعدتُ بلقائك يا {name}. حفظتُ وجهك وسأتذكّرك في زيارتك القادمة.", "ar")
        return Identification(pid, name, True, pred_sex, pred_age)

    # بلا كاميرا: ذاكرة بالاسم عبر IdentityRegistry (مفيد للتصحيح)
    reg = IdentityRegistry()
    tts.speak("ما اسمك من فضلك؟", "ar")
    name = parse_name(stt.listen()) or "ضيف"
    existing = reg.lookup(name)
    if existing:
        return Identification(existing, name, False)
    return Identification(reg.register(name), name, True)


# ─────────────────────────────────────────────────────────────────────────────
# الجسر مع قاعدة البيانات (كتابة منظَّمة ومُتحقَّقة)
# ─────────────────────────────────────────────────────────────────────────────

def preload_picture(record: PatientRecord) -> ClinicalPicture:
    """جهّز ClinicalPicture من سجلّ معروف، فلا يُعيد الـLLM سؤال ما نعرفه."""
    pic = ClinicalPicture()
    pic.age = record.age
    if record.sex in (Sex.MALE, Sex.FEMALE):
        pic.sex = record.sex.value
    pic.medications = [m.name for m in record.medications if m.name]
    pic.allergies = [a.substance for a in record.allergies if a.substance]
    pic.chronic_conditions = list(record.chronic_conditions)
    return pic


def write_back(record: PatientRecord, picture: ClinicalPicture, result) -> None:
    """ادمج ما استخرجه الـLLM في سجلّ المريض (حقول مُتحقَّقة، لا SQL حرّ)."""
    if picture.age is not None:
        record.age = picture.age
    if picture.sex in ("male", "female"):
        record.sex = Sex(picture.sex)

    chronic = list(record.chronic_conditions)
    for c in picture.chronic_conditions:
        if c and c not in chronic:
            chronic.append(c)
    record.chronic_conditions = chronic

    meds = list(record.medications)
    have = {m.name.lower() for m in meds}
    for m in picture.medications:
        if m and m.lower() not in have:
            meds.append(Medication(name=m))
    record.medications = meds

    algs = list(record.allergies)
    havea = {a.substance.lower() for a in algs}
    for a in picture.allergies:
        if a and a.lower() not in havea:
            algs.append(Allergy(substance=a))
    record.allergies = algs

    if result is not None and result.working_diagnosis is not None:
        cond = result.working_diagnosis.condition
        past = list(record.past_diagnoses)
        if cond and cond not in past:
            past.append(cond)
            record.past_diagnoses = past
        record.visit_history = [
            *record.visit_history,
            PastVisitSummary(visit_id=uuid.uuid4().hex, date=_now(),
                             chief_complaint=picture.chief_complaint or "",
                             working_diagnosis=cond or None),
        ]
    record.updated_at = _now()


# ─────────────────────────────────────────────────────────────────────────────
# الحلقة الرئيسية
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    from robot.camera_config import CAMERA_SOURCE
    from robot.vision.camera_manager import CameraManager

    ap = argparse.ArgumentParser(description="تشغيل روبوت المساعد الطبي")
    ap.add_argument("--mode", choices=("voice", "text"), default="voice")
    ap.add_argument("--no-face", action="store_true", help="بلا كاميرا (للتصحيح)")
    ap.add_argument("--camera", default=CAMERA_SOURCE, help="فهرس أو رابط كاميرا الرؤية")
    ap.add_argument("--serial", help="منفذ USB Serial للـESP32 (مثال COM5) بدل WiFi")
    args = ap.parse_args()

    tts, stt = build_voice(args.mode)

    # المزوّد والوكيل
    concierge_gw = make_gateway(models={ModelTier.FAST: CONCIERGE_FAST,
                                         ModelTier.REASONING: CONCIERGE_REASONING})
    cds_gw = make_gateway(models={ModelTier.FAST: CDSS_FAST,
                                  ModelTier.REASONING: CDSS_REASONING})
    facade = build_reasoning_facade(cds_gw)
    agent = ConciergeAgent(
        facade=facade,
        gateway=concierge_gw,
        safety_check=ConciergeSafety(SafetyGuard(cds_gw)),
    )
    knowledge_agent = build_knowledge_agent()

    # قاعدة البيانات
    db = Database(os.environ.get("DB_PATH", "medical_ai.db"))
    await db.connect()
    store = ConsultationStore(db)

    camera_manager = None
    if not args.no_face:
        camera_manager = CameraManager(args.camera)
        camera_manager.start()

    try:
        # 1) الهوية عبر الوجه/الاسم (+ تقدير الجنس/العمر من الوجه)
        ident = await asyncio.to_thread(
            identify_patient, not args.no_face, camera_manager, tts, stt)
        pid, name, is_new = ident.patient_id, ident.name, ident.is_new

        record = await store.patients.get(pid)
        need_save = record is None
        if record is None:
            record = PatientRecord(patient_id=pid)

        # الجنس المُقدَّر من الوجه يملأ الفراغ فقط (لا يُلغي قيمةً محفوظة) — فلا يُسأل عنه.
        if record.sex not in (Sex.MALE, Sex.FEMALE) and ident.sex in ("male", "female"):
            record.sex = Sex(ident.sex)
            need_save = True
        if need_save:
            await store.patients.save(record)

        # حمّل الوكيل بما نعرفه عن المريض (هوية + سجلّ سابق + لغته المحفوظة) — عبر الواجهة
        # العامّة. المريض العائد يُستأنف بلغته المقفولة من أول كلمة؛ الجديد يُسأل عن لغته
        # (ثنائياً) وتُقفل إجابته لهذه الزيارة وكل ما بعدها.
        saved_lang = record.preferred_language or None
        agent.preload(name=name, patient_id=pid, picture=preload_picture(record),
                      language=saved_lang)

        if saved_lang:
            past = record.past_diagnoses[-1] if record.past_diagnoses else ""
            from app.concierge import prompts as concierge_prompts
            opening = concierge_prompts.opening(
                name, saved_lang, returning=not is_new, last_topic=past)
        else:
            from app.concierge import prompts as concierge_prompts
            opening = concierge_prompts.ask_language(name)
        await asyncio.to_thread(tts.speak, opening, saved_lang or "ar")

        # تشغيل تتبع الرأس في الخلفية إذا كانت الكاميرا مفعلة
        stop_tracking_event = None
        tracking_thread = None
        if not args.no_face:
            import threading
            from robot.head_tracking import run_tracker
            stop_tracking_event = threading.Event()
            tracking_thread = threading.Thread(
                target=run_tracker,
                args=(
                    camera_manager,
                    os.environ.get("ESP32_IP", "192.168.1.50"),
                    args.serial,
                    True,
                    stop_tracking_event,
                ),
                daemon=True
            )
            tracking_thread.start()

        try:
            # 2) حلقة المقابلة الصوتية
            for _ in range(40):                              # سقف أمان للأدوار
                user = (await asyncio.to_thread(stt.listen)).strip()
                if not user:
                    continue
                if user.lower() in EXIT_WORDS:
                    break
                print(f"[المريض]: {user}")
                reply = await agent.respond(user)
                print(f"[الروبوت]: {reply}")
                await asyncio.to_thread(tts.speak, reply, detect_language(reply))
                if agent._complete:
                    break

            # 3) احفظ الزيارة كاملةً عبر وصلة الحفظ الموحّدة (محادثة + تشخيص + تقرير PENDING +
            #    أدلّة + وصفات PENDING) — تظهر للطبيب من الروبوت ومن الموقع بنفس الشكل.
            visit_id = await persist_concierge_visit(agent.snapshot(), store=store, facade=facade)
            if knowledge_agent is not None:
                knowledge_agent.submit_visit(store, visit_id)
            print(f"\n💾 حُفِظت الزيارة (patient_id={pid}, visit_id={visit_id}).")
        finally:
            if stop_tracking_event:
                stop_tracking_event.set()
            if tracking_thread:
                tracking_thread.join(timeout=2.0)
    finally:
        if knowledge_agent is not None:
            await knowledge_agent.close()
        await db.close()
        if camera_manager is not None:
            camera_manager.stop()


if __name__ == "__main__":
    asyncio.run(main())

