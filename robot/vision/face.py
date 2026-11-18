"""
robot/vision/face.py
--------------------
التعرّف على وجوه المرضى (المرحلة 1: الرؤية في خارطة طريق الفصل 6).

يحلّ هذا محلّ/يُكمّل سجلّ الأسماء (app/concierge/identity.py:IdentityRegistry): بدل أن يكتب
المريض اسمه، تتعرّف الكاميرا على وجهه فتُرجع `patient_id` يدخل نفس مسار استعادة التاريخ
(`ConsultationService.start(patient_id=...)`).

التصميم يعكس فلسفة المشروع:
  - `FaceRecognizer` تجريد لمحرّك الـembeddings (InsightFace افتراضياً، DeepFace احتياطياً،
    غيرهما لاحقاً) — مثل LLMGateway. المحرّك قابل للتبديل دون لمس منطق المطابقة.
  - `FaceRegistry` يوازي IdentityRegistry: `identify(image)` بدل lookup، `enroll(image)` بدل
    register، وملف JSON للتخزين. أضيفت `identify_all(image)` للتعرّف على *كل* من في الإطار
    (يسأل الروبوت بعدها: من يريد التشخيص؟).
  - منطق المطابقة (cosine + عتبة + رفض المجهول) نقيّ ولا يعتمد على المحرّك ولا الكاميرا،
    فيُختبر offline بمحرّك وهمي (مثل FakeGateway). الصورة مجرّد مصفوفة numpy (BGR).

لماذا محرّكان؟ التجربة على مشروع سابق أثبتت أن InsightFace (RetinaFace للكشف + ArcFace R100
للتمثيل) يفصل الأشخاص أقوى بكثير من كاشف DeepFace الافتراضي (opencv/Haar) الذي يُنتج embeddings
مشوّشة تجعل أشخاصاً مختلفين يقتربون تحت العتبة (فيُنادى الجميع بالاسم نفسه). لذلك InsightFace هو
الافتراضي وعتبته أضيق (0.5 مقابل 0.68 لِDeepFace).

التثبيت للتشغيل الحقيقي:
    pip install insightface onnxruntime opencv-python     # المحرّك الافتراضي (موصى به)
    pip install deepface tf-keras opencv-python           # المحرّك الاحتياطي
"""
from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DEFAULT_PATH = "data/face_registry.json"


# ─────────────────────────────────────────────────────────────────────────────
# أنواع البيانات (وجه مكتشف + نتيجة تعرّف)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectedFace:
    """وجه واحد اكتُشف في إطار: متجه الوصف + الإطار المحيط + ثقة الكشف + تقدير جنس/عمر.

    الجنس/العمر تقديران بصريان من النموذج (InsightFace genderage) — تقريبيان، لا حقيقة قاطعة؛
    يُستعملان كنقطة بداية يصحّحها كلام المريض إن لزم."""
    embedding: np.ndarray
    bbox: tuple[int, int, int, int] | None = None   # (x1, y1, x2, y2)
    det_score: float = 1.0
    sex: str | None = None                           # "male" | "female" | None
    age: int | None = None                           # تقدير العمر (سنوات) أو None

    @property
    def area(self) -> float:
        """مساحة الوجه بالبكسل (0 إن لا bbox) — الأكبر ≈ الأقرب للكاميرا."""
        if self.bbox is None:
            return 0.0
        x1, y1, x2, y2 = self.bbox
        return float(max(0, x2 - x1) * max(0, y2 - y1))


@dataclass
class FaceMatch:
    """نتيجة التعرّف على وجه واحد ضمن `identify_all`/`identify_primary`."""
    bbox: tuple[int, int, int, int] | None
    patient_id: str | None          # None = مجهول (لا تطابق تحت العتبة)
    name: str                       # اسم المطابق أو "" إن مجهول
    distance: float | None          # مسافة أقرب تطابق (أقل = أوثق)، أو None إن مجهول
    sex: str | None = None          # تقدير الجنس من الوجه ("male"/"female"/None)
    age: int | None = None          # تقدير العمر من الوجه (سنوات) أو None

    @property
    def known(self) -> bool:
        return self.patient_id is not None


# ─────────────────────────────────────────────────────────────────────────────
# تجريد محرّك التعرّف (قابل للتبديل)
# ─────────────────────────────────────────────────────────────────────────────

class FaceRecognizer(ABC):
    """يحوّل صورة (مصفوفة BGR) إلى متجهات وصف للوجوه الموجودة فيها."""

    # العتبة المُوصى بها لهذا المحرّك (مسافة cosine): تُستعمل إن لم يمرّر المستخدم عتبة صريحة.
    recommended_threshold: float = 0.5
    # معرّف فضاء المتجهات: embeddings محرّكين مختلفين غير قابلة للمقارنة (نفس البُعد، فضاء مختلف)
    # فيُستعمل هذا لتجاهل المتجهات المسجّلة بمحرّك آخر بدل إنتاج مسافات بلا معنى.
    engine_id: str = ""

    @abstractmethod
    def embed(self, image: np.ndarray) -> list[np.ndarray]:
        """أرجع قائمة embeddings (متجه لكل وجه مكتشف). فارغة إن لا وجه."""

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """أرجع الوجوه المكتشفة مع بياناتها (bbox/ثقة). الافتراضي يلفّ `embed` بلا bbox؛
        المحرّكات التي تعرف مواقع الوجوه (InsightFace/DeepFace) تتجاوز هذه الدالة."""
        return [DetectedFace(embedding=np.asarray(v, dtype=np.float32))
                for v in self.embed(image)]

    def embed_primary(self, image: np.ndarray) -> np.ndarray | None:
        """متجه الوجه الأبرز فقط (الأكبر ≈ الأقرب للكاميرا)، أو None إن لا وجه.
        ملائم للتسجيل/التعرّف الفردي (الشخص الذي اقترب من الكاميرا)."""
        faces = self.detect(image)
        if not faces:
            return None
        return max(faces, key=lambda f: f.area).embedding


class InsightFaceRecognizer(FaceRecognizer):
    """محرّك InsightFace (buffalo_l = RetinaFace + ArcFace R100). أقوى فصلاً؛ عتبة 0.5.

    يُستورد بكسل فلا يُثقل بقية النظام. أول تشغيل ينزّل حزمة النموذج مرّة واحدة."""

    recommended_threshold: float = 0.5
    engine_id: str = "insightface:buffalo_l"

    def __init__(self, model_pack: str = "buffalo_l",
                 det_size: tuple[int, int] = (640, 640), ctx_id: int = -1) -> None:
        # ctx_id = -1 يفرض CPU (يتجنّب انتظار CUDA الطويل إن غابت مكتباته) — مناسب للـPi.
        self._pack = model_pack
        self._det_size = det_size
        self._ctx = ctx_id
        self._app = None

    def _get_app(self):
        if self._app is None:
            from insightface.app import FaceAnalysis  # استيراد كسول
            # 'genderage' يُقدّر الجنس والعمر من الوجه، فلا يُسأل المريض عن جنسه.
            app = FaceAnalysis(name=self._pack,
                               allowed_modules=["detection", "recognition", "genderage"])
            app.prepare(ctx_id=self._ctx, det_size=self._det_size)
            self._app = app
        return self._app

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        app = self._get_app()
        faces = app.get(image)          # كشف + تمثيل + جنس/عمر لكل الوجوه دفعةً واحدة
        out: list[DetectedFace] = []
        for f in faces:
            emb = getattr(f, "embedding", None)
            if emb is None:
                emb = getattr(f, "normed_embedding", None)
            if emb is None:
                continue
            emb = np.asarray(emb, dtype=np.float32)
            n = np.linalg.norm(emb)
            if n > 0:                    # نطبّع لوحدة الطول لتسهيل مقارنة الـcosine
                emb = emb / n
            bbox = None
            if getattr(f, "bbox", None) is not None:
                x1, y1, x2, y2 = (int(v) for v in f.bbox[:4])
                bbox = (x1, y1, x2, y2)
            # الجنس: gender=1 ذكر، 0 أنثى (InsightFace). قد يغيب إن لم يُحمّل genderage.
            sex = None
            g = getattr(f, "gender", None)
            if g is not None:
                sex = "male" if int(g) == 1 else "female"
            age = getattr(f, "age", None)
            age = int(age) if age is not None else None
            out.append(DetectedFace(
                embedding=emb, bbox=bbox,
                det_score=float(getattr(f, "det_score", 1.0) or 1.0),
                sex=sex, age=age))
        return out

    def embed(self, image: np.ndarray) -> list[np.ndarray]:
        return [f.embedding for f in self.detect(image)]


class DeepFaceRecognizer(FaceRecognizer):
    """محرّك DeepFace (ArcFace افتراضياً) — احتياطي. DeepFace يُستورد بكسل.

    ملاحظة: الكاشف الافتراضي 'opencv' ضعيف؛ يُفضّل 'retinaface' أو 'mtcnn' إن توفّرا،
    وعتبته أوسع (0.68). يبقى للتوافق مع البيئات التي لا يتوفّر فيها InsightFace."""

    recommended_threshold: float = 0.68
    engine_id: str = "deepface:ArcFace"

    def __init__(self, model_name: str = "ArcFace",
                 detector_backend: str = "opencv") -> None:
        self._model = model_name
        self._detector = detector_backend

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        from deepface import DeepFace  # استيراد كسول
        try:
            reps = DeepFace.represent(
                img_path=image,                 # DeepFace يقبل مصفوفة numpy
                model_name=self._model,
                detector_backend=self._detector,
                enforce_detection=True,         # ارفع خطأً إن لا وجه
                align=True,
            )
        except ValueError:
            return []                           # لا وجه مكتشف
        out: list[DetectedFace] = []
        for r in reps:
            emb = np.asarray(r["embedding"], dtype=np.float32)
            fa = r.get("facial_area") or {}
            bbox = None
            if all(k in fa for k in ("x", "y", "w", "h")):
                x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
                bbox = (int(x), int(y), int(x + w), int(y + h))
            score = float(r.get("face_confidence", 1.0) or 1.0)
            out.append(DetectedFace(embedding=emb, bbox=bbox, det_score=score))
        return out

    def embed(self, image: np.ndarray) -> list[np.ndarray]:
        return [f.embedding for f in self.detect(image)]


def make_default_recognizer(prefer_insightface: bool = True) -> FaceRecognizer:
    """اختر أقوى محرّك متاح: InsightFace إن ثُبِّت، وإلا DeepFace. (سياسة المشروع.)"""
    if prefer_insightface:
        try:
            import insightface  # noqa: F401 — فحص توفّر فقط
            return InsightFaceRecognizer()
        except Exception as exc:  # noqa: BLE001
            print(f"[vision] InsightFace غير متاح ({exc}) — سأستخدم DeepFace الاحتياطي.")
    return DeepFaceRecognizer()


# ─────────────────────────────────────────────────────────────────────────────
# منطق المطابقة (نقيّ — قابل للاختبار offline)
# ─────────────────────────────────────────────────────────────────────────────

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """0 = متطابقان تماماً، 1 = متعامدان، 2 = متعاكسان."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


# ─────────────────────────────────────────────────────────────────────────────
# سجلّ الوجوه (يوازي IdentityRegistry)
# ─────────────────────────────────────────────────────────────────────────────

class FaceRegistry:
    """سجلّ patient_id -> (اسم + عدّة embeddings)، مدعوم بملف JSON.

    العتبة الافتراضية تُؤخذ من المحرّك (`recommended_threshold`): 0.5 لِInsightFace،
    0.68 لِDeepFace. المسافة cosine (أقل = أكثر تشابهاً). فوق العتبة ⇒ مجهول (رفض صريح)."""

    def __init__(self, recognizer: FaceRecognizer, path: str = _DEFAULT_PATH,
                 threshold: float | None = None) -> None:
        self._rec = recognizer
        self._path = Path(path)
        # إن لم تُمرَّر عتبة صريحة، استعمل عتبة المحرّك الموصى بها.
        self._threshold = (threshold if threshold is not None
                           else getattr(recognizer, "recommended_threshold", 0.5))
        # patient_id -> {"name": str, "embeddings": list[list[float]]}
        self._people: dict[str, dict] = {}
        if self._path.exists():
            try:
                self._people = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - سجلّ تالف يبدأ فارغاً، لا ينهار
                self._people = {}

    # ── مطابقة متجه واحد ضدّ السجلّ (نقيّة) ────────────────────────────────────
    def _match_vector(self, vec: np.ndarray) -> tuple[str | None, float | None]:
        """أرجع (patient_id, distance) لأقرب وجه مسجّل دون العتبة، أو (None, None).
        تُتجاهَل السجلّات المُسجَّلة بمحرّك مختلف (فضاء متجهات غير قابل للمقارنة)."""
        cur_engine = getattr(self._rec, "engine_id", "")
        best_id, best_dist = None, self._threshold
        for pid, rec in self._people.items():
            # سجلّ قديم بلا وسم محرّك ⇒ نفترض DeepFace (بيئة السجلّ التاريخية).
            if rec.get("engine", "deepface:ArcFace") != cur_engine:
                continue
            for stored in rec.get("embeddings", []):
                d = cosine_distance(vec, np.asarray(stored, dtype=np.float32))
                if d < best_dist:
                    best_id, best_dist = pid, d
        return (best_id, best_dist) if best_id is not None else (None, None)

    def _to_match(self, f: DetectedFace) -> FaceMatch:
        """حوّل وجهاً مكتشفاً إلى نتيجة تعرّف (مع الجنس/العمر المُقدَّرين)."""
        pid, dist = self._match_vector(f.embedding)
        return FaceMatch(bbox=f.bbox, patient_id=pid,
                         name=self.name_of(pid) if pid else "",
                         distance=dist, sex=f.sex, age=f.age)

    # ── التعرّف على الوجه الأبرز/الأقرب (مثل lookup) ───────────────────────────
    def identify_primary(self, image: np.ndarray) -> FaceMatch | None:
        """أرجع نتيجة التعرّف للوجه الأكبر/الأقرب (هوية + جنس/عمر مُقدَّران)، أو None
        إن لا وجه. patient_id=None يعني مجهولاً (لم يُطابِق تحت العتبة)."""
        faces = self._rec.detect(image)
        if not faces:
            return None
        return self._to_match(max(faces, key=lambda f: f.area))

    def identify(self, image: np.ndarray) -> tuple[str, float] | None:
        """توافقيّة: أرجع (patient_id, distance) للوجه الأبرز إن طابَق، وإلا None."""
        m = self.identify_primary(image)
        if m is None or m.patient_id is None:
            return None
        return (m.patient_id, m.distance)

    # ── التعرّف على كل من في الإطار (مسار «من يريد التشخيص؟») ──────────────────
    def identify_all(self, image: np.ndarray) -> list[FaceMatch]:
        """تعرّف على *كل* وجه في الإطار. النتائج مرتّبة يساراً→يميناً (حسب bbox)،
        فيسهل على الروبوت مناداة المعروفين والسؤال عمّن يريد التشخيص. المجهول
        (لا تطابق تحت العتبة) يبقى مجهولاً (patient_id=None) بدل إلصاقه بأقرب هوية."""
        matches = [self._to_match(f) for f in self._rec.detect(image)]
        # يساراً→يميناً (الوجوه بلا bbox في النهاية)
        matches.sort(key=lambda m: (m.bbox[0] if m.bbox else float("inf")))
        return matches

    # ── التسجيل (مثل register) ────────────────────────────────────────────────
    def enroll(self, image: np.ndarray, name: str | None = None,
               patient_id: str | None = None) -> str:
        """سجّل وجهاً (الأبرز/الأقرب في الإطار). إن طابق شخصاً موجوداً يضيف المتجه إليه
        (تحسين متانة)؛ وإلا ينشئ patient_id جديداً. يُرجع الـpatient_id."""
        vec = self._rec.embed_primary(image)
        if vec is None:
            raise ValueError("لا وجه مكتشف في الصورة")

        if patient_id is None:
            pid, _ = self._match_vector(vec)
            patient_id = pid if pid is not None else uuid.uuid4().hex

        rec = self._people.setdefault(
            patient_id,
            {"name": name or "", "embeddings": [],
             "engine": getattr(self._rec, "engine_id", "")})
        if name and not rec.get("name"):
            rec["name"] = name
        rec["embeddings"].append([round(float(x), 6) for x in vec.tolist()])
        self._save()
        return patient_id

    def name_of(self, patient_id: str | None) -> str:
        if not patient_id:
            return ""
        return self._people.get(patient_id, {}).get("name", "")

    def __len__(self) -> int:
        return len(self._people)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._people, ensure_ascii=False, indent=1),
                              encoding="utf-8")
