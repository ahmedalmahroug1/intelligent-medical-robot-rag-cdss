"""
tests/test_router.py
--------------------
Standalone tests for File 5 (router + path contract). Run from the project root:

    python -m tests.test_router

No pytest required. The dispatch tests build a ConsultationState, so they need
pydantic (your env has it); the two pure tests at the end run without it.

What these tests PROVE (not just exercise):
    1. The router routes among PEERS by classification, with no privileged
       diagnosis fallback. This is the concrete enforcement of the "diagnosis
       must not be the de facto root" rule — a passing test, not an intention.
    2. An already-active path continues WITHOUT re-classifying.
    3. A full Diagnosis -> Test-Review -> Diagnosis suspend/resume round trip
       works, with resume_route written by the suspending path, read by the
       interim path, and cleared by the router on switch-back.
    4. An unclassifiable turn produces a clarifying question and dispatches to
       no path.
    5. Transition invariants and router registration guards hold.

The stub paths below are TEST DOUBLES (like FakeGateway in tests/fakes.py). They
are named after real routes only so the round-trip test reads like the real
Diagnosis/Test-Review scenario; each is registered as an ordinary peer.
"""

from __future__ import annotations

import asyncio
import inspect
import traceback
from collections.abc import Iterable

from app.core.enums import Route
from app.core.state import ConsultationState, PatientRecord, SessionState, VisitState
from app.routing import (
    Path,
    PathResult,
    Router,
    Transition,
    TransitionKind,
    TurnClassifier,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test doubles
# ─────────────────────────────────────────────────────────────────────────────

class ScriptedClassifier(TurnClassifier):
    """
    Deterministic classifier for offline tests. Returns the first route whose
    keyword appears in the turn (and is actually available), else `default`.
    Records how many times it was consulted so tests can assert that an active
    path did NOT trigger classification.
    """

    def __init__(
        self,
        *,
        default: Route = Route.UNKNOWN,
        mapping: dict[str, Route] | None = None,
    ) -> None:
        self.default = default
        self.mapping = mapping or {}
        self.calls = 0

    async def classify(
        self, *, turn: str, state: ConsultationState, available: Iterable[Route]
    ) -> Route:
        self.calls += 1
        options = set(available)
        for keyword, route in self.mapping.items():
            if keyword in turn and route in options:
                return route
        return self.default


class DiagnosisStub(Path):
    route = Route.DIAGNOSIS

    def __init__(self) -> None:
        self.turns: list[str] = []

    async def handle(self, *, turn: str, state: ConsultationState) -> PathResult:
        self.turns.append(turn)
        if "[suspend]" in turn:
            # Write our OWN resume intent before asking the router to suspend us.
            state.visit.resume_route = Route.DIAGNOSIS
            return PathResult(
                "Before we go on, let's review your test result.",
                Transition.suspend(Route.TEST_REVIEW),
            )
        return PathResult(f"diagnosis handled: {turn}", Transition.stay())


class TestReviewStub(Path):
    route = Route.TEST_REVIEW

    def __init__(self) -> None:
        self.turns: list[str] = []

    async def handle(self, *, turn: str, state: ConsultationState) -> PathResult:
        self.turns.append(turn)
        # The interim path does NOT know who suspended it; it reads the blackboard.
        resume = state.visit.resume_route or Route.UNKNOWN
        return PathResult(
            "Thanks, I've reviewed that result. Back to your consultation.",
            Transition.switch(resume),
        )


class QaStub(Path):
    route = Route.QA

    def __init__(self) -> None:
        self.turns: list[str] = []

    async def handle(self, *, turn: str, state: ConsultationState) -> PathResult:
        self.turns.append(turn)
        return PathResult(f"qa answered: {turn}", Transition.stay())


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fresh_state(
    *, active: Route = Route.UNKNOWN, resume: Route | None = None
) -> ConsultationState:
    patient = PatientRecord(patient_id="p1")
    visit = VisitState(visit_id="v1", patient_id="p1", resume_route=resume)
    session = SessionState(
        session_id="s1", patient_id="p1", visit_id="v1", active_route=active
    )
    return ConsultationState(patient=patient, visit=visit, session=session)


# ─────────────────────────────────────────────────────────────────────────────
# Tests (dispatch — need pydantic)
# ─────────────────────────────────────────────────────────────────────────────

async def test_routes_among_peers_no_diagnosis_root() -> None:
    """A turn routes to whichever peer the classifier picks — diagnosis is not
    a privileged fallback."""
    clf = ScriptedClassifier(mapping={"[qa]": Route.QA, "[dx]": Route.DIAGNOSIS})
    qa, dx = QaStub(), DiagnosisStub()
    router = Router(clf)
    router.register(qa)
    router.register(dx)

    # Classified to QA -> QA handles, diagnosis does NOT.
    st = _fresh_state()
    res = await router.route_turn(turn="[qa] what is a normal heart rate?", state=st)
    assert "qa answered" in res.reply, res.reply
    assert qa.turns == ["[qa] what is a normal heart rate?"]
    assert dx.turns == [], "diagnosis must not handle a QA turn"
    assert st.session.active_route is Route.QA

    # Fresh state classified to diagnosis -> diagnosis handles.
    st2 = _fresh_state()
    res2 = await router.route_turn(turn="[dx] my chest has been hurting", state=st2)
    assert "diagnosis handled" in res2.reply, res2.reply
    assert dx.turns == ["[dx] my chest has been hurting"]
    assert len(qa.turns) == 1, "QA must not have been touched by the diagnosis turn"
    assert st2.session.active_route is Route.DIAGNOSIS


async def test_active_path_continues_without_reclassifying() -> None:
    """Once a path is active, the router continues it and never consults the
    classifier."""
    clf = ScriptedClassifier(default=Route.QA)  # would misroute to QA if consulted
    dx = DiagnosisStub()
    router = Router(clf)
    router.register(dx)
    router.register(QaStub())

    st = _fresh_state(active=Route.DIAGNOSIS)
    res = await router.route_turn(turn="it still hurts today", state=st)

    assert "diagnosis handled" in res.reply, res.reply
    assert clf.calls == 0, "an active path must not trigger classification"
    assert st.session.active_route is Route.DIAGNOSIS


async def test_suspend_resume_round_trip() -> None:
    """Diagnosis -> Test-Review -> Diagnosis, driven entirely through the
    blackboard and transitions."""
    clf = ScriptedClassifier(mapping={"[dx]": Route.DIAGNOSIS})
    dx, tr = DiagnosisStub(), TestReviewStub()
    router = Router(clf)
    router.register(dx)
    router.register(tr)
    st = _fresh_state()

    # Turn 1: classify -> diagnosis, which suspends to test-review.
    await router.route_turn(turn="[dx] [suspend] should I get an X-ray?", state=st)
    assert st.session.active_route is Route.TEST_REVIEW, "interim path is now active"
    assert st.visit.resume_route is Route.DIAGNOSIS, "suspending path recorded resume"
    assert len(dx.turns) == 1 and tr.turns == []

    # Turn 2: active path is test-review; it resumes diagnosis via the blackboard.
    await router.route_turn(turn="the X-ray came back normal", state=st)
    assert tr.turns == ["the X-ray came back normal"], "test-review handled turn 2"
    assert st.session.active_route is Route.DIAGNOSIS, "resumed to the original path"
    assert st.visit.resume_route is None, "resume target cleared on switch-back"
    assert clf.calls == 1, "only the first turn needed classification"

    # Turn 3: back on diagnosis as an ordinary active peer.
    res3 = await router.route_turn(turn="ok, what's next?", state=st)
    assert "diagnosis handled" in res3.reply, res3.reply
    assert len(dx.turns) == 2


async def test_unknown_classification_asks_to_clarify() -> None:
    """An unclassifiable turn yields a clarifying reply and dispatches to no
    path; the active route stays unrouted."""
    clf = ScriptedClassifier(default=Route.UNKNOWN)
    dx = DiagnosisStub()
    router = Router(clf)
    router.register(dx)

    st = _fresh_state()
    res = await router.route_turn(turn="hmm", state=st)

    assert "right thing" in res.reply, "expected the clarifying reply"
    assert dx.turns == [], "no path should handle an unclassified turn"
    assert st.session.active_route is Route.UNKNOWN, "active route stays unrouted"


async def test_unregistered_route_from_classifier_clarifies() -> None:
    """If the classifier names a route with no registered path, the router
    clarifies rather than crashing."""
    clf = ScriptedClassifier(default=Route.TREATMENT)  # no treatment path registered
    router = Router(clf)
    router.register(QaStub())

    st = _fresh_state()
    res = await router.route_turn(turn="anything", state=st)
    assert "right thing" in res.reply
    assert st.session.active_route is Route.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# Tests (pure — run without pydantic)
# ─────────────────────────────────────────────────────────────────────────────

def test_transition_invariants() -> None:
    assert Transition.stay().kind is TransitionKind.STAY
    assert Transition.stay().route is None
    assert Transition.switch(Route.QA).route is Route.QA
    assert Transition.suspend(Route.TEST_REVIEW).kind is TransitionKind.SUSPEND

    # switch / suspend require a target route
    for kind in (TransitionKind.SWITCH, TransitionKind.SUSPEND):
        try:
            Transition(kind)
            raise AssertionError(f"{kind} without a route should have raised")
        except ValueError:
            pass

    # stay must not carry a route
    try:
        Transition(TransitionKind.STAY, Route.QA)
        raise AssertionError("stay with a route should have raised")
    except ValueError:
        pass


def test_registration_guards() -> None:
    router = Router(ScriptedClassifier())
    router.register(DiagnosisStub())

    # duplicate route rejected
    try:
        router.register(DiagnosisStub())
        raise AssertionError("duplicate route registration should have raised")
    except ValueError:
        pass

    # UNKNOWN path rejected
    class UnknownPath(Path):
        route = Route.UNKNOWN

        async def handle(self, *, turn: str, state: ConsultationState) -> PathResult:
            return PathResult("x")

    try:
        router.register(UnknownPath())
        raise AssertionError("registering a Route.UNKNOWN path should have raised")
    except ValueError:
        pass

    assert router.routes == frozenset({Route.DIAGNOSIS})
    assert router.has(Route.DIAGNOSIS) and not router.has(Route.QA)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_routes_among_peers_no_diagnosis_root,
    test_active_path_continues_without_reclassifying,
    test_suspend_resume_round_trip,
    test_unknown_classification_asks_to_clarify,
    test_unregistered_route_from_classifier_clarifies,
    test_transition_invariants,
    test_registration_guards,
]


def main() -> None:
    passed = failed = 0
    for test in ALL_TESTS:
        try:
            result = test()
            if inspect.iscoroutine(result):
                asyncio.run(result)
            print(f"PASS  {test.__name__}")
            passed += 1
        except Exception as exc:  # noqa: BLE001 - test runner reports everything
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
