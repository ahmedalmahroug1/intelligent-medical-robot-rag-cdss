"""
app/routing/contract.py
-----------------------
The PATH CONTRACT: what every clinical path IS, and what it RETURNS.

Three things live here, all part of one idea — how a path reports its outcome to
the router WITHOUT ever touching another path:

    Path        — the abstract base every clinical path implements. It handles
                  exactly one Route and exposes a single async `handle(...)`.
    Transition  — a path's REQUEST to the router about what happens next: stay on
                  me, switch to another path, or suspend me and run another path
                  (resuming me later). Exactly three kinds, by deliberate design.
    PathResult  — what `handle` returns: the patient-facing reply plus the
                  Transition request.

Why transitions are *requests*, not actions
    A path never imports, constructs, or calls another path. It returns a
    Transition naming a Route (an enum value = pure data). The ROUTER performs
    the actual switch and owns `session.active_route`. This is the same
    blackboard discipline the agents follow: components coordinate through shared
    state and a single deciding authority, never by calling each other. It is the
    rule that stops a multi-path system from turning into paths-calling-paths.

Why PathResult / Transition are frozen dataclasses (not Pydantic models)
    They are per-turn CONTROL FLOW — produced by a path, consumed immediately by
    the router, never persisted and never written to the blackboard. So they do
    NOT belong in state.py (which is the persisted blackboard). A frozen
    dataclass is the right tool: immutable, dependency-light, and consistent with
    the other infrastructure value type in the codebase (LLMMessage in
    app/llm/base.py). The handoff grouped "PathResult" with the state models; on
    building it the cleaner placement is here, in the routing layer. The one
    piece of transition state that genuinely OUTLIVES the turn — where to resume
    after a suspend — does live on the blackboard, as `VisitState.resume_route`.

Note on imports: ConsultationState is imported only under TYPE_CHECKING. With
`from __future__ import annotations` the type hints are strings and never
evaluated at runtime, so this module (and the whole routing package) carries no
runtime dependency on the heavy pydantic state module — it depends only on the
Route enum. That keeps the layering clean and lets the contract be exercised in
isolation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from app.core.enums import Route

if TYPE_CHECKING:  # type-only; not imported at runtime
    from app.core.state import ConsultationState


class TransitionKind(str, Enum):
    """
    The three — and only three — things a path can ask the router to do next.

    STAY     keep this same path active for the next turn (the common case: an
             interview still in progress, a Q&A thread continuing).
    SWITCH   this path is done; hand the conversation to another path, with no
             expectation of return. (Also how a suspended interim path hands
             control BACK: it switches to the route it read from resume_route.)
    SUSPEND  pause this path, run another one for a while, then come back. The
             suspending path writes its OWN resume target to
             VisitState.resume_route before returning this; the interim path
             reads that target when it finishes. Deliberately NOT a general call
             stack — a single resume target until a real case needs more.
    """
    STAY = "stay"
    SWITCH = "switch"
    SUSPEND = "suspend"


@dataclass(frozen=True)
class Transition:
    """
    A path's request to the router about what happens after this turn.

    Construct via the factories (`Transition.stay()`, `.switch(route)`,
    `.suspend(route)`) rather than the raw constructor — they enforce the
    invariant that STAY carries no route while SWITCH/SUSPEND require one.
    """
    kind: TransitionKind
    route: Route | None = None

    def __post_init__(self) -> None:
        needs_route = self.kind in (TransitionKind.SWITCH, TransitionKind.SUSPEND)
        if needs_route and self.route is None:
            raise ValueError(f"{self.kind.value} transition requires a target route")
        if not needs_route and self.route is not None:
            raise ValueError(f"{self.kind.value} transition must not carry a route")

    @classmethod
    def stay(cls) -> Transition:
        """Keep the current path active next turn."""
        return cls(TransitionKind.STAY)

    @classmethod
    def switch(cls, route: Route) -> Transition:
        """Hand off to `route` and end this path's involvement."""
        return cls(TransitionKind.SWITCH, route)

    @classmethod
    def suspend(cls, route: Route) -> Transition:
        """
        Run `route` next, intending to resume this path later. The caller MUST
        have written its own resume target to `state.visit.resume_route` before
        returning this transition.
        """
        return cls(TransitionKind.SUSPEND, route)


@dataclass(frozen=True)
class PathResult:
    """
    What a path returns from `handle`: the patient-facing reply and the
    transition request. Defaults to STAY, so a path that simply replies need not
    spell out "and stay on me."
    """
    reply: str
    transition: Transition = field(default_factory=Transition.stay)


class Path(ABC):
    """
    A clinical path: one self-contained way of handling a patient turn
    (diagnosis, treatment, Q&A, follow-up, test-review). Every path is a PEER —
    the router treats them uniformly and none is privileged as a default root.

    A path implements exactly two things:
        `route`  — the single Route it handles (set as a class attribute).
        `handle` — produce a PathResult for one turn, reading from and writing to
                   the blackboard (ConsultationState) as needed.

    A path MUST NOT import or call another path. To involve another path it
    returns a Transition; the router decides and performs the switch.
    """

    @property
    @abstractmethod
    def route(self) -> Route:
        """The single route this path handles. Subclasses provide it as a class
        attribute, e.g. `route = Route.DIAGNOSIS`."""
        ...

    @abstractmethod
    async def handle(self, *, turn: str, state: ConsultationState) -> PathResult:
        """
        Handle one patient turn; return the reply plus a transition request.

        Contract:
            - `turn` is the current patient message. By the time a path runs, the
              safety guard has already cleared the turn (the guard sits ABOVE the
              router), so a path does not re-run red-flag checks.
            - The path reads context from, and writes its structured contribution
              to, the three blackboard tiers in `state`. It does NOT append the
              reply to the transcript — transcript management belongs to the
              orchestrator around the turn (the same convention the guard uses:
              the current turn is not yet in `session.transcript`).
            - A path identifies ITSELF via `self.route`; it must not rely on
              `session.active_route` already pointing at it (the router sets that
              from the transition the path returns).
            - The path must run safely against whatever state exists; it must not
              assume it produced earlier turns in this session.
        """
        ...
