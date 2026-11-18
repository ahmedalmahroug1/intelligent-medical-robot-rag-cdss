"""
app/routing/router.py
---------------------
The ROUTER: the single authority that decides which path handles a turn and
applies the transition a path requests afterwards. It is the path-level analogue
of the diagnostic loop's controller — one deciding authority per level, exactly
as the blackboard model prescribes.

What it does each turn (it is invoked AFTER the safety guard has cleared the turn
— the guard sits above the router and handles escalation itself, so emergencies
never reach here):

    1. Determine the route:
         - if a registered path is already active            → continue it;
         - else if a suspended path is waiting to resume      → resume it;
         - else classify the turn fresh.
    2. Dispatch the turn to that path's `handle`.
    3. Apply the path's requested Transition to the blackboard (update
       `session.active_route`, and clear `visit.resume_route` when an episode
       ends).

It NEVER reaches into a path's logic, and a path NEVER reaches into it or into
another path: they communicate only through PathResult / Transition and the
shared blackboard. That one rule is what the whole multi-path design rests on.

Why two-or-more paths from the very first line of this file
    The router is built and tested against a REGISTRY of peers, with no special
    case for diagnosis — there is structurally no "root" path it falls back to.
    The diagnostic path (File 6) will register exactly like any other. This is
    the concrete guarantee behind the design rule that diagnosis must not become
    the de facto root: it is enforced here as code, and asserted by the tests.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.core.enums import Route
from app.routing.classifier import TurnClassifier
from app.routing.contract import Path, PathResult, Transition, TransitionKind

if TYPE_CHECKING:  # type-only; not imported at runtime
    from app.core.state import ConsultationState

logger = logging.getLogger(__name__)


# Shown when the router cannot tell which path a turn belongs to (the classifier
# returned UNKNOWN, or named a route with no registered path). Asking beats
# guessing: a wrong path is a worse patient experience than one clarifying line.
_CLARIFY_REPLY = (
    "I want to make sure I help with the right thing. Could you tell me a little "
    "more about what you need — for example, are you describing new symptoms, "
    "asking about a treatment, or following up on a previous visit or test result?"
)


class Router:
    """Decides the handling path for each turn and applies path transitions."""

    def __init__(self, classifier: TurnClassifier) -> None:
        self._classifier = classifier
        self._paths: dict[Route, Path] = {}

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, path: Path) -> None:
        """
        Register a path under the route it handles.

        Raises if the path's route is UNKNOWN (not a real handler) or if another
        path is already registered for that route (dispatch would be ambiguous).
        """
        route = path.route
        if route is Route.UNKNOWN:
            raise ValueError("A path cannot be registered for Route.UNKNOWN.")
        if route in self._paths:
            raise ValueError(
                f"A path is already registered for route '{route.value}'."
            )
        self._paths[route] = path

    @property
    def routes(self) -> frozenset[Route]:
        """The routes that currently have a registered path."""
        return frozenset(self._paths)

    def has(self, route: Route) -> bool:
        """Whether a path is registered for `route`."""
        return route in self._paths

    # ── Dispatch ─────────────────────────────────────────────────────────────

    async def route_turn(self, *, turn: str, state: ConsultationState) -> PathResult:
        """
        Route one turn to a path and apply the resulting transition.

        Assumes the safety guard has already run and permitted the turn to
        continue (escalation is handled above the router). Returns the path's
        PathResult; if the turn cannot be routed, returns a clarifying reply and
        leaves the active route unchanged.
        """
        route = await self._determine_route(turn, state)
        if not self.has(route):
            # UNKNOWN, or a route we have no path for → ask, don't guess.
            return PathResult(reply=_CLARIFY_REPLY, transition=Transition.stay())

        result = await self._paths[route].handle(turn=turn, state=state)
        self._apply_transition(result.transition, handled=route, state=state)
        return result

    async def _determine_route(self, turn: str, state: ConsultationState) -> Route:
        # 1. An active, registered path keeps the conversation.
        active = state.session.active_route
        if self.has(active):
            return active
        # 2. Nothing active, but a suspended path is waiting to resume.
        resume = state.visit.resume_route
        if resume is not None and self.has(resume):
            return resume
        # 3. Fresh classification.
        return await self._classifier.classify(
            turn=turn, state=state, available=self.routes
        )

    def _apply_transition(
        self, transition: Transition, *, handled: Route, state: ConsultationState
    ) -> None:
        """
        Translate a path's transition request into blackboard updates.

        STAY     the path that just handled the turn stays active.
        SWITCH   the named route becomes active; any pending resume is cleared
                 (the suspend/resume episode, if there was one, is now over —
                 this is how an interim path hands control back to its caller).
        SUSPEND  the named (interim) route becomes active. resume_route is NOT
                 touched here: the suspending path already wrote its own resume
                 target before returning, per the path contract.
        """
        session = state.session
        visit = state.visit
        kind = transition.kind

        if kind is TransitionKind.STAY:
            session.active_route = handled
        elif kind is TransitionKind.SWITCH:
            session.active_route = transition.route  # validated non-None
            if visit.resume_route is not None:
                visit.resume_route = None
        elif kind is TransitionKind.SUSPEND:
            session.active_route = transition.route  # the interim path runs next
        else:  # defensive: the enum is closed, but fail loud if it ever grows
            raise ValueError(f"Unhandled transition kind: {kind!r}")
