"""
app/execution/executor.py
-------------------------
A small parallel DAG executor: the engine that drives the diagnosis MILESTONE's
fan-out (differential -> test recommendation -> PubMed / guidelines / image in
parallel -> ranking -> medication-safety -> working diagnosis). Re-derived from
the V2 executor, whose scheduling was the part worth keeping.

It is deliberately DOMAIN-AGNOSTIC and standard-library only: it knows nothing
about hypotheses, evidence, or the gateway. A caller supplies a set of Tasks
(each an async callable plus its dependency names) and an opaque shared `state`
object; the executor runs them with maximum safe parallelism and returns an
ExecutionReport. That is why it can be tested exhaustively offline, and why the
clinical milestone (File 7b) is just data — a list of Tasks — handed to it.

Guarantees:
    - Parallelism: every task whose dependencies are satisfied is launched
      immediately and concurrently; completions are processed as they arrive
      (asyncio.wait FIRST_COMPLETED), unlocking newly-ready tasks at once.
    - Graph safety: the dependency graph is validated up front — duplicate
      names, unknown dependencies, and cycles all raise ExecutionError BEFORE any
      task runs.
    - Per-task timeout: each task may bound its own runtime; a timeout cancels
      only that task.
    - Graceful partial failure: a task that fails or times out does NOT cancel
      its siblings. Tasks that REQUIRE a failed task are skipped; tasks that list
      it only as an OPTIONAL dependency still run, without its result. This is
      what lets the milestone proceed on guideline evidence when PubMed times out.

Failure policy is intentionally left to the CALLER. The executor records what
happened (ok / failed / timeout / skipped, with results and errors) and returns
it; the milestone inspects the report and decides how to synthesise. The engine
never decides clinical degradation on its own.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ExecutionError(RuntimeError):
    """Raised for an invalid task graph (duplicate name, unknown dep, or cycle).
    This is a programming error in how the graph was built, surfaced before any
    task runs — distinct from a task failing at runtime (which is reported, not
    raised)."""


class TaskStatus(str, Enum):
    OK = "ok"            # ran to completion and returned a result
    FAILED = "failed"    # raised an exception
    TIMEOUT = "timeout"  # exceeded its timeout
    SKIPPED = "skipped"  # a required dependency did not succeed, so it never ran


@dataclass(frozen=True)
class TaskContext:
    """What a task receives when it runs: the shared blackboard plus the results
    of the dependencies that succeeded (by dependency name)."""
    state: Any
    results: Mapping[str, Any]


# A task body: an async function of the context, returning anything.
TaskFn = Callable[[TaskContext], Awaitable[Any]]


@dataclass(frozen=True)
class Task:
    """
    One node in the milestone graph.

    name           unique identifier, and the key its result is stored under.
    run            async callable; receives a TaskContext, returns a result.
    deps           REQUIRED dependencies: if any does not succeed, this task is
                   skipped. Their results are provided in the context.
    optional_deps  BEST-EFFORT dependencies: their results are provided if
                   available, but a missing/failed one never blocks or skips this
                   task. (Used for the parallel evidence sources feeding ranking.)
    timeout        per-task timeout in seconds; None means use the executor
                   default (which may itself be None for no limit).
    """
    name: str
    run: TaskFn
    deps: tuple[str, ...] = ()
    optional_deps: tuple[str, ...] = ()
    timeout: float | None = None

    def all_dep_names(self) -> tuple[str, ...]:
        return (*self.deps, *self.optional_deps)


@dataclass(frozen=True)
class TaskOutcome:
    """The recorded result of a single task."""
    name: str
    status: TaskStatus
    result: Any = None
    error: BaseException | None = None
    skipped_reason: str | None = None
    duration_s: float | None = None

    @property
    def ok(self) -> bool:
        return self.status is TaskStatus.OK


@dataclass(frozen=True)
class ExecutionReport:
    """The outcome of running a whole graph: one TaskOutcome per task."""
    outcomes: dict[str, TaskOutcome] = field(default_factory=dict)

    def outcome(self, name: str) -> TaskOutcome:
        return self.outcomes[name]

    def succeeded(self, name: str) -> bool:
        out = self.outcomes.get(name)
        return out is not None and out.ok

    def result(self, name: str, default: Any = None) -> Any:
        """The task's result if it succeeded, else `default`. Never raises, so a
        caller can ask for an optional source's result without guarding."""
        out = self.outcomes.get(name)
        return out.result if (out is not None and out.ok) else default

    def ok_results(self) -> dict[str, Any]:
        return {n: o.result for n, o in self.outcomes.items() if o.ok}

    @property
    def all_ok(self) -> bool:
        return all(o.ok for o in self.outcomes.values())

    @property
    def failures(self) -> list[str]:
        return [n for n, o in self.outcomes.items() if not o.ok]


# Internal readiness decision for a pending task.
class _Decision(str, Enum):
    NOT_READY = "not_ready"  # a dependency is still undecided
    READY = "ready"          # all deps decided; required ones all succeeded
    SKIP = "skip"            # all deps decided; a required one did not succeed


class Executor:
    """Runs a set of Tasks as a dependency graph, in parallel where possible."""

    def __init__(self, *, default_timeout: float | None = None) -> None:
        self._default_timeout = default_timeout

    async def run(self, tasks: Sequence[Task], *, state: Any = None) -> ExecutionReport:
        index = self._validate(tasks)
        outcomes: dict[str, TaskOutcome] = {}
        pending: set[str] = set(index)
        running: dict[asyncio.Task[TaskOutcome], str] = {}

        while pending or running:
            ready = self._resolve_skips_and_collect_ready(pending, outcomes, index)

            for name in ready:
                task = index[name]
                ctx = TaskContext(
                    state=state, results=self._collect_results(task, outcomes)
                )
                handle = asyncio.create_task(self._run_one(task, ctx))
                running[handle] = name

            if running:
                done, _ = await asyncio.wait(
                    set(running), return_when=asyncio.FIRST_COMPLETED
                )
                for handle in done:
                    name = running.pop(handle)
                    outcomes[name] = handle.result()  # _run_one never raises
            elif pending:
                # Nothing runnable and nothing running, but tasks remain. With a
                # validated (acyclic) graph this only happens if they are blocked
                # on failed deps that the skip-resolution above already cascaded;
                # resolve any stragglers defensively so run() always terminates.
                for name in list(pending):
                    outcomes[name] = TaskOutcome(
                        name, TaskStatus.SKIPPED,
                        skipped_reason="dependency did not succeed",
                    )
                pending.clear()

        return ExecutionReport(outcomes=outcomes)

    # ── Scheduling internals ──────────────────────────────────────────────────

    def _resolve_skips_and_collect_ready(
        self,
        pending: set[str],
        outcomes: dict[str, TaskOutcome],
        index: Mapping[str, Task],
    ) -> list[str]:
        """
        Examine every pending task. Mark as SKIPPED (cascading) any whose required
        dependency did not succeed; collect those that are READY to launch. Repeat
        until no further skips appear, so a chain of skips resolves in one call.
        Tasks left in `pending` are genuinely NOT_READY (waiting on running deps).
        """
        ready: list[str] = []
        changed = True
        while changed:
            changed = False
            for name in list(pending):
                decision, reason = self._evaluate(index[name], outcomes)
                if decision is _Decision.NOT_READY:
                    continue
                pending.discard(name)
                if decision is _Decision.SKIP:
                    outcomes[name] = TaskOutcome(
                        name, TaskStatus.SKIPPED, skipped_reason=reason
                    )
                    changed = True  # may unblock skips of this task's dependents
                else:  # READY
                    ready.append(name)
        return ready

    def _evaluate(
        self, task: Task, outcomes: Mapping[str, TaskOutcome]
    ) -> tuple[_Decision, str | None]:
        # Required deps must all be decided AND successful.
        for dep in task.deps:
            out = outcomes.get(dep)
            if out is None:
                return _Decision.NOT_READY, None
            if not out.ok:
                return _Decision.SKIP, f"required dependency '{dep}' did not succeed"
        # Optional deps must be decided (ok or not) before we run, so their
        # results — if any — are available. They never cause a skip.
        for dep in task.optional_deps:
            if dep not in outcomes:
                return _Decision.NOT_READY, None
        return _Decision.READY, None

    def _collect_results(
        self, task: Task, outcomes: Mapping[str, TaskOutcome]
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for dep in task.all_dep_names():
            out = outcomes.get(dep)
            if out is not None and out.ok:
                results[dep] = out.result
        return results

    async def _run_one(self, task: Task, ctx: TaskContext) -> TaskOutcome:
        """Run one task, capturing success/failure/timeout. Never raises (so the
        scheduling loop can treat every completion uniformly)."""
        timeout = task.timeout if task.timeout is not None else self._default_timeout
        start = time.perf_counter()
        try:
            if timeout is not None:
                result = await asyncio.wait_for(task.run(ctx), timeout)
            else:
                result = await task.run(ctx)
            return TaskOutcome(
                task.name, TaskStatus.OK, result=result,
                duration_s=time.perf_counter() - start,
            )
        except asyncio.TimeoutError as exc:
            logger.warning("Task '%s' timed out after %.2fs", task.name, timeout or 0.0)
            return TaskOutcome(
                task.name, TaskStatus.TIMEOUT, error=exc,
                duration_s=time.perf_counter() - start,
            )
        except Exception as exc:  # noqa: BLE001 - failures are reported, not raised
            logger.warning("Task '%s' failed: %s", task.name, exc)
            return TaskOutcome(
                task.name, TaskStatus.FAILED, error=exc,
                duration_s=time.perf_counter() - start,
            )

    # ── Graph validation (up front, before anything runs) ─────────────────────

    def _validate(self, tasks: Sequence[Task]) -> dict[str, Task]:
        index: dict[str, Task] = {}
        for task in tasks:
            if task.name in index:
                raise ExecutionError(f"Duplicate task name: '{task.name}'")
            index[task.name] = task

        for task in tasks:
            for dep in task.all_dep_names():
                if dep not in index:
                    raise ExecutionError(
                        f"Task '{task.name}' depends on unknown task '{dep}'"
                    )
                if dep == task.name:
                    raise ExecutionError(f"Task '{task.name}' depends on itself")

        self._check_acyclic(index)
        return index

    def _check_acyclic(self, index: Mapping[str, Task]) -> None:
        WHITE, GREY, BLACK = 0, 1, 2
        colour = {name: WHITE for name in index}

        def visit(name: str, stack: list[str]) -> None:
            colour[name] = GREY
            stack.append(name)
            for dep in index[name].all_dep_names():
                if colour[dep] == GREY:
                    cycle = stack[stack.index(dep):] + [dep]
                    raise ExecutionError(
                        "Dependency cycle detected: " + " -> ".join(cycle)
                    )
                if colour[dep] == WHITE:
                    visit(dep, stack)
            stack.pop()
            colour[name] = BLACK

        for name in index:
            if colour[name] == WHITE:
                visit(name, [])
