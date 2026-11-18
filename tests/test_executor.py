"""
tests/test_executor.py
----------------------
Standalone tests for File 7 (the parallel DAG executor). Run from the project
root:

    python -m tests.test_executor

No pytest, no pydantic, no network — the executor is pure asyncio, so this whole
suite runs anywhere.

What these tests PROVE:
    - independent tasks run CONCURRENTLY (peak concurrency, not wall-clock luck);
    - a linear chain runs in order and passes results down the context;
    - a diamond DAG resolves correctly with the join running once after both arms;
    - duplicate names, unknown deps, self-deps, and cycles raise before running;
    - a task failure is reported, its REQUIRED dependents are skipped, and an
      independent branch still completes;
    - a per-task timeout is reported and cancels only that task;
    - an OPTIONAL dependency that fails does NOT skip its dependent; the dependent
      runs without that result;
    - the shared `state` object is visible to every task.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import traceback

from app.execution import (
    ExecutionError,
    Executor,
    Task,
    TaskContext,
    TaskStatus,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class ConcurrencyProbe:
    """Tracks how many task bodies are running at once (peak)."""

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0

    def make(self, value: str, sleep: float = 0.02):
        async def body(ctx: TaskContext) -> str:
            self.current += 1
            self.peak = max(self.peak, self.current)
            await asyncio.sleep(sleep)
            self.current -= 1
            return value
        return body


def const(value):
    async def body(ctx: TaskContext):
        return value
    return body


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

async def test_independent_tasks_run_concurrently() -> None:
    probe = ConcurrencyProbe()
    tasks = [Task(name=f"t{i}", run=probe.make(f"v{i}")) for i in range(3)]

    report = await Executor().run(tasks)

    assert probe.peak == 3, f"expected 3 concurrent, saw peak {probe.peak}"
    assert report.all_ok
    assert report.ok_results() == {"t0": "v0", "t1": "v1", "t2": "v2"}


async def test_linear_chain_passes_results_down() -> None:
    async def start(ctx: TaskContext) -> int:
        return 1

    async def add(ctx: TaskContext) -> int:
        return ctx.results["start"] + 10

    async def double(ctx: TaskContext) -> int:
        return ctx.results["add"] * 2

    tasks = [
        Task(name="start", run=start),
        Task(name="add", run=add, deps=("start",)),
        Task(name="double", run=double, deps=("add",)),
    ]
    report = await Executor().run(tasks)
    assert report.result("double") == 22
    assert report.all_ok


async def test_diamond_join_runs_once_after_both_arms() -> None:
    probe = ConcurrencyProbe()
    join_inputs = {}

    async def join(ctx: TaskContext) -> str:
        join_inputs.update(ctx.results)
        return "joined"

    tasks = [
        Task(name="root", run=const("r")),
        Task(name="left", run=probe.make("L"), deps=("root",)),
        Task(name="right", run=probe.make("R"), deps=("root",)),
        Task(name="join", run=join, deps=("left", "right")),
    ]
    report = await Executor().run(tasks)

    assert report.result("join") == "joined"
    assert join_inputs == {"left": "L", "right": "R"}, join_inputs
    assert probe.peak == 2, f"left/right should overlap; peak {probe.peak}"


async def test_required_failure_skips_dependent_but_not_independent_branch() -> None:
    async def boom(ctx: TaskContext):
        raise ValueError("kaboom")

    tasks = [
        Task(name="bad", run=boom),
        Task(name="needs_bad", run=const("x"), deps=("bad",)),
        Task(name="independent", run=const("ok")),
    ]
    report = await Executor().run(tasks)

    assert report.outcome("bad").status is TaskStatus.FAILED
    assert isinstance(report.outcome("bad").error, ValueError)
    assert report.outcome("needs_bad").status is TaskStatus.SKIPPED
    assert "bad" in (report.outcome("needs_bad").skipped_reason or "")
    assert report.outcome("independent").status is TaskStatus.OK
    assert not report.all_ok
    assert set(report.failures) == {"bad", "needs_bad"}


async def test_skip_cascades_through_chain() -> None:
    async def boom(ctx: TaskContext):
        raise RuntimeError("x")

    tasks = [
        Task(name="a", run=boom),
        Task(name="b", run=const("b"), deps=("a",)),
        Task(name="c", run=const("c"), deps=("b",)),
    ]
    report = await Executor().run(tasks)
    assert report.outcome("a").status is TaskStatus.FAILED
    assert report.outcome("b").status is TaskStatus.SKIPPED
    assert report.outcome("c").status is TaskStatus.SKIPPED


async def test_timeout_is_reported_and_isolated() -> None:
    async def slow(ctx: TaskContext):
        await asyncio.sleep(1.0)
        return "should not get here"

    tasks = [
        Task(name="slow", run=slow, timeout=0.05),
        Task(name="fast", run=const("quick")),
    ]
    started = time.perf_counter()
    report = await Executor().run(tasks)
    elapsed = time.perf_counter() - started

    assert report.outcome("slow").status is TaskStatus.TIMEOUT
    assert report.outcome("fast").status is TaskStatus.OK
    assert elapsed < 0.8, f"timeout should not wait the full sleep (took {elapsed:.2f}s)"


async def test_optional_dependency_failure_does_not_skip_dependent() -> None:
    async def boom(ctx: TaskContext):
        raise ValueError("evidence source down")

    async def rank(ctx: TaskContext) -> dict:
        # ranking runs even though the optional source failed; it just isn't
        # in ctx.results.
        return dict(ctx.results)

    tasks = [
        Task(name="pubmed", run=boom),
        Task(name="guidelines", run=const(["guide-1"])),
        Task(
            name="ranking", run=rank,
            optional_deps=("pubmed", "guidelines"),
        ),
    ]
    report = await Executor().run(tasks)

    assert report.outcome("pubmed").status is TaskStatus.FAILED
    assert report.outcome("ranking").status is TaskStatus.OK
    assert report.result("ranking") == {"guidelines": ["guide-1"]}, report.result("ranking")


async def test_optional_dep_waited_for_before_dependent_runs() -> None:
    order = []

    async def slow_optional(ctx: TaskContext):
        await asyncio.sleep(0.03)
        order.append("optional")
        return "late"

    async def dependent(ctx: TaskContext):
        order.append("dependent")
        return ctx.results.get("opt")

    tasks = [
        Task(name="opt", run=slow_optional),
        Task(name="dep", run=dependent, optional_deps=("opt",)),
    ]
    report = await Executor().run(tasks)
    # the dependent must wait for the optional to finish, then see its result
    assert order == ["optional", "dependent"], order
    assert report.result("dep") == "late"


async def test_shared_state_is_visible_to_tasks() -> None:
    async def writer(ctx: TaskContext):
        ctx.state["touched"] = True
        return None

    state: dict = {}
    await Executor().run([Task(name="w", run=writer)], state=state)
    assert state == {"touched": True}


def test_cycle_is_rejected_before_running() -> None:
    tasks = [
        Task(name="a", run=const("a"), deps=("b",)),
        Task(name="b", run=const("b"), deps=("a",)),
    ]
    try:
        asyncio.run(Executor().run(tasks))
        raise AssertionError("expected ExecutionError for a cycle")
    except ExecutionError as exc:
        assert "cycle" in str(exc).lower()


def test_unknown_dependency_is_rejected() -> None:
    tasks = [Task(name="a", run=const("a"), deps=("ghost",))]
    try:
        asyncio.run(Executor().run(tasks))
        raise AssertionError("expected ExecutionError for unknown dep")
    except ExecutionError as exc:
        assert "unknown" in str(exc).lower()


def test_duplicate_name_is_rejected() -> None:
    tasks = [Task(name="a", run=const(1)), Task(name="a", run=const(2))]
    try:
        asyncio.run(Executor().run(tasks))
        raise AssertionError("expected ExecutionError for duplicate name")
    except ExecutionError as exc:
        assert "duplicate" in str(exc).lower()


def test_self_dependency_is_rejected() -> None:
    tasks = [Task(name="a", run=const(1), deps=("a",))]
    try:
        asyncio.run(Executor().run(tasks))
        raise AssertionError("expected ExecutionError for self-dependency")
    except ExecutionError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_independent_tasks_run_concurrently,
    test_linear_chain_passes_results_down,
    test_diamond_join_runs_once_after_both_arms,
    test_required_failure_skips_dependent_but_not_independent_branch,
    test_skip_cascades_through_chain,
    test_timeout_is_reported_and_isolated,
    test_optional_dependency_failure_does_not_skip_dependent,
    test_optional_dep_waited_for_before_dependent_runs,
    test_shared_state_is_visible_to_tasks,
    test_cycle_is_rejected_before_running,
    test_unknown_dependency_is_rejected,
    test_duplicate_name_is_rejected,
    test_self_dependency_is_rejected,
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
        except Exception as exc:  # noqa: BLE001 - runner reports everything
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
