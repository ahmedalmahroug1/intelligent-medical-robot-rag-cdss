"""
app/execution
-------------
The parallel DAG executor that drives the diagnosis milestone's fan-out. Generic
and standard-library only — the clinical milestone (File 7b) supplies the Tasks.

Public surface:
    Executor                         — the engine
    Task, TaskContext                — defining a node and what it receives
    ExecutionReport, TaskOutcome     — results
    TaskStatus                       — ok / failed / timeout / skipped
    ExecutionError                   — invalid graph (raised before running)
"""

from app.execution.executor import (
    ExecutionError,
    ExecutionReport,
    Executor,
    Task,
    TaskContext,
    TaskOutcome,
    TaskStatus,
)

__all__ = [
    "Executor",
    "Task",
    "TaskContext",
    "ExecutionReport",
    "TaskOutcome",
    "TaskStatus",
    "ExecutionError",
]
