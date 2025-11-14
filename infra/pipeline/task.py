# infra/pipeline/task.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, List, Sequence, TypeVar, cast
import logging

from .config import MigConfig


@dataclass(frozen=True)
class TaskId:
    id: str

    def __str__(self) -> str:
        return f"task_{self.id}"


class TaskStatus(Enum):
    """
    Status values for pipeline tasks during execution.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CACHED = "cached"
    FAILED = "failed"


TOut = TypeVar("TOut")
TDep = TypeVar("TDep")


class Task(ABC, Generic[TOut]):
    """
    Base class for a task within the pipeline.

    - Each task produces a value of type TOut (its output).
    - Each task can depend on other Task objects (not TaskIds).
    """

    task_id: TaskId
    config: MigConfig
    depends_on: List["Task[Any]"]
    output: TOut | None
    status: TaskStatus

    def __init__(
        self,
        task_id: TaskId,
        config: MigConfig,
        depends_on: Sequence["Task[Any]"] = (),
    ) -> None:
        self.task_id = task_id
        self.config = config
        self.depends_on = list(depends_on)
        self.output = None
        self.status = TaskStatus.PENDING
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def run(self) -> TOut:
        """Execute the main task logic and return the output."""
        ...

    @abstractmethod
    def should_run(self) -> bool:
        """
        Check if task needs to execute or can reuse a previous result.
        Return True to execute, False to skip and load cached result.
        """
        ...

    @abstractmethod
    def load_cached_result(self) -> TOut:
        """Load and return the cached result."""
        ...

    def get_dep_output(self, dep: "Task[TDep]") -> TDep:
        """
        Typed helper to access the output of a dependency.

        Usage:
            repo: Repo = self.get_dep_output(self.repo_dl_task)
        """
        if dep not in self.depends_on:
            raise ValueError(f"{self.task_id} does not depend on {dep.task_id}")

        if dep.output is None and dep.status not in (
            TaskStatus.COMPLETED,
            TaskStatus.CACHED,
        ):
            raise RuntimeError(
                f"Dependency {dep.task_id} has not produced an output yet "
                f"(status={dep.status})"
            )

        return cast(TDep, dep.output)

    def execute(self) -> TOut:
        """
        Execute this task (assuming dependencies have already been executed
        by the DAGExecutor) and return its output.
        """

        # Already done in this run?
        if (
            self.status in (TaskStatus.COMPLETED, TaskStatus.CACHED)
            and self.output is not None
        ):
            return self.output  # type: ignore[return-value]

        # Caching logic
        if not self.should_run():
            self.logger.info(f"Reusing cached result for {self.task_id}")
            self.output = self.load_cached_result()
            self.status = TaskStatus.CACHED
            return self.output  # type: ignore[return-value]

        # Run task
        self.logger.info(f"Running {self.task_id} ...")
        self.status = TaskStatus.RUNNING
        try:
            self.output = self.run()
            self.status = TaskStatus.COMPLETED
            self.logger.info(f"Completed {self.task_id}")
            return self.output  # type: ignore[return-value]
        except Exception:
            self.status = TaskStatus.FAILED
            self.logger.exception(f"Task {self.task_id} failed")
            raise
