# example/task/SourceTask.py
from dataclasses import dataclass
from infra.pipeline.task import Task, TaskId
from infra.pipeline.config import MigConfig


@dataclass
class ValueOutput:
    """Output type for SourceTask."""

    value: int


class SourceTask(Task[ValueOutput]):
    """Source task that emits a fixed number."""

    def __init__(self, task_name: str, value: int, config: MigConfig) -> None:
        task_id = TaskId(task_name)
        super().__init__(task_id, config, depends_on=[])
        self.value = value

    def should_run(self) -> bool:
        """Always run - no caching for demo."""
        return True

    def load_cached_result(self) -> ValueOutput:
        """No caching in this demo."""
        return ValueOutput(value=self.value)

    def run(self) -> ValueOutput:
        """Return the fixed value."""
        print(f"  [{self.task_id}] Emitting value: {self.value}")
        return ValueOutput(value=self.value)
