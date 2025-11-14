# example/task/IncrTask.py
from infra.pipeline.task import Task, TaskId
from infra.pipeline.config import MigConfig
from .SourceTask import SourceTask, ValueOutput


class IncrTask(Task[ValueOutput]):
    """Task that increments input by a fixed delta."""

    def __init__(
        self, task_name: str, source: SourceTask, delta: int, config: MigConfig
    ) -> None:
        task_id = TaskId(task_name)
        super().__init__(task_id, config, depends_on=[source])
        self.source = source
        self.delta = delta

    def should_run(self) -> bool:
        """Always run - no caching for demo."""
        return True

    def load_cached_result(self) -> ValueOutput:
        """No caching in this demo."""
        return ValueOutput(value=0)

    def run(self) -> ValueOutput:
        """Increment the input value by delta."""
        source_output: ValueOutput = self.get_dep_output(self.source)
        result = source_output.value + self.delta
        print(
            f"  [{self.task_id}] Incrementing {source_output.value} by {self.delta} = {result}"
        )
        return ValueOutput(value=result)
