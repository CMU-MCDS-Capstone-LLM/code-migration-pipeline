# example/task/AddTask.py
from dataclasses import dataclass
from infra.pipeline.task import Task, TaskId
from infra.pipeline.config import MigConfig
from .IncrTask import ValueOutput


@dataclass
class AddOutput:
    """Output type for AddTask."""

    result: int


class AddTask(Task[AddOutput]):
    """Task that adds two input values."""

    def __init__(
        self,
        task_name: str,
        input1: Task[ValueOutput],
        input2: Task[ValueOutput],
        config: MigConfig,
    ) -> None:
        task_id = TaskId(task_name)
        super().__init__(task_id, config, depends_on=[input1, input2])
        self.input1 = input1
        self.input2 = input2

    def should_run(self) -> bool:
        """Always run - no caching for demo."""
        return True

    def load_cached_result(self) -> AddOutput:
        """No caching in this demo."""
        return AddOutput(result=0)

    def run(self) -> AddOutput:
        """Add the two input values."""
        output1: ValueOutput = self.get_dep_output(self.input1)
        output2: ValueOutput = self.get_dep_output(self.input2)
        result = output1.value + output2.value
        print(f"  [{self.task_id}] Adding {output1.value} + {output2.value} = {result}")
        return AddOutput(result=result)
