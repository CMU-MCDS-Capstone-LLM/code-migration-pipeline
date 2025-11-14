# infra/tasks/eval_testing_agent.py
from ..pipeline.task import Task


class EvalTestingAgentTask(Task[None]):
    def should_run(self) -> bool:
        return True

    def load_cached_result(self) -> None:
        raise NotImplementedError(
            "Load cached result for eval testing agent has not been implemented!"
        )

    def run(self) -> None:
        pass
