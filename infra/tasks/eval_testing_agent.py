from ..pipeline.task import Task 

class EvalTesingAgentTask(Task):
    def should_run(self) -> bool:
        return True
    
    def load_cached_result(self) -> None:
        raise NotImplementedError("Load cached result for eval testing agent has not been implemented!")
    
    def run(self) -> None:
        pass

