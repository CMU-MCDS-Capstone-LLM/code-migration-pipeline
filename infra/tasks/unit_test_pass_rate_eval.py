from ..pipeline.task import Task 

class UnitTestPassRateEvalTask(Task):
    def should_run(self) -> bool:
        return True
    
    def load_cached_result(self) -> None:
        raise NotImplementedError("Load cached result for unit test pass rate evaluation task has not been implemented!")
    
    def run(self) -> None:
        pass

