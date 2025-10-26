from typing import Dict, Any

from infra.pipeline.task import TaskId

class PipelineContext:
    """Shared context for all tasks"""
    artifacts: Dict[TaskId, Any]
    
    def __init__(self):
        self.artifacts: Dict[TaskId, Any] = {}
    
    def store_artifact(self, task_id: TaskId, artifact: Any):
        """Store task result"""
        self.artifacts[task_id] = artifact
    
    def get_artifact(self, task_id: TaskId) -> Any:
        """Retrieve task result"""
        return self.artifacts.get(task_id)

