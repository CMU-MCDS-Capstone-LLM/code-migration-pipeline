from typing import Collection 
import subprocess

from ..pipeline.config import MigConfig
from ..pipeline.task import Task, TaskId

class CodingAgentTask(Task):
    """Download repository from PyMigBench"""
    
    def __init__(self, config: MigConfig, depends_on: Collection[TaskId]):
        task_id = TaskId(f"coding_agent_{config.identifier}")
        super().__init__(task_id, config, depends_on)
    
    def should_run(self) -> bool:
        """Run only if repo doesn't exist"""
        return True
    
    def load_cached_result(self) -> None:
        """Load existing repo info"""
        raise NotImplementedError("Load cached result for coding agent is not implemented")
    
    def run(self) -> None:
        """Download the repo"""
        subprocess.run(["docker", "compose", "up", "--build"], cwd="coding-agent/", check=True)
