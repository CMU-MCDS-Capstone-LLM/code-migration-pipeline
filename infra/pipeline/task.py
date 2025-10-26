from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Collection, Dict, Iterable, List, Self
from enum import Enum, auto
import logging

from .config import MigConfig

@dataclass
class TaskId:
    id: str

    def __str__(self):
        return f"task_{self.id}"

class TaskStatus(Enum):
    """
    Status values for pipeline tasks during execution.
    
    Tasks transition through these states during their lifecycle:
    PENDING -> RUNNING -> {COMPLETED | CACHED | FAILED }
    """
    
    PENDING = "pending"
    """
    Initial state when task is created.
    Task has not started execution yet and is waiting for its dependencies.
    """
    
    RUNNING = "running"
    """
    Task is currently executing.
    Set when execute() is called and should_run() returns True.
    """
    
    COMPLETED = "completed"
    """
    Task has successfully finished executing and produced new output.
    Set after run() completes without errors.
    """
    
    CACHED = "cached"
    """
    Task was skipped because cached results were reused.
    Set when should_run() returns False and load_cached_result() succeeds.
    """
    
    FAILED = "failed"
    """
    Task execution failed with an error.
    Set when run() raises an exception.
    The task's output is None and the exception should be propagated.
    """
    
class Task(ABC):
    """
    Base class for a task within the pipeline

    Each task reprersents a runnable unit that takes input from some task, 
    and generate output for other tasks.
    """
    task_id: TaskId 
    config: MigConfig
    inputs: Dict[TaskId, Any]
    output: Any
    status: TaskStatus
    depended_by: Collection[TaskId]
    
    def __init__(self, task_id: TaskId, config: MigConfig, depends_on: Collection[TaskId]):
        self.task_id = task_id
        self.config = config
        self.inputs = {}
        self.output = None
        self.status = TaskStatus.PENDING
        self.depended_by = depends_on
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def run(self) -> Any:
        """Execute the task logic"""
        pass
    
    @abstractmethod
    def should_run(self) -> bool:
        """
        Check if task needs to execute or can reuse previous result.
        Return True to execute, False to skip and load cached result.
        """
        pass
    
    @abstractmethod
    def load_cached_result(self) -> Any:
        """Load and return the cached result"""
        pass
    
    def execute(self) -> Any:
        """Wrapper that handles caching, logging, error handling"""
        
        # Check if we can skip execution
        if not self.should_run():
            self.logger.info(f"Reusing cached result for {self.task_id}")
            self.output = self.load_cached_result()
            self.status = TaskStatus.CACHED
            return self.output
        
        # Execute task
        self.logger.info(f"Running {self.task_id}...")
        self.status = TaskStatus.RUNNING
        try:
            self.output = self.run()
            self.status = TaskStatus.COMPLETED
            self.logger.info(f"Completed {self.task_id}")
            return self.output
        except Exception as e:
            self.status = TaskStatus.FAILED
            self.logger.error(f"Failed {self.task_id}: {e}")
            raise
