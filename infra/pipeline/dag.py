from collections import defaultdict
import logging
from typing import Dict, List

from .task import Task, TaskId
from .context import PipelineContext

class DAGExecutor:
    """Topological sort executor with parallel execution support"""
    tasks: Dict[TaskId, Task]
    context: PipelineContext
    logger: logging.Logger
    
    def __init__(self, tasks: List[Task]):
        self.tasks = {task.task_id: task for task in tasks}
        self.context = PipelineContext()
        self.logger = logging.getLogger(__name__)
    
    def _build_dependency_graph(self) -> Dict[TaskId, List[TaskId]]:
        """Returns adjacency list of dependencies"""
        self.logger.debug("Build dependency graph for pipeline.")
        graph = defaultdict(list)
        for task in self.tasks.values():
            for dep_id in task.depended_by:
                graph[dep_id].append(task.task_id)
        return graph
    
    def _topological_sort(self) -> List[List[TaskId]]:
        """Returns list of execution levels (can run in parallel)"""
        graph = self._build_dependency_graph()
        self.logger.debug("Topological sort the dependency graph to get execution order.")
        in_degree = defaultdict(int)
        
        # Calculate in-degrees
        for task_id in self.tasks:
            in_degree[task_id] = len(self.tasks[task_id].depended_by)
        
        # Find tasks with no dependencies (level 0)
        levels = []
        current_level = [tid for tid, deg in in_degree.items() if deg == 0]
        
        while current_level:
            levels.append(current_level)
            next_level = []
            
            for task_id in current_level:
                for dependent_id in graph[task_id]:
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        next_level.append(dependent_id)
            
            current_level = next_level
        
        return levels
    
    def execute(self, parallel: bool = False):
        """Execute DAG level by level"""
        levels = self._topological_sort()
        self.logger.info("Execute pipeline.")

        # Technically, we need a thread pool, but our pipeline is very small, so this will work as well.
        for level_idx, level in enumerate(levels):
            print(f"Level {level_idx}: {len(level)} task(s)")
            
            if parallel and len(level) > 1:
                # Use ThreadPoolExecutor for parallel execution
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=len(level)) as executor:
                    futures = {
                        executor.submit(self._execute_task, tid): tid 
                        for tid in level
                    }
                    for future in futures:
                        future.result()  # Wait and handle errors
            else:
                # Sequential execution
                for task_id in level:
                    self._execute_task(task_id)
    
    def _execute_task(self, task_id: TaskId):
        """Execute a single task"""
        task = self.tasks[task_id]
        
        # Inject dependencies from context
        for dep_id in task.depended_by:
            dep_artifact = self.context.get_artifact(dep_id)
            task.inputs[dep_id] = dep_artifact
        
        # Execute
        result = task.execute()
        
        # Store result
        self.context.store_artifact(task_id, result)

