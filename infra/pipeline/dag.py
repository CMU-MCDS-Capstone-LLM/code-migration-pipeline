# infra/pipeline/dag.py
from __future__ import annotations

from collections import defaultdict
import logging
from typing import Any, Dict, Iterable, List
import traceback

from .task import Task


class DAGExecutor:
    """
    Topological sort executor.

    - Works over Task objects directly.
    - Assumes each Task.execute() handles caching and output.
    """

    def __init__(self, tasks: Iterable[Task[Any]]):
        self.tasks: List[Task[Any]] = list(tasks)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _build_dependency_graph(self) -> Dict[Task[Any], List[Task[Any]]]:
        """Returns adjacency list: dep_task -> [tasks that depend on it]."""
        graph: Dict[Task[Any], List[Task[Any]]] = defaultdict(list)
        for task in self.tasks:
            for dep in task.depends_on:
                graph[dep].append(task)
        return graph

    def _topological_levels(self) -> List[List[Task[Any]]]:
        """
        Returns list of execution levels (can run in parallel if needed).

        Each level is a list of tasks whose dependencies are all in earlier levels.
        """
        graph = self._build_dependency_graph()
        in_degree: Dict[Task[Any], int] = {}

        for task in self.tasks:
            in_degree[task] = len(task.depends_on)

        levels: List[List[Task[Any]]] = []
        current_level = [t for t, deg in in_degree.items() if deg == 0]

        while current_level:
            levels.append(current_level)
            next_level: List[Task[Any]] = []
            for t in current_level:
                for child in graph.get(t, []):
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_level.append(child)
            current_level = next_level

        # Sanity check: did we cover all tasks?
        if sum(len(l) for l in levels) != len(self.tasks):
            raise RuntimeError("Cycle detected in task graph")

        return levels

    def execute(self) -> None:
        """Execute the DAG level by level (single-threaded)."""
        levels = self._topological_levels()
        self.logger.info("Execute pipeline.")

        cleanup_queue: List[Task["Any"]] = []

        try:
            for level_idx, level in enumerate(levels):
                self.logger.info("Level %d: %d task(s)", level_idx, len(level))
                for task in level:
                    cleanup_queue.append(task)
                    task.execute()
        except Exception as e:
            if len(cleanup_queue) > 0:
                cur_task = cleanup_queue[-1]
                self.logger.error(f"Encounter error when executing task {cur_task}.")
            else:
                self.logger.error("Encounter error when executing task.")
            self.logger.error(f"Error message: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            for task in cleanup_queue[::-1]:
                try:
                    task.cleanup()
                except Exception as e:
                    # If cleanup of one task fails, ignore it and continue to clean up other tasks
                    self.logger.error(f"Got error when clean up task {task}: {e}")
