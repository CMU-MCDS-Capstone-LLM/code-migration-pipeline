from typing import Collection
from infra.pipeline.config import MigConfig
from infra.pipeline.task import Task, TaskId


class RepoContainerTask(Task):
    """Spin up repo container (with repograph server)"""

    def __init__(self, config: MigConfig, depends_on: Collection[TaskId]):
        task_id = TaskId(f"repo_container_{config.identifier}")
