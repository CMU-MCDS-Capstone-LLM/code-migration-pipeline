from dataclasses import dataclass
from typing import Collection
from pprint import pformat

from infra.pipeline.config import MigConfig
from infra.pipeline.task import Task, TaskId
from infra.tasks.env_setup_agent import EnvSetup


class RepoGraphServer:
    pass


@dataclass
class RepoContainer:
    """Docker container that runs the repository to be migrated.

    Note that this container also runs the repograph server, which answers repograph query on the repo. Repo and RepoGraph server runs in two different python version and two isolated python environments. This is made possible by pyenv
    """

    name: str
    repograph_server: RepoGraphServer


class RepoContainerTask(Task[RepoContainer]):
    """Spin up repo container (with repograph server)"""

    env_setup_task: Task[EnvSetup]

    def __init__(self, config: MigConfig, env_setup_task: Task[EnvSetup]):
        task_id = TaskId(f"repo_container_{config.identifier}")
        super().__init__(task_id, config, [env_setup_task])
        self.env_setup_task = env_setup_task

    def should_run(self) -> bool:
        return True

    def load_cached_result(self) -> RepoContainer:
        """Load existing repo info."""
        raise NotImplementedError(
            "load cached result for repo container task is not implemented"
        )

    def run(self) -> RepoContainer:
        """Download the repo."""
        env_setup = self.get_dep_output(self.env_setup_task)
        self.logger.info(f"Spin up container from env setup:\n{pformat(env_setup)}")
        return RepoContainer(name="???", repograph_server=RepoGraphServer())

    def cleanup(self) -> None:
        # TODO: Tear down repo container
        return super().cleanup()
