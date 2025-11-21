import hashlib
from typing import Self
from pprint import pformat
import docker
import logging

from pymigbench_dl.loader import CommitInfo

from infra.pipeline.config import MigConfig
from infra.pipeline.task import Task, TaskId
from infra.tasks.env_setup_agent import EnvSetup


logger = logging.getLogger(__name__)


class RepoContainer:
    """Docker container that runs the repository to be migrated.

    Note that this container also runs the repograph server, which answers repograph query on the repo. Repo and RepoGraph server runs in two different python version and two isolated python environments. This is made possible by pyenv
    """

    name: str
    # TODO: May need a dedicated class, maybe from urllib3?
    repograph_server_url: str

    def __init__(self, name):
        self.name = name
        self.repograph_server_url = ""

    @classmethod
    def from_env_setup(cls, config: MigConfig, env_setup: EnvSetup) -> Self:
        def build_container_name(commit_info: CommitInfo) -> str:
            commit_part = commit_info.folder_name[:32]
            hash_part = hashlib.md5(commit_info.folder_name.encode()).hexdigest()[:8]
            name = f"repo-{commit_part}-{hash_part}"
            return name

        return cls(name=build_container_name(config.commit_info))

    def build(self):
        pass

    def run(self):
        # TODO: Spin up repo container
        # TODO: Then spin up repograph server
        pass

    def destroy(self) -> None:
        logger.debug(f"Cleean up RepoContainer with name {self.name}")
        # TODO: Destroy container
        # NOTE: We don't need to remove image, since it will always be built from Dockerfile, whether DockerOrchTask is cached or not


class DockerContext:
    repo_container: RepoContainer

    def __init__(self, repo_container: RepoContainer):
        self.repo_container = repo_container

    def start(self):
        """
        On return, everything should be set up already. In specific, repo_container and the repograph server in it can be accessed normally on start return
        """
        self._prepare()
        self._build()
        self._run()

    def _run(self):
        """
        Run the repo container. We will
        -
        """
        self.repo_container.run()

    def _prepare(self):
        """
        Set up prerequisites for docker context. This includes
        - a user-hosted network (bridge)
        - ???
        """
        # TODO: Copy repo to a tmp folder on host.
        # - At build time, bind-mount the original repo in read-only mode
        # - At run time, bind-mount the tmp copy in rw mode instead of the original snapshot
        pass

    def _build(self):
        """
        Build containers. This includes
        - the repo container
        """
        self.repo_container.build()

    @classmethod
    def from_env_setup(cls, config: MigConfig, env_setup: EnvSetup) -> Self:
        repo_container = RepoContainer.from_env_setup(config, env_setup)
        return cls(repo_container=repo_container)

    def destroy(self) -> None:
        # TODO: Delet the tmp repo copy
        logger.debug("Cleean up DockerContext")
        self.repo_container.destroy()


class DockerOrchTask(Task[DockerContext]):
    """
    Orchastrate docker containers used across pipeline. This includes
    - Spin up repo container (with repograph server)
    """

    env_setup_task: Task[EnvSetup]
    context: DockerContext

    def __init__(self, config: MigConfig, env_setup_task: Task[EnvSetup]):
        task_id = TaskId(f"docker_orch_{config.identifier}")
        super().__init__(task_id, config, [env_setup_task])
        self.env_setup_task = env_setup_task

    def should_run(self) -> bool:
        return True

    def load_cached_result(self) -> DockerContext:
        """Load existing repo info."""
        raise NotImplementedError(
            "load cached result for repo container task is not implemented"
        )

    def run(self) -> DockerContext:
        """Download the repo."""
        env_setup = self.get_dep_output(self.env_setup_task)
        self.logger.info(f"Spin up container from env setup:\n{pformat(env_setup)}")
        self.repo_container = RepoContainer.from_env_setup(self.config, env_setup)
        self.context = DockerContext.from_env_setup(self.config, env_setup)
        self.context.start()
        return self.context

    def cleanup(self) -> None:
        # TODO: Tear down repo container
        self.context.destroy()
        return super().cleanup()
