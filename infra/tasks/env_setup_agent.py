# infra/tasks/env_setup_agent.py
from dataclasses import dataclass
from enum import Enum
import json
from logging import getLogger
from pathlib import Path
from typing import Self
from env_setup_agent.consts import (
    DECISION_JSON_FILENAME,
    DEFAULT_DOCKER_BUILD_SCRIPT_FILENAME,
    DEFAULT_DOCKER_RUN_SCRIPT_FILENAME,
    DEFAULT_DOCKERFILE_FILENAME,
    FAILURE_FLAG_FILENAME,
    SUCCESS_FLAG_FILENAME,
)
from env_setup_agent.core import DecisionStatus, DockerVars
from env_setup_agent.agent.claude_runner import map_decision
from typing_extensions import override
from infra.models import Repo
from infra.pipeline.config import MigConfig
from infra.utils.path import check_path
from ..pipeline.task import Task, TaskId

logger = getLogger(__name__)


class EnvSetupStatus(Enum):
    """Status of the env setup. Determined by the success / failure flag file"""

    SUCCESS = 0
    FAILURE = 1

    @classmethod
    def from_env_path(cls, env_path: Path) -> "EnvSetupStatus":
        success_flag = (env_path / SUCCESS_FLAG_FILENAME).exists()
        failure_flag = (env_path / FAILURE_FLAG_FILENAME).exists()
        if success_flag and failure_flag:
            raise RuntimeError(
                f"Can't identify env setup status because both success and failure flag exist in env path {env_path}."
            )
        if success_flag:
            return cls.SUCCESS
        if failure_flag:
            return cls.FAILURE
        raise RuntimeError(
            f"Can't identify env setup status because neither success nor failure flag exists in env path {env_path}."
        )


@dataclass
class EnvSetup:
    env_path: Path
    variables: DockerVars
    dockerfile_path: Path
    build_script_path: Path
    run_script_path: Path

    @classmethod
    def from_env_path(cls, env_path: Path) -> Self:
        try:
            env_setup_status = EnvSetupStatus.from_env_path(env_path)
        except Exception as e:
            logger.error(f"Failed to check env setup status. Got error: {e}")
            raise e
        assert env_setup_status != EnvSetupStatus.FAILURE, (
            f"Can't proceed because a failed env setup is present in env path {env_path}"
        )

        decision_json_path = check_path(env_path / DECISION_JSON_FILENAME)
        dockerfile_path = check_path(env_path / DEFAULT_DOCKERFILE_FILENAME)
        run_script_path = check_path(env_path / DEFAULT_DOCKER_RUN_SCRIPT_FILENAME)
        build_script_path = check_path(env_path / DEFAULT_DOCKER_BUILD_SCRIPT_FILENAME)

        with open(decision_json_path, "r") as f:
            decision_json_dict = json.load(f)
        decision_json = map_decision(decision_json_dict)
        assert decision_json.status == DecisionStatus.PROCEED, (
            "Can't proceed because the cached decision json is a failed decision."
        )
        assert decision_json.variables is not None, (
            "Can't proceed because a success decision doesn't have variables"
        )
        return cls(
            env_path=env_path,
            variables=decision_json.variables,
            dockerfile_path=dockerfile_path,
            build_script_path=build_script_path,
            run_script_path=run_script_path,
        )


class EnvSetupAgentTask(Task[EnvSetup]):
    repo_dl_task: Task[Repo]

    def __init__(self, config: MigConfig, repo_dl_task: Task[Repo]) -> None:
        task_id = TaskId(f"env_setup_agent_{config.identifier}")
        super().__init__(task_id, config, [repo_dl_task])
        self.repo_dl_task = repo_dl_task

    def should_run(self) -> bool:
        env_path = self.config.env_path
        if not env_path.exists():
            return True
        try:
            env_setup_status = EnvSetupStatus.from_env_path(env_path)
        except Exception as e:
            logger.error(f"Failed to check env setup status. Got error: {e}")
            raise e
        assert env_setup_status != EnvSetupStatus.FAILURE, (
            f"Can't proceed because a failed env setup is present in env path {env_path}"
        )
        return False

    def load_cached_result(self) -> EnvSetup:
        return EnvSetup.from_env_path(self.config.env_path)

    def run(self) -> EnvSetup:
        """ """
        repo = self.get_dep_output(self.repo_dl_task)
        self.logger.info(f"Set up env for repo at path {repo.path}")
        raise NotImplementedError("run() for env setup agent has not been implemented!")
        # TODO: Determine env setup config path
        # env_setup_config_path = Path("???").resolve()
        # # TODO: Create env setup config at path
        # asyncio.run(run_from_config(env_setup_config_path))
        # _setup_repo_container()

    @override
    def cleanup(self) -> None:
        self.logger.debug("Destroy the repo container")
        return super().cleanup()
