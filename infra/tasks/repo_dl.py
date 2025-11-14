# infra/tasks/repo_dl.py
import os

from pymigbench_dl import PyMigBenchDownloader
from pymigbench_dl.const.git import (
    DEFAULT_PRE_MIG_BRANCH_NAME,
    DEFAULT_GT_PATCH_BRANCH_NAME,
)

from ..pipeline.config import MigConfig
from ..pipeline.task import Task, TaskId
from ..const import GITHUB_TOKEN_NAME
from ..models import Repo


class RepoDlTask(Task[Repo]):
    """Download repository from PyMigBench."""

    def __init__(self, config: MigConfig) -> None:
        task_id = TaskId(f"repo_dl_{config.identifier}")
        # No dependencies for this leaf task
        super().__init__(task_id, config)
        self.downloader = PyMigBenchDownloader(
            os.getenv(GITHUB_TOKEN_NAME), str(self.config.repo_path.parent)
        )

    def should_run(self) -> bool:
        """Run only if repo doesn't exist."""
        # NOTE: if has_downloaded == True means it's already downloaded,
        # so we invert it to check if we should run
        return not self.downloader.has_downloaded(self.config.commit_info)

    def load_cached_result(self) -> Repo:
        """Load existing repo info."""
        repo_path = self.config.repo_path
        assert repo_path.exists() and (repo_path / ".git").exists(), (
            f"Attempt to load cached result but got an invalid repo at path {repo_path}"
        )
        return Repo(
            path=self.config.repo_path,
            commit_hash=self.config.commit_info.commit_sha,
        )

    def run(self) -> Repo:
        """Download the repo."""
        self.downloader.download_single_from_commit_info(
            self.config.commit_info,
            DEFAULT_PRE_MIG_BRANCH_NAME,
            DEFAULT_GT_PATCH_BRANCH_NAME,
        )
        return Repo(
            path=self.config.repo_path,
            commit_hash=self.config.commit_info.commit_sha,
        )
