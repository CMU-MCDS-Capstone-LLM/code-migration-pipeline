import os
from typing import Collection 
from pymigbench_dl.providers.github.models import CommitInfo
from pymigbench_dl import PyMigBenchDownloader
from pymigbench_dl.const.git import DEFAULT_PRE_MIG_BRANCH_NAME, DEFAULT_GT_PATCH_BRANCH_NAME

from ..pipeline.config import MigConfig
from ..pipeline.task import Task, TaskId
from ..const import GITHUB_TOKEN_NAME
from ..models import Repo

class RepoDlTask(Task):
    """Download repository from PyMigBench"""
    
    def __init__(self, config: MigConfig, depends_on: Collection[TaskId]):
        task_id = TaskId(f"repo_dl_{config.identifier}")
        super().__init__(task_id, config, depends_on)
        self.downloader = PyMigBenchDownloader(
            os.getenv(GITHUB_TOKEN_NAME),
            str(self.config.repo_path.parent)
        )
    
    def should_run(self) -> bool:
        """Run only if repo doesn't exist"""
        return self.downloader.has_downloaded(self.config.commit_info)
    
    def load_cached_result(self) -> Repo:
        """Load existing repo info"""
        return Repo(path=self.config.repo_path, commit_hash=self.config.commit_info.commit_sha)
    
    def run(self) -> Repo:
        """Download the repo"""
        self.downloader.download_single_from_commit_info(
            self.config.commit_info, 
            DEFAULT_PRE_MIG_BRANCH_NAME,
            DEFAULT_GT_PATCH_BRANCH_NAME
        )
        return Repo(path=self.config.repo_path, commit_hash=self.config.commit_info.commit_sha)

