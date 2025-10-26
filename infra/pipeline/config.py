from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pymigbench_dl.const.git import DEFAULT_GT_PATCH_BRANCH_NAME
from pymigbench_dl.loader import CommitInfo

from ..const.config import (
    ENV_BASE_FOLDER,
    EVAL_TESTS_BASE_FOLDER,
    HELPER_TESTS_BASE_FOLDER,
    REPO_BASE_FOLDER,
    SCORE_BASE_FOLDER,
    TRAJECTORY_BASE_FOLDER,
)

@dataclass(slots=True)
class MigConfig:
    """
    Concrete per-datapoint configuration. Shared across all components of the migration system.

    Note that, on init, this config will 
    - resolve all paths (must be folders) to absolute path, and create them if not exists
    """

    commit_info: CommitInfo
    env_path: Path
    repo_path: Path
    eval_tests_path: Path
    helper_tests_path: Path
    score_path: Path
    trajectory_path: Path
    post_migration_branch: str = DEFAULT_GT_PATCH_BRANCH_NAME

    def __post_init__(self) -> None:
        # Convert to absolute path
        self.env_path = self.env_path.resolve()
        self.repo_path = self.repo_path.resolve()
        self.eval_tests_path = self.eval_tests_path.resolve()
        self.helper_tests_path = self.helper_tests_path.resolve()
        self.score_path = self.score_path.resolve()
        self.trajectory_path = self.trajectory_path.resolve()

        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        for path in (
            self.env_path,
            self.eval_tests_path,
            self.helper_tests_path,
            self.repo_path,
            self.score_path,
            self.trajectory_path,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_base_dir_and_commit(cls, base_dir: Path, commit_info: CommitInfo) -> Self:
        """
        Standard layout builder using shared infra constants. Subclasses (or callers)
        can reuse this to avoid duplicating path-join logic.
        """
        base_dir = base_dir.resolve()
        identifier = commit_info.folder_name
        return cls(
            env_path=base_dir / ENV_BASE_FOLDER / identifier,
            eval_tests_path=base_dir / EVAL_TESTS_BASE_FOLDER / identifier,
            helper_tests_path=base_dir / HELPER_TESTS_BASE_FOLDER / identifier,
            repo_path=base_dir / REPO_BASE_FOLDER / identifier,
            score_path=base_dir / SCORE_BASE_FOLDER / identifier,
            trajectory_path=base_dir / TRAJECTORY_BASE_FOLDER / identifier,
            commit_info=commit_info
        )

    @property
    def identifier(self) -> str:
        return str(self.commit_info)
