from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pymigbench.migration import Migration as PMBMigration
from pymigbench_dl.providers.github.models import CommitInfo

from ..pipeline.config import MigConfig

@dataclass(slots=True)
class PyMigBenchMigConfig(MigConfig):
    """MigConfig specialization for PyMigBench. No extra fields needed."""

    @classmethod
    def from_mig(cls, base_dir: Path, mig: PMBMigration) -> Self:
        # Choose the folder name PyMigBench DL would use for the repo.
        commit_info = CommitInfo.from_mig(mig)
        identifier = commit_info.folder_name

        # Build the standard layout and return an instance of *this* subclass.
        return cls.from_base_dir_and_identifier(base_dir, identifier, commit_info)
