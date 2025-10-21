from .config import MigConfig
from pymigbench.migration import Migration as PMBMigration
import os

class PyMigBenchConfig(MigConfig):
    def __init__(self, base_path: str, mig: PMBMigration):
        repo_prefix = mig.repo.replace("/", "_")
        commit_sha = mig.commit
        repo_folder = f"{repo_prefix}__{commit_sha}"

        self.eval_tests_path = os.path.join(base_path, "eval_tests", repo_folder)
        # TODO: Finish the rest
