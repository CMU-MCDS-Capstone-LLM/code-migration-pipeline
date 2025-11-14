# infra/main.py
from pathlib import Path

from pymigbench.database import Database
from pymigbench.migration import Migration
from pymigbench_dl.utils.repo import CommitInfo

from infra.pipeline.config import MigConfig
from infra.tasks.env_setup_agent import EnvSetupAgentTask
from infra.tasks.repo_container import RepoContainerTask
from infra.utils.log import setup_logging
from .pipeline.dag import DAGExecutor
from .tasks.repo_dl import RepoDlTask
from .tasks.pymigbench_config import PyMigBenchMigConfig

import logging

logger = logging.getLogger(__name__)


def build_pipeline(config):
    """Build task DAG."""

    repo_dl_task = RepoDlTask(config)
    env_setup_agent_task = EnvSetupAgentTask(config, repo_dl_task=repo_dl_task)
    repo_container_task = RepoContainerTask(config, env_setup_task=env_setup_agent_task)

    return [repo_dl_task, env_setup_agent_task, repo_container_task]


def build_config_from_mig(base_dir: Path, mig: Migration) -> MigConfig:
    commit = CommitInfo(mig.repo, mig.commit)
    config = PyMigBenchMigConfig.from_base_dir_and_commit(
        base_dir=base_dir,
        commit_info=commit,
        problem_statement=(
            f"Migrate from {mig.source} to {mig.target}. "
            "Update all API endpoints, middleware, and configuration to use "
            "slack-sdk instead of slackclient."
        ),
    )
    return config


def main():
    """Run migration pipeline for a single migration."""
    base_dir = Path(
        "/home/eiger/CMU/2025_Spring/11634_Capstone/codebase/small_data/sample_data"
    )
    repo_yamls_dir = base_dir / "repo-yamls"

    db = Database.load_from_dir(repo_yamls_dir)
    print(f"Loaded PyMigBench database from {repo_yamls_dir}")

    migs = db.migs()
    print(f"Found {len(migs)} migrations in {repo_yamls_dir}")

    for mig in migs[:1]:
        config = build_config_from_mig(base_dir, mig)
        log_file_path = (
            base_dir
            / "logs"
            / config.commit_info.folder_name
            / "code-migration-pipeline.log"
        ).resolve()
        setup_logging(log_file=log_file_path, level=logging.DEBUG)
        logger.info("foo")
        logger.debug("bar")
        logger.error("what?")

        tasks = build_pipeline(config)
        executor = DAGExecutor(tasks)
        executor.execute()


if __name__ == "__main__":
    main()
