from pathlib import Path

from pymigbench_dl.utils.repo import CommitInfo
from .pipeline.dag import DAGExecutor
from .tasks.repo_dl import RepoDlTask
from .tasks.docker_container import DockerContainerTask
from .tasks.coding_agent import CodingAgentTask
from .tasks.patch_cmp_eval import PatchCmpEvalTask
from .tasks.pymigbench_config import PyMigBenchMigConfig


def main():
    """Run migration pipeline for a single migration"""

    # config = PyMigBenchMigConfig.from_base_dir_and_commit(
    #     base_dir=Path("/home/eiger/CMU/2025_Spring/11634_Capstone/codebase/tiny_data"),
    #     commit_info=CommitInfo(
    #         repo="adithyabsk/keep2roam",
    #         commit_sha="d340eea2fdedde8908334eda34325d058fc88282",
    #     ),
    # )
    config = PyMigBenchMigConfig.from_base_dir_and_commit(
        base_dir=Path("./data"),
        commit_info=CommitInfo(
            repo="alice-biometrics/petisco",
            commit_sha="9abf7b1f6ef8c55bdddcb9a5c2eff513f6a93130",
        ),
        problem_statement="Migrate from slackclient to slack-sdk framework. Update all API endpoints, middleware, and configuration to use slack-sdk instead of slackclient.",
    )

    # Build task list
    tasks = build_pipeline(config)

    # Execute
    executor = DAGExecutor(tasks)
    executor.execute()


def build_pipeline(config):
    """Build task DAG"""

    repo_dl = RepoDlTask(config, depends_on=[])

    docker_container = DockerContainerTask(config, depends_on=[repo_dl.task_id])

    coding_agent = CodingAgentTask(config, depends_on=[docker_container.task_id])

    patch_eval = PatchCmpEvalTask(
        config, depends_on=[repo_dl.task_id, coding_agent.task_id]
    )

    return [repo_dl, docker_container, coding_agent, patch_eval]


if __name__ == "__main__":
    main()
