from pathlib import Path

from pymigbench_dl.utils.repo import CommitInfo
from .pipeline.dag import DAGExecutor
from .tasks.repo_dl import RepoDlTask
from .tasks.coding_agent import CodingAgentTask
from .tasks.patch_cmp_eval import PatchCmpEvalTask
from .tasks.pymigbench_config import PyMigBenchMigConfig


def main():
    """Run migration pipeline for a single migration"""

    config = PyMigBenchMigConfig.from_base_dir_and_commit(
        base_dir=Path("/home/eiger/CMU/2025_Spring/11634_Capstone/codebase/tiny_data"),
        commit_info=CommitInfo(
            repo="adithyabsk/keep2roam", 
            commit_sha="d340eea2fdedde8908334eda34325d058fc88282"
        ),
    )

    # Build task list
    tasks = build_pipeline(config)

    # Execute
    executor = DAGExecutor(tasks)
    executor.execute()

def build_pipeline(config):
    """Build task DAG"""
    
    # Create all tasks
    repo_dl = RepoDlTask(config, depends_on=[])

    coding_agent = CodingAgentTask(
        config, 
        depends_on=[repo_dl.task_id]
    )
    
    patch_eval = PatchCmpEvalTask(config, depends_on=[repo_dl.task_id, coding_agent.task_id])
    
    return [repo_dl, coding_agent, patch_eval]

if __name__ == '__main__':
    main()

