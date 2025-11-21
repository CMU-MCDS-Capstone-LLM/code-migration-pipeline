from pathlib import Path
import yaml

from pymigbench_dl.utils.repo import CommitInfo
from .pipeline.dag import DAGExecutor
from .tasks.repo_dl import RepoDlTask
from .tasks.docker_container import DockerContainerTask
from .tasks.coding_agent import CodingAgentTask
from .tasks.patch_cmp_eval import PatchCmpEvalTask
from .tasks.pymigbench_config import PyMigBenchMigConfig
from .const.config import REPO_YAMLS_BASE_FOLDER


def discover_repos(base_dir: Path) -> list[dict]:
    """Discover all repos from repo-yamls directory"""
    repo_yamls_dir = base_dir / REPO_YAMLS_BASE_FOLDER
    repos = []

    if not repo_yamls_dir.exists():
        return repos

    # Iterate through each subdirectory in repo-yamls
    for repo_dir in repo_yamls_dir.iterdir():
        if not repo_dir.is_dir():
            continue

        # Find YAML files in this directory
        yaml_files = list(repo_dir.glob("*.yaml")) + list(repo_dir.glob("*.yml"))
        if not yaml_files:
            continue

        # Parse the first YAML file
        yaml_file = yaml_files[0]
        try:
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f)

            if "repo" in data and "commit" in data:
                repos.append(
                    {
                        "repo": data["repo"],
                        "commit_sha": data["commit"],
                        "source": data.get("source", ""),
                        "target": data.get("target", ""),
                        "yaml_path": yaml_file,
                    }
                )
        except Exception as e:
            print(f"Error parsing {yaml_file}: {e}")
            continue

    return repos


def generate_problem_statement(source: str, target: str) -> str:
    """Generate a problem statement from source and target libraries"""
    if source and target:
        return f"Migrate from {source} to {target} framework. Update all imports, API calls, and configuration to use {target} instead of {source}."
    return ""


def run_single_repo(config: PyMigBenchMigConfig):
    """Run pipeline for a single repo"""
    print(f"\n{'='*80}")
    print(
        f"Running pipeline for: {config.commit_info.repo} ({config.commit_info.commit_sha[:8]})"
    )
    print(f"{'='*80}\n")

    tasks = build_pipeline(config)
    executor = DAGExecutor(tasks)
    executor.execute()

    print(f"\n✓ Completed: {config.commit_info.repo}\n")


def main():
    """Run migration pipeline for all discovered repos"""
    project_root = Path(__file__).parent.parent
    base_dir = project_root / "data"

    # Discover all repos
    repos = discover_repos(base_dir)

    if not repos:
        print("No repos found in repo-yamls directory")
        return

    print(f"Found {len(repos)} repo(s) to process:")
    for i, repo_info in enumerate(repos, 1):
        print(f"  {i}. {repo_info['repo']} ({repo_info['commit_sha'][:8]})")

    # Run pipeline for each repo sequentially
    for repo_info in repos:
        try:
            problem_statement = generate_problem_statement(
                repo_info["source"], repo_info["target"]
            )

            config = PyMigBenchMigConfig.from_base_dir_and_commit(
                base_dir=base_dir,
                commit_info=CommitInfo(
                    repo=repo_info["repo"],
                    commit_sha=repo_info["commit_sha"],
                ),
                problem_statement=problem_statement,
            )

            run_single_repo(config)

        except Exception as e:
            print(f"\n✗ Error processing {repo_info['repo']}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"All repos processed: {len(repos)} total")
    print(f"{'='*80}\n")


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
