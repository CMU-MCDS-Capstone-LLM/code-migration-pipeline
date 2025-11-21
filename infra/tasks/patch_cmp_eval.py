import json
from pathlib import Path
from typing import Collection, Optional
from pymigbench_dl.providers.github.models import CommitInfo
from pymigbench_dl import PyMigBenchDownloader
from evaluator import PatchSimilarityEvaluator


from ..pipeline.config import MigConfig
from ..pipeline.task import Task, TaskId
from ..models import Score
from ..const.config import REPO_YAMLS_BASE_FOLDER


class PatchCmpEvalTask(Task):
    """Evaluate patch similarity using PatchSimilarityEvaluator"""

    evaluator: PatchSimilarityEvaluator

    def __init__(self, config: MigConfig, depends_on: Collection[TaskId]):
        task_id = TaskId(f"patch_cmp_eval_{config.identifier}")
        super().__init__(task_id, config, depends_on)

        # Find mig-diff.yaml in repo-yamls folder
        mig_diff_yaml_path = self._find_mig_diff_yaml(config)
        self.evaluator = PatchSimilarityEvaluator(
            config, mig_diff_yaml_path=mig_diff_yaml_path
        )

    def _find_mig_diff_yaml(self, config: MigConfig) -> Optional[Path]:
        """Find mig-diff.yaml file in repo-yamls folder using folder_name."""
        # Get base_dir from repo_path (assumes structure: base_dir/repos/{identifier})
        repo_path = Path(config.repo_path)
        if "repos" in str(repo_path):
            base_dir = repo_path.parent.parent
            folder_name = config.commit_info.folder_name
            repo_yamls_dir = base_dir / REPO_YAMLS_BASE_FOLDER / folder_name

            if repo_yamls_dir.exists():
                yaml_files = list(repo_yamls_dir.glob("*.yaml")) + list(
                    repo_yamls_dir.glob("*.yml")
                )
                if yaml_files:
                    return yaml_files[0]
        return None

    def should_run(self) -> bool:
        """Run if no score exists or needs updating"""
        score_file = self.config.score_path / "patch_similarity.json"
        return not score_file.exists()

    def load_cached_result(self) -> Score:
        """Load existing score from file"""
        score_file = self.config.score_path / "patch_similarity.json"
        if score_file.exists():
            with open(score_file) as f:
                data = json.load(f)
            return Score(value=data.get("similarity_score", 0.0), metadata=data)
        return Score(value=0.0, metadata={})

    def run(self) -> Score:
        """Evaluate patch similarity"""
        result = self.evaluator.evaluate()

        if result.get("status") == "success":
            similarity_score = result.get("similarity_score", 0.0)
        else:
            similarity_score = 0.0

        score = Score(value=similarity_score, metadata=result)

        # Write score to file
        score_file = self.config.score_path / "patch_similarity.json"
        score_data = {"similarity_score": score.value, **score.metadata}

        score_file.write_text(json.dumps(score_data, indent=2))

        if not score_file.exists():
            raise RuntimeError(f"Failed to write score file to {score_file}")

        return score
