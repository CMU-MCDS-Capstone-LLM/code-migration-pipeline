# infra/tasks/patch_cmp_eval.py
from __future__ import annotations

import json

from evaluator import PatchSimilarityEvaluator

from ..pipeline.config import MigConfig
from ..pipeline.task import Task, TaskId
from ..models import Score, Repo
from ..tasks.repo_dl import RepoDlTask
from ..tasks.coding_agent import CodingAgentTask


class PatchCmpEvalTask(Task[Score]):
    """Evaluate patch similarity using PatchSimilarityEvaluator."""

    evaluator: PatchSimilarityEvaluator

    def __init__(
        self,
        config: MigConfig,
        repo_dl: RepoDlTask,
        coding_agent: CodingAgentTask,
    ) -> None:
        task_id = TaskId(f"patch_cmp_eval_{config.identifier}")
        # Declare task dependencies explicitly
        super().__init__(task_id, config, depends_on=[repo_dl, coding_agent])

        # Store typed refs for convenience inside run()
        self.repo_dl = repo_dl
        self.coding_agent = coding_agent
        self.evaluator = PatchSimilarityEvaluator(config)

    def should_run(self) -> bool:
        """Run if no score exists or needs updating."""
        score_file = self.config.score_path / "patch_similarity.json"
        return not score_file.exists()

    def load_cached_result(self) -> Score:
        """Load existing score from file."""
        score_file = self.config.score_path / "patch_similarity.json"
        if score_file.exists():
            with open(score_file) as f:
                data = json.load(f)
            return Score(
                value=data.get("similarity_score", 0.0),
                metadata=data,
            )
        return Score(value=0.0, metadata={})

    def run(self) -> Score:
        """Evaluate patch similarity."""

        # Typed access to dependencies (if/when you need them)
        repo: Repo = self.get_dep_output(self.repo_dl)
        trajectory_path: str = self.get_dep_output(self.coding_agent)

        # Currently your evaluator only uses config; repo/trajectory are available for
        # future enhancements (e.g. passing explicit paths).
        result = self.evaluator.evaluate()

        if result.get("status") == "success":
            similarity_score = result.get("similarity_score", 0.0)
        else:
            similarity_score = 0.0

        # Optionally persist the result
        score_file = self.config.score_path / "patch_similarity.json"
        score_file.parent.mkdir(parents=True, exist_ok=True)
        with open(score_file, "w") as f:
            json.dump(result, f, indent=2)

        return Score(
            value=similarity_score,
            metadata=result,
        )
