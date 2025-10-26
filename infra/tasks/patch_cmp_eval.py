import os
from typing import Collection
from pymigbench_dl.providers.github.models import CommitInfo
from pymigbench_dl import PyMigBenchDownloader
from evaluator import PatchSimilarityEvaluator


from ..pipeline.config import MigConfig
from ..pipeline.task import Task, TaskId
from ..models import Score

class PatchCmpEvalTask(Task):
    """Download repository from PyMigBench"""
    evaluator: PatchSimilarityEvaluator
    
    def __init__(self, config: MigConfig, depends_on: Collection[TaskId]):
        task_id = TaskId(f"patch_cmp_eval_{config.identifier}")
        super().__init__(task_id, config, depends_on)
        self.evaluator = PatchSimilarityEvaluator(config)
    
    def should_run(self) -> bool:
        """Run only if repo doesn't exist"""
        return True
    
    def load_cached_result(self) -> Score:
        """Load existing repo info"""
        raise NotImplementedError("Load cached result for patch cmp eval is not implemented")
    
    def run(self) -> Score:
        """Download the repo"""
        score = self.evaluator.evaluate()
        return Score(
            value=score["similarity_score"],
            metadata=score
        )
