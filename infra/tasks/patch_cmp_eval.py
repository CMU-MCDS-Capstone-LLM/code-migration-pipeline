import os
from typing import Collection
from pymigbench_dl.providers.github.models import CommitInfo
from pymigbench_dl import PyMigBenchDownloader
from evaluator import PatchSimilarityEvaluator


from ..pipeline.config import MigConfig
from ..pipeline.task import Task, TaskId
from ..models import Score

class PatchCmpEvalTask(Task):
    """Evaluate patch similarity using PatchSimilarityEvaluator"""
    evaluator: PatchSimilarityEvaluator
    
    def __init__(self, config: MigConfig, depends_on: Collection[TaskId]):
        task_id = TaskId(f"patch_cmp_eval_{config.identifier}")
        super().__init__(task_id, config, depends_on)
        self.evaluator = PatchSimilarityEvaluator(config)
    
    def should_run(self) -> bool:
        """Run if no score exists or needs updating"""
        score_file = self.config.score_path / "patch_similarity.json"
        return not score_file.exists()
    
    def load_cached_result(self) -> Score:
        """Load existing score from file"""
        score_file = self.config.score_path / "patch_similarity.json"
        if score_file.exists():
            import json
            with open(score_file) as f:
                data = json.load(f)
            return Score(
                value=data.get("similarity_score", 0.0),
                metadata=data
            )
        return Score(value=0.0, metadata={})
    
    def run(self) -> Score:
        """Evaluate patch similarity"""
        result = self.evaluator.evaluate()
        
        if result.get("status") == "success":
            similarity_score = result.get("similarity_score", 0.0)
        else:
            # Handle error case
            similarity_score = 0.0
            
        return Score(
            value=similarity_score,
            metadata=result
        )
