from dataclasses import dataclass
from pymigbench_dl.const.git import DEFAULT_PRE_MIG_BRANCH_NAME, DEFAULT_GT_PATCH_BRANCH_NAME
from pathlib import Path

@dataclass
class Repo:
    path: Path
    commit_hash: str
    pre_mig_branch: str = DEFAULT_PRE_MIG_BRANCH_NAME
    gt_branch: str = DEFAULT_GT_PATCH_BRANCH_NAME
