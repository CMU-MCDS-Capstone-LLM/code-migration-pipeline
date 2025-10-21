from typing import Protocol

class MigConfig(Protocol):
    """
    Protocol for per-datapoint configuration. Shared across all components of the migration system.

    Concrete way to construct this config is dependent on the benchmark setting. For example, for PyMigBench, the construction takes a `pymigbench.Migraition` object as input, and the envs path is `envs/<repo_name>__<commit_sha>`
    """
    eval_tests_path: str
    input_tests_path: str
    repo_path: str
    score_path: str
    trajectory_path: str
