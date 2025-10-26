# AI-Based Code Migration Pipeline: Complete Design Summary

## Context

We're building an infrastructure for automated code migration using AI agents. The system:

- Takes Python repositories and migrates them from one library version to another (e.g., urllib3 1.x → 2.x)
- Uses two git submodules: `coding-agent` (generates migration patches) and `testing-agent` (generates tests)
- Processes repositories from benchmarks like PyMigBench
- Needs to cache intermediate results to avoid redundant computation
- Stores all artifacts in a structured data directory

The key challenge is orchestrating multiple containerized agents while efficiently caching results.

---

## Abstract Design

### Core Concepts

**1. Task-Based DAG Pipeline**

- Each operation (download repo, generate tests, run migration, evaluate) is a `Task`
- Tasks declare dependencies, forming a DAG (Directed Acyclic Graph)
- A `DAGExecutor` runs tasks in topological order with optional parallelization
- Tasks can run in parallel when they have no dependencies on each other

**2. Per-Task Caching Strategy**

- Each task implements `should_run()` to decide if it needs execution
- Each task implements `load_cached_result()` to load previous results
- Caching logic is task-specific and file-based (no complex hash database)
- Tasks check if output files exist and whether inputs have changed

**3. Shared Configuration**

- A `MigConfig` protocol defines all file paths for a migration
- Concrete `MigConfigImpl` constructs paths based on repo name and commit
- All tasks share the same config instance
- Paths follow pattern: `data/{category}/{repo_name}__{commit_sha}/`

**4. Agent Isolation**

- Git submodules (`coding-agent`, `testing-agent`) are accessed only through wrapper classes
- Wrappers handle CLI invocation, environment variables, and YAML configs
- Containers run agents with mounted volumes from the data directory

### Pipeline Flow

```
RepoDl → EnvSetup → HelperTestAgent → CodingAgent → PatchEval
                  ↘                                 ↗
                    EvalTestAgent ----------------→ UnitTestEval
```

**Execution Levels (for parallelization):**

- Level 0: RepoDl
- Level 1: EnvSetup
- Level 2: HelperTestAgent, EvalTestAgent (parallel)
- Level 3: CodingAgent
- Level 4: PatchEval, UnitTestEval (parallel)

---

## Folder Structure

```
.
├── coding-agent/              # Git submodule - coding AI agent
├── testing-agent/             # Git submodule - testing AI agent
├── data/                      # Runtime data (gitignored)
│   ├── envs/                  # Docker environments per repo
│   ├── eval-tests/            # Generated evaluation tests
│   ├── input-tests/           # Generated helper tests
│   ├── repos/                 # Cloned repositories
│   ├── repo-yamls/            # Migration YAML configs
│   ├── score/                 # Evaluation scores
│   └── trajectories/          # Generated patches and logs
├── config/                    # User-facing configs
│   ├── pipeline_config.yaml   # Global pipeline settings
│   └── repos/                 # Per-repo configs (optional)
│       └── requests.yaml
├── infra/                     # Core infrastructure code
│   ├── __init__.py
│   ├── pipeline/              # Pipeline orchestration
│   │   ├── __init__.py
│   │   ├── config.py          # MigConfig protocol & implementation
│   │   ├── dag.py             # DAGExecutor
│   │   ├── task.py            # Task base class
│   │   └── context.py         # PipelineContext
│   ├── tasks/                 # Concrete task implementations
│   │   ├── __init__.py
│   │   ├── repo.py            # RepoDlTask
│   │   ├── env.py             # EnvSetupAgentTask
│   │   ├── test_gen.py        # HelperTestAgentTask, EvalTestAgentTask
│   │   ├── coding.py          # CodingAgentTask
│   │   └── eval.py            # PatchCmpEvalTask, UnitTestEvalTask
│   ├── runners/               # Container & agent runners
│   │   ├── __init__.py
│   │   ├── container.py       # ContainerRunner (Docker SDK wrapper)
│   │   ├── coding_agent.py    # CodingAgentRunner (CLI wrapper)
│   │   └── testing_agent.py   # TestingAgentRunner (CLI wrapper)
│   ├── models/                # Data models
│   │   ├── __init__.py
│   │   ├── artifacts.py       # Repo, Patch, TestSuite, Dockerfile, Score
│   │   └── migration.py       # MigrationInfo
│   ├── utils/                 # Utilities
│   │   ├── __init__.py
│   │   ├── git.py             # Git operations
│   │   └── docker.py          # Docker utilities
│   ├── benchmarks/            # Benchmark-specific adapters
│   │   ├── __init__.py
│   │   ├── base.py            # Base Benchmark interface
│   │   └── pymigbench.py      # PyMigBench adapter
│   └── cli.py                 # CLI entry point
├── scripts/                   # Helper scripts
│   ├── setup_env.sh
│   └── run_batch.py
├── tests/                     # Unit tests
│   ├── test_pipeline/
│   ├── test_tasks/
│   └── test_runners/
├── notes/                     # Project notes
├── environment.yml            # Conda environment
├── pyproject.toml            # Python package config
├── README.md
└── .gitignore
```

---

## Concrete Code

### 1. Configuration (`infra/pipeline/config.py`)

```python
from pathlib import Path
from typing import Protocol
from dataclasses import dataclass

class MigConfig(Protocol):
    """
    Protocol for per-datapoint configuration. 
    Shared across all components of the migration system.
    """
    yaml_root_path: Path
    eval_tests_path: Path
    helper_tests_path: Path
    repo_path: Path
    env_path: Path
    score_path: Path
    trajectory_path: Path

@dataclass
class MigConfigImpl:
    """Concrete implementation of MigConfig"""
    yaml_root_path: Path
    eval_tests_path: Path
    helper_tests_path: Path
    repo_path: Path
    env_path: Path
    score_path: Path
    trajectory_path: Path
    
    @classmethod
    def from_base_dir(cls, base_dir: Path, repo_name: str, 
                      commit_sha: str) -> 'MigConfigImpl':
        """
        Factory method to construct config from base directory.
        
        Example:
            config = MigConfigImpl.from_base_dir(
                Path("data"), "requests", "abc123"
            )
        """
        identifier = f"{repo_name}__{commit_sha}"
        
        return cls(
            yaml_root_path=base_dir / "repo-yamls" / repo_name,
            eval_tests_path=base_dir / "eval-tests" / identifier,
            helper_tests_path=base_dir / "input-tests" / identifier,
            repo_path=base_dir / "repos" / identifier,
            env_path=base_dir / "envs" / identifier,
            score_path=base_dir / "score" / identifier,
            trajectory_path=base_dir / "trajectories" / identifier
        )
    
    @classmethod
    def from_pymigbench(cls, migration: 'pymigbench.Migration', 
                        base_dir: Path) -> 'MigConfigImpl':
        """Construct from PyMigBench migration object"""
        return cls.from_base_dir(
            base_dir,
            repo_name=migration.repo_name,
            commit_sha=migration.commit_sha
        )
    
    def ensure_directories(self):
        """Create all necessary directories"""
        for path in [
            self.yaml_root_path,
            self.eval_tests_path,
            self.helper_tests_path,
            self.repo_path.parent,
            self.env_path,
            self.score_path,
            self.trajectory_path
        ]:
            path.mkdir(parents=True, exist_ok=True)
```

### 2. Data Models (`infra/models/artifacts.py`)

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Repo:
    path: Path
    commit_hash: str
    main_branch: str = "main"
    gt_branch: str = "gt-patch"

@dataclass
class Dockerfile:
    path: Path
    image_tag: str

@dataclass
class TestSuite:
    path: Path
    branch: str

@dataclass
class Patch:
    path: Path
    content: str

@dataclass
class Score:
    value: float
    metadata: dict
```

```python
# infra/models/migration.py
from dataclasses import dataclass

@dataclass
class MigrationInfo:
    lib_a: str
    lib_b: str
    version_a: str
    version_b: str
```

### 3. Task Base Class (`infra/pipeline/task.py`)

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from .config import MigConfig

class Task(ABC):
    """Base class where each task manages its own cache logic"""
    
    def __init__(self, task_id: str, config: MigConfig):
        self.task_id = task_id
        self.config = config
        self.inputs: Dict[str, Any] = {}
        self.output: Any = None
        self.status: str = "pending"
    
    @abstractmethod
    def run(self) -> Any:
        """Execute the task logic"""
        pass
    
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """Return list of task_ids this task depends on"""
        pass
    
    @abstractmethod
    def should_run(self) -> bool:
        """
        Check if task needs to execute or can reuse previous result.
        Return True to execute, False to skip and load cached result.
        """
        pass
    
    @abstractmethod
    def load_cached_result(self) -> Any:
        """Load and return the cached result"""
        pass
    
    def execute(self, context: 'PipelineContext') -> Any:
        """Wrapper that handles caching, logging, error handling"""
        
        # Check if we can skip execution
        if not self.should_run():
            print(f"✓ Reusing cached result for {self.task_id}")
            self.output = self.load_cached_result()
            self.status = "cached"
            return self.output
        
        # Execute task
        print(f"⚙ Running {self.task_id}...")
        self.status = "running"
        try:
            self.output = self.run()
            self.status = "completed"
            return self.output
        except Exception as e:
            self.status = "failed"
            print(f"✗ Failed {self.task_id}: {e}")
            raise
```

### 4. Pipeline Context (`infra/pipeline/context.py`)

```python
from typing import Dict, Any, List

class PipelineContext:
    """Shared context for all tasks"""
    
    def __init__(self):
        self.artifacts: Dict[str, Any] = {}
        self.logs: List[str] = []
    
    def store_artifact(self, task_id: str, artifact: Any):
        """Store task result"""
        self.artifacts[task_id] = artifact
    
    def get_artifact(self, task_id: str) -> Any:
        """Retrieve task result"""
        return self.artifacts.get(task_id)
```

### 5. DAG Executor (`infra/pipeline/dag.py`)

```python
from collections import defaultdict
from typing import Dict, List
from .task import Task
from .context import PipelineContext

class DAGExecutor:
    """Topological sort executor with parallel execution support"""
    
    def __init__(self, tasks: List[Task]):
        self.tasks = {task.task_id: task for task in tasks}
        self.context = PipelineContext()
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Returns adjacency list of dependencies"""
        graph = defaultdict(list)
        for task in self.tasks.values():
            for dep_id in task.get_dependencies():
                graph[dep_id].append(task.task_id)
        return graph
    
    def _topological_sort(self) -> List[List[str]]:
        """Returns list of execution levels (can run in parallel)"""
        graph = self._build_dependency_graph()
        in_degree = defaultdict(int)
        
        # Calculate in-degrees
        for task_id in self.tasks:
            in_degree[task_id] = len(self.tasks[task_id].get_dependencies())
        
        # Find tasks with no dependencies (level 0)
        levels = []
        current_level = [tid for tid, deg in in_degree.items() if deg == 0]
        
        while current_level:
            levels.append(current_level)
            next_level = []
            
            for task_id in current_level:
                for dependent_id in graph[task_id]:
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        next_level.append(dependent_id)
            
            current_level = next_level
        
        return levels
    
    def execute(self, parallel: bool = False):
        """Execute DAG level by level"""
        levels = self._topological_sort()
        
        for level_idx, level in enumerate(levels):
            print(f"\n📋 Level {level_idx}: {len(level)} task(s)")
            
            if parallel and len(level) > 1:
                # Use ThreadPoolExecutor for parallel execution
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=len(level)) as executor:
                    futures = {
                        executor.submit(self._execute_task, tid): tid 
                        for tid in level
                    }
                    for future in futures:
                        future.result()  # Wait and handle errors
            else:
                # Sequential execution
                for task_id in level:
                    self._execute_task(task_id)
    
    def _execute_task(self, task_id: str):
        """Execute a single task"""
        task = self.tasks[task_id]
        
        # Inject dependencies from context
        for dep_id in task.get_dependencies():
            dep_artifact = self.context.get_artifact(dep_id)
            task.inputs[dep_id] = dep_artifact
        
        # Execute
        result = task.execute(self.context)
        
        # Store result
        self.context.store_artifact(task_id, result)
```

### 6. Example Task Implementation (`infra/tasks/repo.py`)

```python
from pathlib import Path
from infra.pipeline.task import Task
from infra.pipeline.config import MigConfig
from infra.models.artifacts import Repo
import subprocess

class RepoDlTask(Task):
    """Download repository from PyMigBench"""
    
    def __init__(self, config: MigConfig, repo_name: str):
        super().__init__(f"repo_dl_{repo_name}", config)
        self.repo_name = repo_name
    
    def get_dependencies(self) -> list[str]:
        return []  # No dependencies
    
    def should_run(self) -> bool:
        """Run only if repo doesn't exist"""
        if not self.config.repo_path.exists():
            return True
        
        # Check if it's a valid git repo
        if not (self.config.repo_path / ".git").exists():
            return True
        
        return False
    
    def load_cached_result(self) -> Repo:
        """Load existing repo info"""
        # Get commit hash from git
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.config.repo_path,
            capture_output=True,
            text=True
        )
        commit_hash = result.stdout.strip()
        
        return Repo(path=self.config.repo_path, commit_hash=commit_hash)
    
    def run(self) -> Repo:
        """Download the repo"""
        self.config.repo_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Clone logic using pymigbench_dl or git
        print(f"Cloning {self.repo_name}...")
        # ... actual clone logic ...
        
        return Repo(path=self.config.repo_path, commit_hash="abc123")
```

### 7. Coding Agent Task (`infra/tasks/coding.py`)

```python
from pathlib import Path
from infra.pipeline.task import Task
from infra.pipeline.config import MigConfig
from infra.models.artifacts import Repo, Patch, TestSuite, Dockerfile
from infra.models.migration import MigrationInfo
from infra.runners.coding_agent import CodingAgentRunner

class CodingAgentTask(Task):
    """Run coding-agent to generate migration patch"""
    
    def __init__(self, config: MigConfig, repo: Repo, 
                 mig_info: MigrationInfo, helper_tests: TestSuite,
                 dockerfile: Dockerfile, agent_path: Path,
                 force_rerun: bool = False):
        super().__init__("coding_agent", config)
        self.repo = repo
        self.mig_info = mig_info
        self.helper_tests = helper_tests
        self.dockerfile = dockerfile
        self.agent_path = agent_path
        self.force_rerun = force_rerun
        self.patch_path = config.trajectory_path / "migration.patch"
    
    def get_dependencies(self) -> list[str]:
        return ["helper_test"]
    
    def should_run(self) -> bool:
        """Run if patch doesn't exist or force_rerun is True"""
        if self.force_rerun:
            return True
        return not self.patch_path.exists()
    
    def load_cached_result(self) -> Patch:
        """Load existing patch"""
        content = self.patch_path.read_text()
        return Patch(path=self.patch_path, content=content)
    
    def run(self) -> Patch:
        """Run coding-agent to generate patch"""
        self.config.trajectory_path.mkdir(parents=True, exist_ok=True)
        
        # Use coding agent runner
        runner = CodingAgentRunner(
            agent_path=self.agent_path,
            config_path=Path("config/coding_agent_config.yaml"),
            env_vars={
                "MODEL": "claude-sonnet-4",
                "TEMPERATURE": "0.7"
            }
        )
        
        runner.run(
            workspace=self.config.repo_path,
            output_dir=self.config.trajectory_path
        )
        
        content = self.patch_path.read_text()
        return Patch(path=self.patch_path, content=content)
```

### 8. Agent Runner (`infra/runners/coding_agent.py`)

```python
import subprocess
import os
from pathlib import Path
from typing import Dict, Optional

class CodingAgentRunner:
    """
    Executes coding-agent CLI with proper env vars and config.
    
    Example:
        runner = CodingAgentRunner(
            agent_path=Path("coding-agent"),
            config_path=Path("config/coding_config.yaml"),
            env_vars={"MODEL": "claude-sonnet-4"}
        )
        runner.run(workspace=Path("data/repos/my-repo"))
    """
    
    def __init__(self, agent_path: Path, config_path: Path,
                 env_vars: Optional[Dict[str, str]] = None):
        self.agent_path = agent_path
        self.config_path = config_path
        self.env_vars = env_vars or {}
    
    def run(self, workspace: Path, output_dir: Path,
            timeout: int = 3600) -> subprocess.CompletedProcess:
        """Run coding-agent CLI"""
        
        # Prepare environment variables
        env = os.environ.copy()
        env.update(self.env_vars)
        env["CONFIG_PATH"] = str(self.config_path)
        
        # Build command
        cmd = [
            "python", "-m", "coding_agent",
            "--workspace", str(workspace),
            "--output", str(output_dir)
        ]
        
        # Execute
        result = subprocess.run(
            cmd,
            cwd=self.agent_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Coding agent failed: {result.stderr}")
        
        return result
```

### 9. Container Runner (`infra/runners/container.py`)

```python
import docker
from typing import Dict, List, Optional

class ContainerRunner:
    """Wrapper for Docker SDK operations"""
    
    def __init__(self, image: str, volumes: Dict[str, str],
                 command: List[str], 
                 environment: Optional[Dict[str, str]] = None,
                 timeout: int = 3600):
        self.client = docker.from_env()
        self.image = image
        self.volumes = volumes
        self.command = command
        self.environment = environment or {}
        self.timeout = timeout
    
    def run(self) -> str:
        """Run container and return logs"""
        try:
            container = self.client.containers.run(
                self.image,
                command=self.command,
                volumes=self.volumes,
                environment=self.environment,
                detach=True,
                remove=False
            )
            
            # Wait with timeout
            result = container.wait(timeout=self.timeout)
            logs = container.logs().decode('utf-8')
            
            if result['StatusCode'] != 0:
                raise RuntimeError(f"Container failed: {logs}")
            
            container.remove()
            return logs
            
        except docker.errors.DockerException as e:
            raise RuntimeError(f"Docker error: {e}")
```

### 10. Benchmark Adapter (`infra/benchmarks/pymigbench.py`)

```python
from pathlib import Path
from typing import List
import pymigbench
from .base import Benchmark
from infra.pipeline.config import MigConfigImpl
from infra.models.migration import MigrationInfo

class PyMigBench(Benchmark):
    """Adapter for PyMigBench dataset"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
    
    def list_migrations(self) -> List[str]:
        """List all PyMigBench migrations"""
        return pymigbench.list_all()
    
    def load_migration(self, migration_id: str) -> tuple[MigConfigImpl, MigrationInfo]:
        """Load PyMigBench migration"""
        migration = pymigbench.load_migration(migration_id)
        
        config = MigConfigImpl.from_pymigbench(migration, self.data_dir)
        
        mig_info = MigrationInfo(
            lib_a=migration.lib_a,
            lib_b=migration.lib_b,
            version_a=migration.version_a,
            version_b=migration.version_b
        )
        
        return config, mig_info
```

### 11. CLI Entry Point (`infra/cli.py`)

```python
import click
from pathlib import Path
from infra.benchmarks.pymigbench import PyMigBench
from infra.pipeline.dag import DAGExecutor
from infra.tasks.repo import RepoDlTask
from infra.tasks.env import EnvSetupAgentTask
from infra.tasks.test_gen import HelperTestAgentTask, EvalTestAgentTask
from infra.tasks.coding import CodingAgentTask
from infra.tasks.eval import PatchCmpEvalTask, UnitTestEvalTask

@click.group()
def cli():
    """Migration pipeline CLI"""
    pass

@cli.command()
@click.argument('migration_id')
@click.option('--data-dir', default='data', type=click.Path())
@click.option('--force-coding', is_flag=True)
@click.option('--force-tests', is_flag=True)
def run(migration_id: str, data_dir: str, force_coding: bool, force_tests: bool):
    """Run migration pipeline for a single migration"""
    
    data_path = Path(data_dir)
    
    # Load from benchmark
    benchmark = PyMigBench(data_path)
    config, mig_info = benchmark.load_migration(migration_id)
    
    # Build task list
    tasks = build_pipeline(config, mig_info, force_coding, force_tests)
    
    # Execute
    click.echo(f"Running migration: {migration_id}")
    executor = DAGExecutor(tasks)
    executor.execute(parallel=True)
    
    click.echo("✓ Pipeline completed")

@cli.command()
@click.option('--data-dir', default='data', type=click.Path())
def list_migrations(data_dir: str):
    """List all available migrations"""
    benchmark = PyMigBench(Path(data_dir))
    migrations = benchmark.list_migrations()
    
    click.echo(f"Found {len(migrations)} migrations:")
    for mig in migrations:
        click.echo(f"  - {mig}")

def build_pipeline(config, mig_info, force_coding=False, force_tests=False):
    """Build task DAG"""
    
    # Create all tasks
    repo_dl = RepoDlTask(config, repo_name="...")
    env_setup = EnvSetupAgentTask(config, repo_dl)
    
    helper_test = HelperTestAgentTask(
        config, repo_dl, mig_info, env_setup,
        force_rerun=force_tests
    )
    
    eval_test = EvalTestAgentTask(
        config, repo_dl, force_rerun=force_tests
    )
    
    coding = CodingAgentTask(
        config, repo_dl, mig_info, helper_test, env_setup,
        agent_path=Path("coding-agent"),
        force_rerun=force_coding
    )
    
    patch_eval = PatchCmpEvalTask(config, coding, gt_patch=None)
    unit_eval = UnitTestEvalTask(config, coding, eval_test, repo_dl, env_setup)
    
    return [repo_dl, env_setup, helper_test, eval_test, 
            coding, patch_eval, unit_eval]

if __name__ == '__main__':
    cli()
```

### 12. Package Configuration (`pyproject.toml`)

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "migration-infra"
version = "0.1.0"
description = "Infrastructure for AI-based code migration"
requires-python = ">=3.10"
dependencies = [
    "docker>=6.0.0",
    "click>=8.0.0",
    "pyyaml>=6.0",
]

[project.scripts]
mig-pipeline = "infra.cli:cli"

[tool.setuptools.packages.find]
where = ["."]
include = ["infra*"]
```

---

## Usage Examples

```bash
# Install in development mode
pip install -e .

# List available migrations
mig-pipeline list-migrations

# Run single migration
mig-pipeline run requests_urllib3_1.0_to_2.0

# Force rerun coding agent only
mig-pipeline run requests_urllib3_1.0_to_2.0 --force-coding

# Force rerun all tests
mig-pipeline run requests_urllib3_1.0_to_2.0 --force-tests
```

```python
# From Python
from pathlib import Path
from infra.benchmarks.pymigbench import PyMigBench
from infra.pipeline.dag import DAGExecutor

# Load migration
benchmark = PyMigBench(Path("data"))
config, mig_info = benchmark.load_migration("requests_urllib3")

# Build and run
tasks = build_pipeline(config, mig_info)
executor = DAGExecutor(tasks)
executor.execute(parallel=True)

# Access results
context = executor.context
patch = context.get_artifact("coding_agent")
score = context.get_artifact("patch_eval")
```

---

## Key Design Benefits

1. **Efficient Caching**: File-based caching avoids redundant computation
2. **Parallel Execution**: DAG executor runs independent tasks in parallel
3. **Modularity**: Each component (task, runner, model) is independent
4. **Extensibility**: Easy to add new tasks, benchmarks, or agents
5. **Testability**: Each component can be unit tested
6. **Type Safety**: Strong typing with dataclasses and protocols
7. **Configuration-Driven**: Behavior controlled via shared config
8. **Submodule Isolation**: Agents accessed only through wrappers
