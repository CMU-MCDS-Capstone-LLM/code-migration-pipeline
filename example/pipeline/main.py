# example/main.py
"""
Demo pipeline showcasing typed task dependencies.

Pipeline structure:
    source1(5) -> incr1(+10) -> add(result: 42)
                                 ^
    source2(7) -> incr2(+20) ---+
"""

from pathlib import Path

from infra.pipeline.dag import DAGExecutor
from infra.pipeline.config import MigConfig
from pymigbench_dl.providers.github.models import CommitInfo

from .task.SourceTask import SourceTask
from .task.IncrTask import IncrTask
from .task.AddTask import AddTask


def get_dummy_config() -> MigConfig:
    p = Path(".")
    return MigConfig(CommitInfo("", ""), p, p, p, p, p, p)


def demo_1():
    print("Demo 1: 5 + (7 incr by 20)")
    config = get_dummy_config()
    # Create two source tasks
    source1 = SourceTask("source1", value=5, config=config)
    source2 = SourceTask("source2", value=7, config=config)

    # Create two increment tasks
    incr2 = IncrTask("incr2", source=source2, delta=20, config=config)

    # Create the add task that depends on both increments
    add = AddTask("add", input1=source1, input2=incr2, config=config)

    tasks = [source1, source2, incr2, add]
    executor = DAGExecutor(tasks)
    executor.execute()

    # Print final result
    if add.output is not None:
        print(f"Output: {add.output.result}")
    else:
        print("No output")


def demo_2():
    print("Demo 2: (5 incr by 10) + (7 incr by 20)")
    config = get_dummy_config()
    # Create two source tasks
    source1 = SourceTask("source1", value=5, config=config)
    source2 = SourceTask("source2", value=7, config=config)

    # Create two increment tasks
    incr1 = IncrTask("incr1", source=source1, delta=10, config=config)
    incr2 = IncrTask("incr2", source=source2, delta=20, config=config)

    # Create the add task that depends on both increments
    add = AddTask("add", input1=incr1, input2=incr2, config=config)

    tasks = [source1, source2, incr1, incr2, add]
    executor = DAGExecutor(tasks)
    executor.execute()

    # Print final result
    if add.output is not None:
        print(f"Output: {add.output.result}")
    else:
        print("No output")


def main():
    """Run the demo pipeline."""
    demo_1()
    demo_2()


if __name__ == "__main__":
    main()
