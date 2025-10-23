# Testing Agent

## Logic of `run.sh`

`run.sh` is the entry point of testing agent

line 1: Export API key (OPEN_AI, LiteLLM)

line 24: Env is installed around `run.sh`

line 82 piece together env var to use repograph to find context files

line 142 incorrect logic of match test file name with file name (e.g. force `aa.py` to have a `test_aa.py`)

line 267: add dummy test, qodo-cover quirk

line 278 - 321: add import (why?)

line 335: run qodo-cover

## TODO

## Design

Testing agent accepts config

- helper test path:

  helper test is the generated test

  helper test path is the folder that holds the generated test

- repo path

- mig metadata

  - (libA, libB)

  - commit message

Test must run in repo

```
approach 1: tmp test folder

repo_folder/
  tests/
    tmp/test_xxx.py

helpter_test_path = foobar/tests/
foobar/
  tests/
    test_xxx.py

(preferred) approach 2: directly write to repo

helper_test_path = "foobar/syn_tests/", translate to "$REPO_FOLDER/<helper_test_path>/"
repo_folder/
  foobar/syn_tests/
    test_xxx.py

cd repo_folder/
pytest

pytest --collect-only
```

container communication

```
server: repo container (small process running api server, allow pytest execution)

client: testing agent container (access api server in repo container)
```

What does testing agent need from the server to generate test?

Format of testing agent python impl

```python
class TestingAgent:
  def __init__(self, test_config: TestConfig, test_executor: TestExecutor):
    pass 

class TestExecutor(ABC):
  def run_test(self, *args) -> TestInfo:
    pass

class TestConfig:
  pass

# Information needed for qodo cover to generate test
class TestInfo:
  ...

class RemoteTestExecutor(TestExecutor):
  def run_test(self, *args) -> TestInfo:
    ...
    curl http://server:8000/run_test?...
    return TestInfo()

class LocalTestExecutor(TestExecutor):
  def run_test(self, *args) -> TestInfo:
    conda activate xxxx
    pytest <args>
    return TestInfo()
```

### Xinyu

- [ ] Env setup stuff

### Shanru

- [ ] Use a yaml config file to run

- [ ] Replace `run.sh` with python code, and a `run.py` entry point

- [ ] Coverage logic:

  - run existing test to get coverage

  - find what's not covered, and generate more tests

    To avoid running unnecessary tests, we utilize lsp

  - run lsp to find a subset X of mig-related file under src/

  - run lsp to find a subset of test cases X_test that covers X

  - run X_test to see its coverage on X

  - generate more test cases to compliment X_test and fully cover X

#### Dev stages (each stage is a PR)

Each todo item below is a development stage, whose code should be pushed to GitHub as a PR and reviewed by Xinyu.

- [ ] b/f 10/24 LSP repograph as package, and cleanup

  - export a requirements.txt and python interpreter version, use Python 3.11.13 (qodo-requirements.txt IS the python package requirements for the testing agent)

  - ignore pycache

  - remove workspace from git repo

    if need to show result, add a demo/ folder

- [ ] b/f 10/24 Use pymigbench as a package, instead of parsing yaml file directly. Expect a repo-yamls/ folder that contains all the yaml files we want to run on

- [ ] **how to run repo's tests within the repo's docker container? The unit test must be executed within a container that has repo's env**.

- [ ] b/f 10/26 `run.sh` -> `run.py`, selectively choose test, and generate more with coverage
