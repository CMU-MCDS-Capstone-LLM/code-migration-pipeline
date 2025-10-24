## TODO

- [ ] b/f Oct 24: Set up fallback ver of swe-agent (no exec on repo)

- [ ] b/f Oct 24: Set up evaluator

- [ ] b/f Oct 24: Set up env for more repos

- [ ] b/f Oct 24: Ignore testing agent, and run on 30 data points

## Schedule

Teammate's schedule starting from Oct 21:

- Yogesh: This week not free, next week free

- Shanru: This week online (time zone diff, subtract 3 hours), next week free till Wednesday

## Architecture

```mermaid
flowchart LR
  RepoDl([pymigbench_dl])
  Repo[[repo, commit bf mig]]
  MigInfo[[libA, libB]]
  
  EnvAgent([env setup agent])
  Docker[[repo env Dockerfile]]

  HelperTestAgent([testing agent to generate helper test])
  HelperTests[[helper tests]]
  
  EvalTestAgent([testing agent for eval])
  EvalTests[[eval tests]]

  Code([coding agent])
  
  GTPatch[[ground truth patch]]
  GenPatch[[generated patch]]

  PatchCmpEval([patch cmp evaluator])
  PatchSimScore[[patch sim scores]]

  UnitTestEval([patch cmp evaluator])
  UnitTestScore[[unit test pass rate scores]]

  MigInfo --> Code 

  RepoDl --> Repo 
  Repo--> EnvAgent --> Docker

  MigInfo --> HelperTestAgent 
  Repo --> HelperTestAgent
  Docker --> HelperTestAgent
  HelperTestAgent --> HelperTests
  
  HelperTests --> Code
  Repo --> Code
  Docker --> Code
  Code --> GenPatch 

  GenPatch --> PatchCmpEval
  GTPatch --> PatchCmpEval
  PatchCmpEval --> PatchSimScore

  GTPatch --> EvalTestAgent
  Repo --> EvalTestAgent
  EvalTestAgent --> EvalTests

  GenPatch --> UnitTestEval
  EvalTests --> UnitTestEval
  UnitTestEval --> UnitTestScore
  
```
