# DataPipelineEnv README

## Pitch
Every company has broken data pipelines. Bad data costs $12.9M/year (IBM). There is no standardized benchmark to evaluate if an AI agent can fix these problems. DataPipelineEnv is the first OpenEnv-compliant environment simulating a broken enterprise data pipeline, where an LLM agent acts as an on-call data engineer.

## Task Table
| ID | Name                      | Difficulty | Max Steps |
|----|---------------------------|------------|-----------|
| 1  | Data Quality Audit        | easy       | 8         |
| 2  | Schema Drift Remediation  | medium     | 8         |
| 3  | Full Data Incident Response | hard     | 8         |

## Action Space
| ActionType      | Description                                  |
|-----------------|----------------------------------------------|
| INSPECT         | Inspect dataset for issues                   |
| RENAME_COLUMN   | Rename a column                              |
| CAST_TYPE       | Cast column type                             |
| FILL_DEFAULT    | Fill missing values                          |
| DROP_COLUMN     | Drop a column                                |
| VALIDATE        | Validate pipeline/data                       |
| MASK_PII        | Mask PII data                                |
| NOOP            | No operation                                 |

## Observation Space
- `dataset_preview`: first 10 rows
- `schema`: {col: {type, nullable}}
- `pipeline_stage`: str
- `validation_report`: list of DetectedIssue
- `time_remaining`: int
- `downstream_health`: float
- `step_count`: int
- `task_id`: int

## Reward Table
| Event                                 | Reward  |
|---------------------------------------|---------|
| Per issue identified                  | +0.15   |
| Per issue fixed                       | +0.20   |
| Pipeline stage passes validation      | +0.25   |
| Full pipeline produces correct output | +0.30   |
| Wrong action (blast radius penalty)   | -0.10*N |
| PII not masked                        | -0.20   |
| Redundant/repeated action             | -0.05   |
| NOOP                                  | 0.00    |

## Blast Radius Mechanic
Dropping a column with dependents triggers a cascade penalty:
- Penalty: -0.10 * number_of_dependents
- Downstream health: -0.15 * number_of_dependents

## Quickstart
```
# Local server
uvicorn env.server:app --reload --port 8000

# Docker
docker build -t pipeline-env . && docker run -p 8000:8000 pipeline-env

# Run agent
python inference.py

# Run tests
pytest tests/test_env.py -v
```

## Baseline
NOOP agent baseline score: 0.0

## Submission Validation
- All endpoints return correct types
- openenv.yaml passes openenv validate
- Dockerfile builds and runs
- /ping returns 200
- inference.py runs all 3 tasks
- Graders return float 0.0–1.0

## Judges: Wow Factors
- Blast radius cascade mechanic
- Justification field partial grading
- Downstream health ticks in real time
- Three grader breakdowns side by side
- Partial credit for correct reasoning
