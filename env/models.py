from pydantic import BaseModel
from typing import Optional, Literal
from enum import Enum

class ActionType(str, Enum):
    INSPECT        = "INSPECT"
    RENAME_COLUMN  = "RENAME_COLUMN"
    CAST_TYPE      = "CAST_TYPE"
    FILL_DEFAULT   = "FILL_DEFAULT"
    DROP_COLUMN    = "DROP_COLUMN"
    VALIDATE       = "VALIDATE"
    MASK_PII       = "MASK_PII"
    NOOP           = "NOOP"

class DataAction(BaseModel):
    action_type:    ActionType
    target_column:  Optional[str] = None
    transformation: Optional[str] = None   # "cast_to_int", "fill_median", "fill_zero", "drop_duplicates"
    justification:  str                     # agent explains reasoning — we partially grade this

class DetectedIssue(BaseModel):
    issue_type:  str
    column:      Optional[str]
    description: str
    severity:    Literal["low", "medium", "high", "critical"]

class DataObservation(BaseModel):
    dataset_preview:   list[dict]       # first 10 rows
    schema:            dict             # {col: {type, nullable}}
    pipeline_stage:    str
    validation_report: list[DetectedIssue]
    time_remaining:    int              # steps left (max 8)
    downstream_health: float            # 0.0–1.0
    step_count:        int
    task_id:           int

class StepResult(BaseModel):
    observation: DataObservation
    reward:      float
    done:        bool
    info:        dict

class GraderResult(BaseModel):
    score:       float           # 0.0–1.0, always round to 4 decimal places
    breakdown:   dict[str, float]
    explanation: str
