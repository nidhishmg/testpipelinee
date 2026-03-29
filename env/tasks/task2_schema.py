import pandas as pd
import json
from typing import Dict, Any
from env.models import DataObservation, DataAction, StepResult, DetectedIssue, ActionType
from env.data.generator import generate_employee_dataset
from env.data.bug_injector import inject_bugs

SCENARIO_PATH = "env/data/scenarios/task2_scenario.json"
COLUMN_DEPENDENCIES = {
    "salary":      ["salary_ratio", "tax_band"],
    "employee_id": ["dept_head_flag", "salary_ratio"],
    "hire_date":   ["tenure_years"],
}

class Task2Schema:
    def __init__(self):
        self.state: Dict[str, Any] = {}
        self.max_steps = 8
        self.task_id = 2
        self.reset()

    def reset(self) -> DataObservation:
        with open(SCENARIO_PATH, "r") as f:
            bug_spec = json.load(f)
        clean_df = generate_employee_dataset(seed=42)
        corrupted_df, ground_truth = inject_bugs(clean_df, bug_spec)
        self.state = {
            "df": corrupted_df,
            "ground_truth": ground_truth,
            "step_count": 0,
            "done": False,
            "downstream_health": 1.0,
        }
        return self._make_observation()

    def _make_observation(self) -> DataObservation:
        df = self.state["df"]
        preview = df.head(10).to_dict(orient="records")
        schema = {col: {"type": str(df[col].dtype), "nullable": df[col].isnull().any()} for col in df.columns}
        validation_report = self._validate()
        return DataObservation(
            dataset_preview=preview,
            schema=schema,
            pipeline_stage="schema_remediation",
            validation_report=validation_report,
            time_remaining=self.max_steps - self.state["step_count"],
            downstream_health=self.state["downstream_health"],
            step_count=self.state["step_count"],
            task_id=self.task_id
        )

    def _validate(self) -> list:
        df = self.state["df"]
        issues = []
        # Check for missing/renamed columns and type
        for gt in self.state["ground_truth"]:
            if gt["issue_type"].startswith("Renamed"):
                if gt["column"] not in df.columns:
                    issues.append(DetectedIssue(
                        issue_type="Schema drift",
                        column=gt["column"],
                        description=f"Column {gt['column']} missing",
                        severity="high"
                    ))
            elif gt["issue_type"].startswith("Type changed"):
                col = gt["column"]
                row = gt["row"]
                if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                    issues.append(DetectedIssue(
                        issue_type="Type drift",
                        column=col,
                        description=f"Type drift in {col} at row {row}",
                        severity="medium"
                    ))
            elif gt["issue_type"].startswith("Missing column"):
                col = gt["column"]
                if col not in df.columns:
                    issues.append(DetectedIssue(
                        issue_type="Missing column",
                        column=col,
                        description=f"Missing column {col}",
                        severity="high"
                    ))
            elif gt["issue_type"].startswith("Added column"):
                col = gt["column"]
                if col not in df.columns:
                    issues.append(DetectedIssue(
                        issue_type="Missing added column",
                        column=col,
                        description=f"Added column {col} missing",
                        severity="medium"
                    ))
        return issues

    def step(self, action: DataAction) -> StepResult:
        reward = 0.0
        info = {}
        done = False
        self.state["step_count"] += 1
        df = self.state["df"]
        # Action logic
        if action.action_type == ActionType.RENAME_COLUMN and action.target_column:
            # Try to rename back
            for gt in self.state["ground_truth"]:
                if gt["issue_type"].startswith("Renamed") and gt["column"] not in df.columns:
                    old = gt["column"]
                    if action.target_column == old:
                        df.rename(columns={action.target_column: gt["column"]}, inplace=True)
                        reward += 0.20
        elif action.action_type == ActionType.CAST_TYPE and action.target_column:
            if action.target_column in df.columns:
                try:
                    df[action.target_column] = pd.to_datetime(df[action.target_column])
                    reward += 0.20
                except Exception:
                    reward -= 0.10
        elif action.action_type == ActionType.FILL_DEFAULT and action.target_column:
            if action.target_column in df.columns:
                df[action.target_column] = df[action.target_column].fillna("default")
                reward += 0.20
        elif action.action_type == ActionType.DROP_COLUMN and action.target_column:
            if action.target_column in df.columns:
                dependents = COLUMN_DEPENDENCIES.get(action.target_column, [])
                penalty = -0.10 * len(dependents)
                self.state["downstream_health"] += -0.15 * len(dependents)
                reward += penalty
                df.drop(columns=[action.target_column], inplace=True)
        elif action.action_type == ActionType.VALIDATE:
            if not self._validate():
                reward += 0.25
        elif action.action_type == ActionType.NOOP:
            reward += 0.0
        else:
            reward -= 0.10
        # Check for done
        if self.state["step_count"] >= self.max_steps or not self._validate():
            done = True
        obs = self._make_observation()
        return StepResult(
            observation=obs,
            reward=round(reward, 4),
            done=done,
            info=info
        )

    def state_obs(self) -> DataObservation:
        return self._make_observation()
