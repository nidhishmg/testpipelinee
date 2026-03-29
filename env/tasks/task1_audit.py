import pandas as pd
import json
from typing import Tuple, Dict, Any
from env.models import DataObservation, DataAction, StepResult, DetectedIssue, ActionType
from env.data.generator import generate_employee_dataset
from env.data.bug_injector import inject_bugs

SCENARIO_PATH = "env/data/scenarios/task1_scenario.json"

class Task1Audit:
    def __init__(self):
        self.state: Dict[str, Any] = {}
        self.max_steps = 8
        self.task_id = 1
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
            "issues_found": [],
            "issues_fixed": [],
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
            pipeline_stage="raw_ingest",
            validation_report=validation_report,
            time_remaining=self.max_steps - self.state["step_count"],
            downstream_health=self.state["downstream_health"],
            step_count=self.state["step_count"],
            task_id=self.task_id
        )

    def _validate(self) -> list:
        df = self.state["df"]
        issues = []
        for gt in self.state["ground_truth"]:
            col = gt["column"]
            row = gt["row"]
            if gt["issue_type"].startswith("NULL") and col and row is not None:
                if pd.isnull(df.at[row, col]):
                    issues.append(DetectedIssue(
                        issue_type="NULL value",
                        column=col,
                        description=f"NULL in {col} at row {row}",
                        severity="medium"
                    ))
            elif gt["issue_type"].startswith("Type corruption") and col and row is not None:
                if isinstance(df.at[row, col], str):
                    issues.append(DetectedIssue(
                        issue_type="Type corruption",
                        column=col,
                        description=f"Type corruption in {col} at row {row}",
                        severity="high"
                    ))
            elif gt["issue_type"].startswith("Out of range") and col and row is not None:
                if df.at[row, col] == 999:
                    issues.append(DetectedIssue(
                        issue_type="Out of range",
                        column=col,
                        description=f"Out of range value in {col} at row {row}",
                        severity="high"
                    ))
            elif gt["issue_type"].startswith("Duplicate row") and row is not None:
                # Check for duplicate rows
                if df.duplicated().any():
                    issues.append(DetectedIssue(
                        issue_type="Duplicate row",
                        column=None,
                        description=f"Duplicate row at {row}",
                        severity="medium"
                    ))
            elif gt["issue_type"].startswith("Phone format") and col and row is not None:
                if df.at[row, col] == "555-1234":
                    issues.append(DetectedIssue(
                        issue_type="Format inconsistency",
                        column=col,
                        description=f"Phone format inconsistency at row {row}",
                        severity="low"
                    ))
        return issues

    def step(self, action: DataAction) -> StepResult:
        reward = 0.0
        info = {}
        done = False
        self.state["step_count"] += 1
        # Action logic
        if action.action_type == ActionType.INSPECT:
            # Reward for identifying issues
            found = len(self._validate())
            reward += 0.15 * found
            self.state["issues_found"].extend(self._validate())
        elif action.action_type == ActionType.FILL_DEFAULT:
            # Fill NULLs in salary
            df = self.state["df"]
            if "salary" in df.columns:
                df["salary"] = df["salary"].fillna(df["salary"].median())
                reward += 0.20
                self.state["issues_fixed"].append("salary_nulls_fixed")
        elif action.action_type == ActionType.CAST_TYPE:
            # Fix type corruption in age
            df = self.state["df"]
            if "age" in df.columns:
                df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median()).astype(int)
                reward += 0.20
                self.state["issues_fixed"].append("age_type_fixed")
        elif action.action_type == ActionType.VALIDATE:
            # Reward for validation
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
