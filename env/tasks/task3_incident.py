import pandas as pd
import json
from typing import Dict, Any
from env.models import DataObservation, DataAction, StepResult, DetectedIssue, ActionType
from env.data.generator import generate_employee_dataset
from env.data.bug_injector import inject_bugs

SCENARIO_PATH = "env/data/scenarios/task3_scenario.json"

BLAST_RADIUS = {
    "revenue_stage3": ["total_revenue", "avg_revenue"],
    "ssn": ["analytics_output"],
}

class Task3Incident:
    def __init__(self):
        self.state: Dict[str, Any] = {}
        self.max_steps = 8
        self.task_id = 3
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
            "diagnosis": False,
            "fix": False,
            "pii_sweep": False,
            "validation": False,
            "justifications": [],
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
            pipeline_stage="incident_response",
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
            if gt["issue_type"].startswith("Schema drift"):
                col = gt["column"]
                if col not in df.columns:
                    issues.append(DetectedIssue(
                        issue_type="Schema drift",
                        column=col,
                        description=f"Schema drift: {col} missing",
                        severity="high"
                    ))
            elif gt["issue_type"].startswith("Type error"):
                col = gt["column"]
                row = gt["row"]
                if col in df.columns and isinstance(df.at[row, col], str):
                    issues.append(DetectedIssue(
                        issue_type="Type error",
                        column=col,
                        description=f"Type error in {col} at row {row}",
                        severity="high"
                    ))
            elif gt["issue_type"].startswith("PII leak"):
                col = gt["column"]
                row = gt["row"]
                if "analytics_output" in df.columns and df.at[row, "analytics_output"] == df.at[row, col]:
                    issues.append(DetectedIssue(
                        issue_type="PII leak",
                        column=col,
                        description=f"PII leak in {col} at row {row}",
                        severity="critical"
                    ))
            elif gt["issue_type"].startswith("Duplicate aggregation"):
                col = gt["column"]
                row = gt["row"]
                if col in df.columns and df.at[row, col] == 2 * 100000:
                    issues.append(DetectedIssue(
                        issue_type="Duplicate aggregation",
                        column=col,
                        description=f"Duplicate aggregation in {col} at row {row}",
                        severity="medium"
                    ))
        return issues

    def step(self, action: DataAction) -> StepResult:
        reward = 0.0
        info = {}
        done = False
        self.state["step_count"] += 1
        df = self.state["df"]
        just = action.justification.lower()
        self.state["justifications"].append(just)
        # DIAGNOSE
        if "stage 3" in just or "join stage" in just:
            self.state["diagnosis"] = True
            reward += 0.15
        # FIX
        if action.action_type == ActionType.CAST_TYPE and action.target_column == "revenue_stage3":
            try:
                df["revenue_stage3"] = pd.to_numeric(df["revenue_stage3"], errors="coerce").fillna(0).astype(int)
                self.state["fix"] = True
                reward += 0.20
            except Exception:
                reward -= 0.10
        # PII SWEEP
        if action.action_type == ActionType.MASK_PII:
            if "analytics_output" in df.columns:
                df["analytics_output"] = df["analytics_output"].apply(lambda x: "XXX-XX-XXXX" if isinstance(x, str) and len(x) == 11 else x)
                self.state["pii_sweep"] = True
                reward += 0.20
        # VALIDATE
        if action.action_type == ActionType.VALIDATE:
            if not self._validate():
                self.state["validation"] = True
                reward += 0.25
        # BLAST RADIUS
        if action.action_type == ActionType.DROP_COLUMN and action.target_column:
            dependents = BLAST_RADIUS.get(action.target_column, [])
            penalty = -0.10 * len(dependents)
            self.state["downstream_health"] += -0.15 * len(dependents)
            reward += penalty
            if action.target_column in df.columns:
                df.drop(columns=[action.target_column], inplace=True)
        # NOOP
        if action.action_type == ActionType.NOOP:
            reward += 0.0
        # Redundant/repeated action
        if self.state["justifications"].count(just) > 1:
            reward -= 0.05
        # Check for done
        if self.state["step_count"] >= self.max_steps or (self.state["diagnosis"] and self.state["fix"] and self.state["pii_sweep"] and self.state["validation"]):
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
