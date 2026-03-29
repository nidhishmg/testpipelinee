import pandas as pd
from typing import Tuple, List, Dict, Any
import copy
import json

def inject_bugs(clean_df: pd.DataFrame, bug_spec_list: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    df = clean_df.copy(deep=True)
    ground_truth = []
    for bug in bug_spec_list:
        bug_type = bug["type"]
        if bug_type == "null_salary":
            for row in bug["rows"]:
                idx = row if row < len(df) else len(df)-1
                df.at[idx, "salary"] = None
                ground_truth.append({"issue_type": "NULL in salary", "row": idx, "column": "salary"})
        elif bug_type == "type_corruption":
            idx = bug["row"]
            df.at[idx, "age"] = "twenty-three"
            ground_truth.append({"issue_type": "Type corruption in age", "row": idx, "column": "age"})
        elif bug_type == "out_of_range":
            idx = bug["row"]
            df.at[idx, "age"] = 999
            ground_truth.append({"issue_type": "Out of range age", "row": idx, "column": "age"})
        elif bug_type == "duplicate_rows":
            for row in bug["rows"]:
                idx = row if row < len(df) else len(df)-1
                df = pd.concat([df.iloc[:idx+1], df.iloc[[idx]], df.iloc[idx+1:]], ignore_index=True)
                ground_truth.append({"issue_type": "Duplicate row", "row": idx, "column": None})
        elif bug_type == "format_inconsistency":
            idx = bug["row"]
            df.at[idx, "phone"] = "555-1234"
            ground_truth.append({"issue_type": "Phone format inconsistency", "row": idx, "column": "phone"})
        elif bug_type == "rename_column":
            df.rename(columns={bug["from"]: bug["to"]}, inplace=True)
            ground_truth.append({"issue_type": f"Renamed {bug['from']} to {bug['to']}", "row": None, "column": bug["to"]})
        elif bug_type == "type_change":
            idx = bug["row"]
            col = bug["column"]
            df.at[idx, col] = pd.to_datetime(df.at[idx, col]).date() if col in df.columns else None
            ground_truth.append({"issue_type": f"Type changed in {col}", "row": idx, "column": col})
        elif bug_type == "missing_column":
            col = bug["column"]
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
            ground_truth.append({"issue_type": f"Missing column {col}", "row": None, "column": col})
        elif bug_type == "add_column":
            col = bug["column"]
            df[col] = bug.get("default", None)
            ground_truth.append({"issue_type": f"Added column {col}", "row": None, "column": col})
        elif bug_type == "schema_drift":
            # For task 3, simulate schema drift at a pipeline stage
            col = bug["column"]
            if col in df.columns:
                df.rename(columns={col: bug["to"]}, inplace=True)
            ground_truth.append({"issue_type": f"Schema drift: {col} to {bug['to']}", "row": None, "column": bug["to"]})
        elif bug_type == "type_error":
            idx = bug["row"]
            col = bug["column"]
            df.at[idx, col] = "100000.00"  # string instead of int
            ground_truth.append({"issue_type": f"Type error in {col}", "row": idx, "column": col})
        elif bug_type == "pii_leak":
            col = bug["column"]
            # Copy SSN to analytics_output
            if "analytics_output" not in df.columns:
                df["analytics_output"] = ""
            df.at[bug["row"], "analytics_output"] = df.at[bug["row"], col]
            ground_truth.append({"issue_type": "PII leak", "row": bug["row"], "column": col})
        elif bug_type == "duplicate_aggregation":
            col = bug["column"]
            idx = bug["row"]
            df.at[idx, col] = df.at[idx, col] * 2
            ground_truth.append({"issue_type": f"Duplicate aggregation in {col}", "row": idx, "column": col})
    return df, ground_truth
