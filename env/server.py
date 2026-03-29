from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from env.tasks.task1_audit import Task1Audit
from env.tasks.task2_schema import Task2Schema
from env.tasks.task3_incident import Task3Incident
from env.models import DataAction, StepResult, DataObservation, GraderResult
from env.graders.grader1 import grader1
from env.graders.grader2 import grader2
from env.graders.grader3 import grader3
from pydantic import ValidationError
from typing import Dict

app = FastAPI()

tasks = {
    1: Task1Audit(),
    2: Task2Schema(),
    3: Task3Incident(),
}

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/tasks")
def get_tasks():
    return [
        {"id": 1, "name": "Data Quality Audit", "difficulty": "easy", "max_steps": 8},
        {"id": 2, "name": "Schema Drift Remediation", "difficulty": "medium", "max_steps": 8},
        {"id": 3, "name": "Full Data Incident Response", "difficulty": "hard", "max_steps": 8},
    ]

@app.post("/reset")
def reset(task_id: int):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    obs = tasks[task_id].reset()
    return obs

@app.post("/step")
def step(task_id: int, action: DataAction):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    result = tasks[task_id].step(action)
    return result

@app.get("/state")
def state(task_id: int):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    obs = tasks[task_id].state_obs()
    return obs

@app.get("/grader")
def grader(task_id: int):
    if task_id == 1:
        # For demo, use dummy values
        return grader1(identified=5, fixed=5, total=5)
    elif task_id == 2:
        return grader2(rows_passing=95, total_rows=100)
    elif task_id == 3:
        return grader3(0.8, 0.6, 1.0, 0.5, ["stage 3", "SSN"])
    else:
        raise HTTPException(status_code=404, detail="Task not found")

@app.get("/baseline")
def baseline():
    # NOOP agent baseline score
    return {"score": 0.0}
