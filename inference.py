import os
import requests
import json
from env.models import DataAction, ActionType
from typing import Any

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN", "")
MAX_STEPS = 8

FALLBACK_ACTION = {
    "action_type": "NOOP",
    "justification": "invalid LLM output"
}

def call_llm(prompt: str) -> dict:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Reply ONLY in valid JSON matching DataAction schema. Think step by step. Prioritize: INSPECT first, then fix highest severity, then VALIDATE. Include justification explaining reasoning."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 512
    }
    try:
        resp = requests.post(f"{API_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return json.loads(text)
    except Exception:
        return FALLBACK_ACTION

def obs_to_prompt(obs: dict) -> str:
    return f"Observation: {json.dumps(obs)}\nWhat DataAction should the agent take next? Reply in valid JSON."

def run_task(task_id: int):
    print(f"\n=== Running Task {task_id} ===")
    # 1. POST /reset
    obs = requests.post(f"{API_BASE_URL}/reset?task_id={task_id}").json()
    for step in range(MAX_STEPS):
        prompt = obs_to_prompt(obs)
        action_dict = call_llm(prompt)
        # Validate action_dict
        if "action_type" not in action_dict or "justification" not in action_dict:
            action_dict = FALLBACK_ACTION
        # 6. POST /step
        step_result = requests.post(f"{API_BASE_URL}/step?task_id={task_id}", json=action_dict).json()
        obs = step_result["observation"]
        print(f"Step {step+1}: Action: {action_dict['action_type']} | Reward: {step_result['reward']} | Done: {step_result['done']}")
        if step_result["done"]:
            break
    # 8. GET /grader
    grader = requests.get(f"{API_BASE_URL}/grader?task_id={task_id}").json()
    print(f"Final Score: {grader['score']} | Breakdown: {grader['breakdown']}")
    return grader["score"]

def main():
    total = 0.0
    for tid in [1, 2, 3]:
        score = run_task(tid)
        total += score
    print(f"\nTotal Score: {round(total, 4)}")

if __name__ == "__main__":
    main()
