import pandas as pd
import numpy as np
from typing import List

def generate_employee_dataset(seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    n = 100
    departments = ["Engineering", "Sales", "HR", "Finance"]
    names = [f"Employee_{i}" for i in range(n)]
    employee_id = np.arange(1, n+1)
    age = np.random.randint(22, 66, size=n)
    salary = np.random.randint(30000, 150001, size=n)
    department = np.random.choice(departments, size=n)
    phone = [f"+1-555-{np.random.randint(1000,9999):04d}" for _ in range(n)]
    ssn = [f"XXX-XX-{np.random.randint(1000,9999):04d}" for _ in range(n)]
    hire_date = pd.to_datetime('2015-01-01') + pd.to_timedelta(np.random.randint(0, 365*10, size=n), unit='D')
    hire_date = hire_date.strftime('%Y-%m-%d').tolist()
    consent_flag = np.random.choice([True, False], size=n)
    df = pd.DataFrame({
        "employee_id": employee_id,
        "name": names,
        "age": age,
        "salary": salary,
        "department": department,
        "phone": phone,
        "ssn": ssn,
        "hire_date": hire_date,
        "consent_flag": consent_flag
    })
    return df
