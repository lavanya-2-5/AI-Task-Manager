import pandas as pd
from collections import Counter

TEAM_MEMBERS = ['Nikeeta', 'Tanisha', 'Lavanya', 'Avi', 'Lavisha']

def auto_assign_tasks():
    df = pd.read_csv("model_predictions.csv")

    # Filter only unassigned tasks
    unassigned_tasks = df[df['AssignedTo'].isna() | (df['AssignedTo'] == '')]

    if unassigned_tasks.empty:
        return df  # Nothing to assign

    # Count current workload
    current_assignments = df['AssignedTo'].dropna()
    workload = Counter(current_assignments)

    # Initialize workload for missing team members
    for member in TEAM_MEMBERS:
        if member not in workload:
            workload[member] = 0

    # Assign unassigned tasks
    for idx in unassigned_tasks.index:
        least_loaded = min(workload, key=workload.get)
        df.at[idx, 'AssignedTo'] = least_loaded
        workload[least_loaded] += 1

    df.to_csv("model_predictions.csv", index=False)
    return df

def reset_assignments():
    df = pd.read_csv("model_predictions.csv")
    df['AssignedTo'] = ""
    df.to_csv("model_predictions.csv", index=False)
    return df