import numpy as np
import wandb
import wandb.wandb_run

api = wandb.Api()

# Define the project (replace with your actual project and entity)
entity = "L65_Project"
project = "transformer-graph-learner"

# Fetch runs
runs = api.runs(f"{entity}/{project}")

for run in runs:
    # if run.group != "debug": continue
    if run.state == "running":
        continue
    if "best_test_loss" in run.summary:
        continue
    test_loss = run.history(keys=["test_loss"])
    if "test_loss" not in test_loss.columns:
        continue
    test_loss = test_loss["test_loss"]
    best_test_loss = np.min(test_loss)
    print(run.name, best_test_loss)
    run.summary["best_test_loss"] = best_test_loss
    run.summary.update()
