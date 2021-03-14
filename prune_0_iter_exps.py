import os
from pathlib import Path
from pprint import pprint
import shutil
import json

log_dir = Path("logs")
tensorboard_dir = Path("tensorboard")
files = sorted(list(os.listdir(log_dir)))
files = [file for file in files if str(file)[:4] == "expr"]

expr_iterations = {}
expr_params_iterations = {}
for expr_log in files:
    if (log_dir/expr_log/"prune_iterations.txt").exists():
        with open(log_dir/expr_log/"prune_iterations.txt", "r") as handle:
            expr_iterations[expr_log] = int(handle.readline().strip())
    else:
        expr_iterations[expr_log] = 0
    
    if (log_dir/expr_log/"expr_params.json").exists():
        with open(log_dir/expr_log/"expr_params.json", "r") as handle:
            expr_params_iterations[expr_log] = json.load(handle)
    else:
        expr_params_iterations[expr_log] = {}

files.sort()

files_to_delete = []

for expr_id in files:
    if expr_id in expr_iterations:
        if expr_iterations[expr_id] == 0:
            print("\u001b[31m", end="")
            files_to_delete.append(expr_id)
        else:
            print("\u001b[0m", end="")
        print(expr_id, expr_iterations[expr_id])
        # print(expr_id, expr_params_iterations[expr_id]["model_training_params"]["meta_lr"])

delete = input(f"found {len(files_to_delete)} 0 iteration logs. delete? (y/n)")

if delete == "y":
    for expr_id in files_to_delete:
        if (log_dir/expr_id).exists():
            print(f"deleting {log_dir/expr_id}")
            shutil.rmtree(log_dir/expr_id)
        if (tensorboard_dir/expr_id).exists():
            print(f"deleting {tensorboard_dir/expr_id}")
            shutil.rmtree(tensorboard_dir/expr_id)
