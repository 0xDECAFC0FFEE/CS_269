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

for expr_id in files:
    if expr_id in expr_iterations:
        print(expr_id, expr_iterations[expr_id])
        # print(expr_id, expr_params_iterations[expr_id]["model_training_params"]["meta_lr"])

delete = input("delete 0 iteration logs? (y/n)")

if delete == "y":
    for expr_id in files:
        if expr_id in expr_iterations and expr_iterations[expr_id] == 0:
            if (log_dir/expr_id).exists():
                shutil.rmtree(log_dir/expr_id)
            if (tensorboard_dir/expr_id).exists():
                shutil.rmtree(tensorboard_dir/expr_id)
