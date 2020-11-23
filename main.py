import torch
import numpy as np
from src import utils, load_data
from src.models import lth_maml as lth
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
from src import utils
from pathlib import Path
from rich.traceback import install
install()

print(f"found gpu: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


expr_params = {
    "project_dir": str(Path(__file__).parent),
    "prune_strategy": {
        "name": "global",
        "rate": .1,
        "iterations": 1,
    },
    "seeds": {
        "torch": 222,
        "cuda": 222,
        "numpy": 222
    },
    "expr_id": utils.new_expr_id("lth_maml"),
    "model_training_params": {
        "training_iterations": 6, #epoch
        "n_way": 5,                        # number of classes to choose between for each task
        "k_spt": 1,                        # k shot for support set (number of examples per class per task)
        "k_qry": 15,                       # k shot for query set (number of examples per class per task)
        "imgsz": 84,                       # image size
        "imgc": 3,                         # this isn't used anywhere????? no idea what it does???? they say its supposed to be 1 or 3...
        "task_num": 4,                     # meta model batch size
        "meta_lr": 1e-3,                   # meta model learning rate
        "update_lr": 0.01,                 # task specific model learning rate
        "update_step": 5,                  # task specific model training epochs
        "update_step_test": 10,            # task specific model testing epochs
#         "optimizer": ("adam", {"lr": 0.0001}),
#         "loss_func": "cross_entropy",
        "model_name": "MAML",
        "dataset_name": "mini_imagenet",
        "dataset_location": "data/miniimagenet",
        "layer_definitions": None
    }
}

expr_params["model_training_params"]["layer_definitions"] = [
    ('conv2d', [32, 3, 3, 3, 1, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('max_pool2d', [2, 2, 0]),
    ('conv2d', [32, 32, 3, 3, 1, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('max_pool2d', [2, 2, 0]),
    ('conv2d', [32, 32, 3, 3, 1, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('max_pool2d', [2, 2, 0]),
    ('conv2d', [32, 32, 3, 3, 1, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('max_pool2d', [2, 1, 0]),
    ('flatten', []),
    ('linear', [expr_params["model_training_params"]["n_way"], 32 * 5 * 5]) # 32 * 5 * 5
]
utils.set_seeds(expr_params["seeds"])



args = expr_params["model_training_params"]
dataset = load_data.mini_imagenet(args, redownload=False)



print(f"starting lth run {expr_params['expr_id']}")
mask = lth.run(dataset, expr_params)
