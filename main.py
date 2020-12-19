import torch
import numpy as np
from src import utils, load_data
from src.models import lth_maml as lth
from pathlib import Path

print(f"found gpu: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_params = {
    "project_dir": str(Path(__file__).parent),
    "expr_id": utils.new_expr_id("lth_maml", "use_xavier_as_per_maml"),
    "random_seed": 223,
    "cudnn_enabled": True,
    "dataset_params": {
        "n_way": 5,                         # number of classes to choose between for each task
        "k_spt": 1,                         # k shot for support set (number of examples per class per task)
        "k_qry": 15,                        # k shot for query set (number of examples per class per task)
        "imgsz": 84,                        # image size
        "task_num": 4,                      # meta model batch size
        "train_bs": 10000,                  # training batch size
        "test_bs": 100,                     # val/test batch size
        "dataset_name": "mini_imagenet",
        "dataset_location": "data/miniimagenet",
    },
    "prune_strategy": {
        "name": "global",
        "rate": .1,
        "iterations": 15,
    },
    "model_training_params": {
        "model_name": "MAML",
        "meta_training_epochs": 8,              # meta model training epochs
        "meta_training_early_stopping": True,   # meta model early stopping
        "meta_lr": .001,                        # meta model learning rate
        "update_lr": 0.01,                      # task specific model learning rate
        "update_step": 5,                       # task specific model training 
        "finetune_step": 10,                    # task specific model finetuning testing 
        "dtype": "float32",
        "layer_definitions": [
            ('conv2d', [32, 3, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            # ('dropout', [.5]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            # ('dropout', [.5]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [64, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [64]),
            # ('dropout', [.5]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [64, 64, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [64]),
            # ('dropout', [.5]),
            ('max_pool2d', [2, 1, 0]),
            ('flatten', []),
            # ('linear', [5, 11552]) # 32 * 5 * 5
            # ("linear", [5, 1600])
            # ('linear', [32, 1600]),
            # ('relu', [True]),
            # ('bn', [32]),
            # ('dropout', [.5]),
            ('linear', [5, 1600]),
            # ('linear', [5, 64])
        ]
    }
}

test_params = {
    "project_dir": str(Path(__file__).parent/"logs"/"expr|2020-12-06|17:31:18|7CpKn|lth_maml"),
    "expr_id": "expr|2020-12-06|17:31:18|7CpKn|lth_maml",
    "train_test": "test",
}

for i, expr_params in enumerate([train_params]):
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> STARTING EXPR {i} <<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(expr_params)
    
    if expr_params.get("train_test", "train") == "train":
        utils.set_seeds(expr_params["random_seed"], expr_params.get("cudnn_enabled", True))

        args = expr_params["model_training_params"]

        print(f"starting lth run {expr_params['expr_id']}")
        dataset = load_data.dataset(expr_params["dataset_params"], redownload=False)
        mask = lth.run(dataset, expr_params)
    else:
        dataset = load_data.dataset(expr_params["dataset_params"], redownload=False)

        train, val, test = dataset
        log_dir = expr_params["project_dir"]

        print(f"log_dir: {log_dir}")
        print(lth.test_finetuning(test, expr_params, log_dir))
