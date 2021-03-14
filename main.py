import torch
import numpy as np
from src import utils, load_data
from src.models import lth_maml as lth
from pathlib import Path
import json

expr_id, uid = utils.new_expr_id("test new mixup code")

if not torch.cuda.is_available():
    raise Exception("CUDA ISN'T AVAILABLE WHAT WENT WRONG")
print(f"found gpu: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_params = {
    "project_dir": str(Path(__file__).parent.resolve()),
    "expr_id": expr_id,
    "uid": uid,
    "random_seed": 222,
    "cudnn_enabled": True,
    "dataset_params": {
        "n_way": 5,                         # number of classes to choose between for each task
        "k_spt": 1,                         # k shot for support set (number of examples per class per task)
        "k_qry": 15,                        # k shot for query set (number of examples per class per task)
        "imgsz": 84,                        # image size
        "task_num": 2,                      # meta model batch size
        "train_bs": 10000,                  # training batch size
        "test_bs": 100,                     # val/test batch size
        "train_image_aug": True,           # turn on image augmentations                                        False
        "mixup": True,                     # turn on mixup augmentations
        "shuffle": False,                   # shuffle dataset                                                   False
        "dataset_name": "mini_imagenet",
        "dataset_location": "data/miniimagenet",
    },
    "prune_strategy": {
        "name": "global",
        "rate": .05,
        "iterations": 15,
    },
    "model_training_params": {
        "model_name": "MAML",
        "meta_training_epochs": 32,              # meta model training epochs                                    8
        "meta_training_early_stopping": True,  # meta model early stopping                                     False
        "meta_lr": 0.0005,                       # meta model learning rate                                      0.001
        "update_lr": 0.01,                      # task specific model learning                                  0.01
        "update_step": 10,                      # task specific model training                                  10
        "finetune_step": 10,                    # task specific model finetuning testing                        10
        "first_order": False,
        "dtype": "float32",
        "layer_definitions": [
            ('conv2d', [32, 3, 3, 3, 1, 1]),
            ('relu', [True]),
            ('bn', [32]),
            # ('dropout', [.5]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [32, 32, 3, 3, 1, 1]),
            ('relu', [True]),
            ('bn', [32]),
            # ('dropout', [.5]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [64, 32, 3, 3, 1, 1]),
            ('relu', [True]),
            ('bn', [64]),
            # ('dropout', [.5]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [64, 64, 3, 3, 1, 1]),
            ('relu', [True]),
            ('bn', [64]),
            # ('dropout', [.5]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [64, 64, 3, 3, 1, 1]),
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
            ('bn', [1024]),
            ('linear', [1024, 1024]),
            ('relu', [True]),
            ('bn', [1024]),
            ('linear', [5, 1024])
        ]
    }
}

test_params = {
    "project_dir": str(Path(__file__).parent/"logs"/"expr.[2020-12-22].[18:02:57].[k9gic].[lth_maml].[test_first_order_approx_acc_and_speed]"),
    "train_test": "test",
}

for i, expr_params in enumerate([train_params]):
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> STARTING EXPR {i} <<<<<<<<<<<<<<<<<<<<<<<<<<<")

    project_dir = Path(expr_params["project_dir"])
    logger = utils.Logger(project_dir, project_dir/"logs")
    logger.snapshot(
        expr_id=expr_params["expr_id"], 
        expr_params_JSON=expr_params,
    )
    utils.tee(project_dir/"logs"/expr_id/"program_outputs.txt")

    if expr_params.get("train_test", "train") == "train":
        print(expr_params["expr_id"], "\n")
        print(expr_params)
        utils.set_seeds(expr_params["random_seed"], expr_params.get("cudnn_enabled", True))

        args = expr_params["model_training_params"]

        print(f"starting lth run {expr_params['expr_id']}")
        dataset = load_data.dataset(expr_params["dataset_params"], redownload=False)
        mask = lth.run(dataset, expr_params, logger)
    else:
        log_dir = expr_params["project_dir"]
        with open(Path(log_dir)/"expr_params.json") as expr_params_handle:
            logged_expr_params = json.load(expr_params_handle)
        logged_expr_params.update(expr_params)
        print(logged_expr_params)
        print(logged_expr_params["expr_id"], "\n")

        dataset = load_data.dataset(logged_expr_params["dataset_params"], redownload=False)

        train, val, test = dataset
        log_dir = Path(logged_expr_params["project_dir"])

        print(f"log_dir: {log_dir}")
        print(lth.test_finetuning(test, logged_expr_params, log_dir))
