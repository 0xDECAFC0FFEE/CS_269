import shutil
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import random
import datetime
from pathlib import Path
import subprocess
import pickle
import json
import torch
import numpy as np

class TopModelSaver():
    def __init__(self, location, config):
        self.prev_best = -np.inf
        
        self.root_folder = location
        if self.root_folder.exists():
            shutil.rmtree(self.root_folder)
        self.model_weights_path = self.root_folder/"model_weights.h5py"
        self.config_path = self.root_folder/"config.json"
        self.source_code_path = self.root_folder/Path(config["file_loc"]).name
        self.saved_config = config

    def reset(self):
        self.prev_best = -np.inf

    def save_best(self, model, score):
        """
        saves best model according to score
        """

        if score > self.prev_best:
            print(f"new best score: {score}; saving weights @ {self.root_folder}")
            if not self.root_folder.exists():
                os.makedirs(self.root_folder)
                with open(self.config_path, "w+") as fp_handle:
                    json.dump(self.saved_config, fp_handle)
                shutil.copyfile(self.saved_config["file_loc"], self.source_code_path)

            model.save_weights(str(self.model_weights_path), save_format="h5")
            self.prev_best = score
        else:
            print(f"cur score {score}. best score remains {self.prev_best}; not saving weights")


def flatten(iterable, max_depth=np.inf):
    """recursively flattens all iterable objects in iterable.

    Args:
        iterable (iterable or numpy array): iterable to flatten
        max_depth (int >= 0, optional): maximum number of objects to iterate into. Defaults to infinity.

    >>> flatten(["01", [2, 3], [[4]], 5, {6:6}.keys(), np.array([7, 8])])
    ['0', '1', 2, 3, 4, 5, 6, 7, 8]

    >>> utils.flatten(["asdf"], max_depth=0)
    ['asdf']

    >>> utils.flatten(["asdf"], max_depth=1)
    ['a', 's', 'd', 'f']
    """
    def recursive_step(iterable, max_depth):
        if max_depth == -1:
            yield iterable
        elif type(iterable) == str:
            for item in iterable:
                yield item
        elif type(iterable) == np.ndarray:
            for array_index in iterable.flatten():
                for item in recursive_step(array_index, max_depth=max_depth-1):
                    yield item
        else:
            try:
                iterator = iter(iterable)
                for sublist in iterator:
                    for item in recursive_step(sublist, max_depth=max_depth-1):
                        yield item
            except (AttributeError, TypeError):
                yield iterable

    assert(max_depth >= 0)
    return recursive_step(iterable, max_depth)


def new_expr_id(prepend=""):
    """
    returns new experiemnt id for process.
    """
    chars = "abcdefghijklmnopqrstuvwxyz"
    nums = "1234567890"

    nonce = random.choices(chars+chars.upper()+nums, k=5)
    nonce = "".join(nonce)
    time = datetime.datetime.now().strftime("%Y-%m-%d | %H:%M:%S")

    return f"expr {time} {prepend} {nonce}"

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

class Logger:
    def __init__(self, project_folder=".", log_folder="./logs"):
        self.project_folder = Path(project_folder).resolve()
        self.log_folder = Path(log_folder).resolve()

    def save_snapshot(self, expr_id, **kwargs):
        """
        saves a copy of the directory project_folder in log_folder/expr_id
        
        - auto skips saving files not tracked by git
        - auto skips saving anything in log_folder (don't want to recursively copy everything) 
        - saves all named arguments as picked files

        ex: 
            l = Logger("/home/user/workspace", "./logs")
            l.save_snapshot(str(datetime.now()))
        """
        log_folder = self.log_folder/expr_id
        
        if log_folder.exists():
            shutil.rmtree(log_folder)
        
        os.makedirs(log_folder)
        files_to_log = subprocess.check_output(["git", "ls-files", "--full-name", self.project_folder]).decode().split("\n")
        
        for path in files_to_log:
            if len(path.strip()) == 0: # stripping newlines
                continue

            src = self.project_folder/path
            dest = log_folder/path
            if str(src.resolve()).startswith(str(self.log_folder)+os.sep):
                continue

            if not dest.parent.exists():
                os.makedirs(dest.parent)

            shutil.copy(src, dest, follow_symlinks=False)

        for name, value in kwargs.items():
            with open(log_folder/f"{name}.pkl", "wb+") as handle:
                pickle.dump(value, handle)

def set_seeds(seeds):
    torch.manual_seed(seeds.get("torch", 222))
    torch.cuda.manual_seed_all(seeds.get("cuda", 222))
    np.random.seed(seeds.get("numpy", 222))