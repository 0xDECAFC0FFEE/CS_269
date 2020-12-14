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
import pickle
from numpy.lib.format import open_memmap

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


def new_expr_id(*args):
    """
    returns new experiemnt id for process.
    """
    chars = "abcdefghijklmnopqrstuvwxyz"
    nums = "1234567890"
    nonce = random.choices(chars+chars.upper()+nums, k=5)
    nonce = "".join(nonce)

    time = datetime.datetime.now().strftime("%Y-%m-%d|%H:%M:%S")

    return "|".join(["expr", time, nonce, *args])


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

        if not self.log_folder.exists():
            os.makedirs(self.log_folder)

        assert(self.project_folder.exists() and self.project_folder.is_dir())

    def zip_logs(self):
        for file in os.listdir(self.log_folder):
            if (self.log_folder/file).is_dir() and not (self.log_folder/(file+".zip")).exists():
                shutil.make_archive(self.log_folder/file, "zip", self.log_folder/file)
                shutil.rmtree(self.log_folder/file)

    def save_snapshot(self, expr_id, **kwargs):
        """
        saves a copy of the directory project_folder and any kwargs in log_folder/expr_id

        - raises exception if directory already exists
        - auto skips saving files not tracked by git
        - auto skips saving anything in log_folder (don't want to recursively copy everything) 
        - arguments that end in "_JSON" are saved as json files
        - arguments that end in "_TXT" are saved as text files
        - by default, arguments are saved as pickle files

        logger = Logger("/project_dir", "/project_dir/logs")
        logger.save_snapshot(
            "experiment 5", 
            training_params_JSON=training_params,
            testing_accs_TXT=testing_accs,
            model_parameters=model_parameters
        )
        # do stuff
        logger.update_snapshot(
            "experiment 5", 
            training_params_JSON=training_params,
            testing_accs_TXT=testing_accs,
            model_parameters=model_parameters
        )
        """
        log_folder = self.log_folder/expr_id
        
        if log_folder.exists():
            raise Exception(f"log folder {log_folder} already exists")
        
        os.makedirs(log_folder)
        files_to_log = subprocess.check_output(["git", "ls-files"]).decode().split("\n")
        
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

        self.update_snapshot(expr_id, **kwargs)

    def update_snapshot(self, expr_id, **kwargs):
        """
        updates a log snapshot that exists with values in the kwargs.

        - arguments that end in "_JSON" are saved as json files
        - arguments that end in "_TXT" are saved as text files
        - by default, arguments are saved as pickle files
        - will save argument values in files based off of the parameter names

        logger = Logger("/project_dir", "/project_dir/logs")
        logger.save_snapshot("experiment 5")
        logger.update_snapshot(
            "experiment 5", 
            training_params_JSON=training_params,
            testing_accs_TXT=testing_accs,
            model_parameters=model_parameters
        )
        """
        log_folder = self.log_folder/expr_id
        
        if not log_folder.exists():
            raise Exception(f"log folder {log_folder} doesn't exist - call save_snapshot first")

        for name, value in kwargs.items():
            json_flag = "_JSON"
            txt_flag = "_TXT"
            
            if name[-len(json_flag):] == json_flag:
                name = name[:-len(json_flag)]
                with open(log_folder/f"{name}.json", "w+") as handle:
                    json.dump(value, handle)
            elif name[-len(txt_flag):] == txt_flag:
                name = name[:-len(txt_flag)]
                with open(log_folder/f"{name}.txt", "w+") as handle:
                    handle.write(value)
            else:
                with open(log_folder/f"{name}.pkl", "wb+") as handle:
                    pickle.dump(value, handle)

    def snapshot(self, expr_id, **kwargs):
        """
        snapshots a directory by saving project_dir directory and saving any kwargs

        first call with a specific expr_id will copy the project_dir directory in that log directory
        subsequent calls will only update the log directory with the kwarg values
        see update_snapshot for kwargs naming scheme

        logger = Logger("/project_dir", "/project_dir/logs")
        logger.snapshot(
            "experiment 5", 
            training_params_JSON=training_params,
            testing_accs_TXT=testing_accs,
            model_parameters=model_parameters
        )
        """
        log_folder = self.log_folder/expr_id

        if not log_folder.exists():
            self.save_snapshot(expr_id, **kwargs)
        else:
            self.update_snapshot(expr_id, **kwargs)

def set_seeds(seeds):
    torch.manual_seed(seeds.get("torch", 222))
    torch.cuda.manual_seed_all(seeds.get("cuda", 222))
    np.random.seed(seeds.get("numpy", 222))

class fs_greedy_load:
    """
    greedily loads everything in lst_arrays and stores it as a memory mapped numpy file
    on second run, loads numpy file instead to save ram
    """
    def __init__(self, path, lst_array=None):
        try:
            self.array = np.load(path, mmap_mode="r")
        except FileNotFoundError:
            ex_val = next(iter(lst_array))
            shape, dtype = (len(lst_array), *ex_val.shape), str(ex_val.dtype)

            self.array = open_memmap(path, mode='w+', dtype=dtype, shape=shape)
            for i, val in enumerate(lst_array):
                self.array[i] = val

    def __getitem__(self, index):
        return self.array[index]

    def __len__(self):
        return len(self.array)
    
def sparsity(model, threshold=0.001):
    state_dict = model
    num_params = sum([np.prod(weights.shape) for n, weights in state_dict.items() ] )
    zeros = sum([torch.sum(torch.abs(weights) < threshold).cpu() for n, weights in state_dict.items() ] )
    return zeros / num_params

if __name__ == "__main__":
    a = fs_greedy_load("array.npy", [np.arange(1000).reshape(2, 5) for _ in range(100000)])
    print(len(a))
    print(a[0])
    del a
    a = fs_greedy_load("array.npy")
    print(len(a))
    print(a[0])
    