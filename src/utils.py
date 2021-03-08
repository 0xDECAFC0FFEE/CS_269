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
from pathlib import Path
from tqdm import tqdm
import csv

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
    nonce = random.choices(chars+nums, k=5)
    nonce = "".join(nonce)

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    time = datetime.datetime.now().strftime("%H:%M:%S")

    args = [arg.replace(" ", "_") for arg in args]

    expr_id = ".".join([f"[{v}]" for v in [date, time, nonce, *args]])
    expr_id = f"expr.{expr_id}"
    return expr_id, nonce


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
        - see update_snapshot for kwargs naming scheme
        
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

    def update_snapshot(self, expr_id, files_to_update=None,**kwargs):
        """
        updates a log snapshot that exists with values in the kwargs.

        - kwarg naming scheme:
            - will save argument values in files based off of the parameter names
            - arguments that end in "_JSON" are saved as json files
            - arguments that end in "_TXT" are saved as text files
            - arguments that end in "_CSV" are saved as csv files
            - models that end in "_PYTORCH" are saved with torch.save
            - by default, arguments are saved as pickle files
        
        file paths in files_to_update will be updated. files in the project directory not in
        files_to_update will not be updated if they already exist in the log directory

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
            csv_flag = "_CSV"
            pytorch_flag = "_PYTORCH"
            
            try:
                if name[-len(json_flag):] == json_flag:
                    name = name[:-len(json_flag)]
                    with open(log_folder/f"{name}.json", "w+") as handle:
                        json.dump(value, handle)
                elif name[-len(txt_flag):] == txt_flag:
                    name = name[:-len(txt_flag)]
                    with open(log_folder/f"{name}.txt", "w+") as handle:
                        handle.write(value)
                elif name[-len(csv_flag):] == csv_flag:
                    name = name[:-len(csv_flag)]
                    with open(log_folder/f"{name}.csv", "w+") as handle:
                        writer = csv.DictWriter(handle, fieldnames=value[0].keys())
                        writer.writeheader()
                        writer.writerows(value)
                elif name[-len(pytorch_flag):] == pytorch_flag:
                    name = name[:-len(pytorch_flag)]
                    torch.save(value, log_folder/f"{name}.pth")
                else:
                    with open(log_folder/f"{name}.pkl", "wb+") as handle:
                        pickle.dump(value, handle)
            except TypeError as e:
                print(f"triggered on key {name}")
                raise TypeError(e)
            
            if files_to_update != None:
                for path in files_to_update:
                    src = self.project_folder/path
                    dest = log_folder/path
                    if str(src.resolve()).startswith(str(self.log_folder)+os.sep):
                        continue

                    if not dest.parent.exists():
                        os.makedirs(dest.parent)

                    shutil.copy(src, dest, follow_symlinks=False)

    def snapshot(self, expr_id, files_to_update=None, **kwargs):
        """
        snapshots a directory by saving project_dir directory and saving any kwargs

        first call with a specific expr_id will copy the project_dir directory in that log directory
        subsequent calls will update the log directory with the kwarg key values
        see update_snapshot for kwargs naming scheme

        note that subsequent calls to snapshot won't update files in the log directory unless
        it is in the files_to_update list

        logger = Logger("/project_dir", "/project_dir/logs")
        logger.snapshot(
            "experiment 5", 
            files_to_update=["program_outputs.txt", "current_model.pkl"],
            training_params_JSON=training_params,
            testing_accs_TXT=testing_accs,
            model_parameters=model_parameters
        )
        """
        log_folder = self.log_folder/expr_id

        if not log_folder.exists():
            self.save_snapshot(expr_id, **kwargs)
        else:
            self.update_snapshot(expr_id, files_to_update=files_to_update, **kwargs)

def set_seeds(seed, cudnn_enabled=True):
    if seed != None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        
        if not cudnn_enabled:
            torch.backends.cudnn.enabled = False
        else:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

class fs_greedy_load:
    """
    greedily loads everything in lst_arrays and stores it as a chunked memory mapped numpy file
    on second run, loads numpy file instead to save ram

    zarr is slow as shit wtf
    """
    def __init__(self, path, lst_array=None):
        try:
            file_chunks = sorted(list(path.iterdir()))
            self.array_chunks = [np.load(file_chunk, mmap_mode="r+") for file_chunk in file_chunks]
            self.chunk_size = len(self.array_chunks[0])
        except FileNotFoundError:
            print("rebuilding transformed dataset cache")
            shape, dtype = lst_array.shape, str(lst_array.dtype)

            arm32_max_filesize = 2*10**9
            total_bytes = lst_array.dtype.itemsize * np.product(shape)
            num_chunks = int(total_bytes/arm32_max_filesize)+1
            self.chunk_size = int(len(lst_array)/num_chunks)

            os.makedirs(path)
            self.array_chunks = []
            for chunk_index, lst_chunk in enumerate(np.array_split(lst_array, num_chunks)):
                chunk_array = open_memmap(path/f"chunk_{chunk_index}.npy", mode='w+', dtype=dtype, shape=lst_chunk.shape)
                self.array_chunks.append(chunk_array)
                
                for i, val in enumerate(lst_chunk):
                    chunk_array[i] = val

    def __getitem__(self, index):
        chunk_index = index // self.chunk_size
        array_index = index % self.chunk_size
        return self.array_chunks[chunk_index][array_index]

    def __len__(self):
        return sum([len(chunk) for chunk in self.array_chunks])

def DummySummaryWriter(*args, **kwargs):
    from unittest.mock import Mock
    return Mock()

def sparsity(model, threshold=0.001):
    state_dict = model
    num_params = sum([np.prod(weights.shape) for n, weights in state_dict.items() ] )
    zeros = sum([torch.sum(torch.abs(weights) < threshold).cpu() for n, weights in state_dict.items() ] )
    return zeros / num_params

class tee:
    def __init__(self, filename):
        """redirects output to file filename and terminal at the same time

        tee("output.log")
        instantiating will automatically print output to terminal and output.log
        plays well with tqdm & Logger
        make sure to add the redirected filename to files_to_update in logger.snapshot()

        Args:
            filename (str): filename to redirect output to
        """
        import sys
        from pathlib import Path

        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(filename, "w+")
        
        self.terminal = sys.stdout
        sys.stdout = self

        print(f"T piping output to stdout and {filename}")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()

        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

if __name__ == "__main__":
    a = fs_greedy_load("fs_greedy_load_test", [np.arange(1000, dtype=np.float32).reshape(5, 2, 5, 20) for _ in range(1000000)])
    print(len(a))
    print(a[0])
    del a
    a = fs_greedy_load("fs_greedy_load_test")
    print(len(a))
    print(a[0])
    