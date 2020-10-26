import shutil
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from tqdm.notebook import tqdm

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


def train_epoch(model, train_loader, optimizer, loss_func):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train() # setting model to train mode
    for batch_idx, (imgs, targets) in tqdm(enumerate(train_loader), leave=False):
        optimizer.zero_grad() # zeroing out the gradients
        imgs, targets = imgs.to(device), targets.to(device) # sending Xs and ys to gpu
        output = model(imgs) # pred ys
        train_loss = loss_func(output, targets) # getting loss
        train_loss.backward() # backpropagating the loss

        # # Freezing Pruned weights by making their gradients Zero
        # for name, p in model.named_parameters():
        #     if 'weight' in name:
        #         tensor = p.data.cpu().numpy()
        #         grad_tensor = p.grad.data.cpu().numpy()
        #         grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
        #         p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizer.step() # incrementing step in optimizer
    return train_loss.item()

# Function for Testing
def test_epoch(model, test_loader, loss_func):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval() # test mode
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    print(f"test accuracy {accuracy}")
    return accuracy
