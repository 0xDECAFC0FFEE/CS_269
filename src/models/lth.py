from itertools import chain
import torch.nn.functional as F
import torch.nn  as nn
from tqdm.notebook import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import copy
from ..utils import Logger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(dataset, lottery_ticket_params):
    """
    executes a lottery ticket hypothesis run (repeatedly trains n epochs, prunes, reinitializes)
    """
    # unpacking inputs
    prune_strategy = lottery_ticket_params["prune_strategy"]
    training_params = lottery_ticket_params["model_training_params"]
    prune_rate = prune_strategy["rate"]
    train_data, val_data, test_data = dataset

    # building model
    images, labels = next(iter(train_data))
    input_shape = images.shape[1:]
    model = build_model(training_params, input_shape)

    # saving initial model weights
    initial_weights = {n: w.cpu().detach() for n, w in model.state_dict().items()}
    mask = build_mask(model, prune_strategy)

    # setting up logging
    experiment_logs, masks = [], []

    Logger("/workspace", "logs").save_snapshot(
        expr_id=lottery_ticket_params["expr_id"], 
        expr_params=lottery_ticket_params,
        initial_weights=initial_weights,
        experiment_logs=experiment_logs,
    )
    writer = SummaryWriter(f'tensorboard/{lottery_ticket_params["expr_id"]}')

    for prune_iter in range(prune_strategy["iterations"]):
        print(f"starting prune iteration {prune_iter}")
        # reinitializing weights
        model.load_state_dict(initial_weights)

        # getting current pruned rate and training network to completion
        pruned_rate = 1-(1-prune_rate)**(prune_iter)
        expr_params = {
            "prune_iter": prune_iter, 
            "pruned_rate": pruned_rate, 
            **training_params
        }
        val_accs, best_mask_model = train(model, mask, train_data, val_data, expr_params, writer)
        model.load_state_dict(best_mask_model)
        masks.append(copy.deepcopy(mask))

        # scoring masked model
        test_acc = test(model, mask, test_data, expr_params)
        writer.add_scalar("test acc", test_acc, prune_iter)
        writer.flush()

        # pruning weights
        next_pass_prune_rate = 1-(1-prune_rate)**(1+prune_iter)
        update_mask(model, mask, next_pass_prune_rate, prune_strategy)

        # saving experiment to logs (idk if ncessary might want it later for graphing)
        experiment_logs.append({
            "prune_iter": prune_iter,
            "val_accs": val_accs,
            "test_acc": test_acc,
            "prune_rate": pruned_rate,
            "perc_left": 1-pruned_rate
        })

        print(f"{prune_iter}. perc_left: {1-pruned_rate}, test_acc {test_acc}")

        if prune_strategy["name"] == "early_bird":
            if detect_early_bird(masks):
                print("found early bird ticket")
                break

    writer.close()
    Logger("/workspace", "logs").save_snapshot(
        expr_id=lottery_ticket_params["expr_id"], 
        expr_params=lottery_ticket_params,
        initial_weights=initial_weights,
        experiment_logs=experiment_logs,
    )
    return mask, experiment_logs

def detect_early_bird(masks):
    if len(masks) < 2:
        return False
    last_name_mask = masks[-1]
    for prev_name_mask in masks[-2:-6:-1]:
        diffs, size = 0, 0
        assert({name for name in last_name_mask} == {name for name in prev_name_mask})

        for name, lst_mask in last_name_mask.items():
            print(name)
            print(lst_mask)
            print(prev_name_mask[name])
            size += np.prod(lst_mask.shape)
            diffs += torch.sum(lst_mask!=prev_name_mask[name])

        print(f"diffs {diffs}")
        if float(diffs) / size > .1:
            return False

    if len(masks) < 5:
        return False

    return True


def build_model(training_params, data_input_shape):
    model_name = training_params["model_name"]
    dataset_name = training_params["dataset_name"]
    layer_definitions = training_params.get("layer_definitions", None)

    if model_name == "VGG" and dataset_name == "cifar10":
        from src.models.vgg import VGG_cifar10
        return VGG_cifar10(layer_definitions, data_input_shape).to(device)
    elif model_name == "lenet" and dataset_name == "mnist":
        from src.models.lenet import LeNet_mnist
        return LeNet_mnist().to(device)
    else:
        raise Exception(f"model {model_name} and dataset {dataset_name} not implemented yet")


def initialize_optimizer(expr_params, model):
    name, params = expr_params["optimizer"]

    if name == "adam":
        return torch.optim.Adam(model.parameters(), **params)
    else:
        raise Exception(f"optimizer {name} not implemented")

def build_mask(model, prune_strategy):
    mask = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if prune_strategy["name"] in ["local", "global"]:
        for name, param in model.named_parameters():
            if "weight" not in name:
                continue
            mask[name] = torch.ones_like(param, requires_grad=False, dtype=torch.bool, device="cpu")
    elif prune_strategy["name"] == "early_bird":
        for name, param in model.named_parameters():
            if "weight" not in name:
                continue
            mask[name] = torch.ones_like(param, requires_grad=False, dtype=torch.bool, device="cpu")
    else:
        raise Exception(f"prune strategy {prune_strategy['name']} not implemented yet")
    return mask

def apply_mask(model, mask):
    model_named_parameters = dict(model.named_parameters())
    device = next(model.parameters()).device
    for name, param_mask in mask.items():
        model_named_parameters[name] = model_named_parameters[name] * param_mask.to(device)
    model.load_state_dict(model_named_parameters, strict=False) # strict is false as we don't want to add buffers

def update_mask(model, mask, prune_rate, prune_strategy):
    """
    prunes model*mask weights at rate prune_rate and updates the mask.
    """

    if prune_strategy["name"] == "local":
        apply_mask(model, mask)
        for name, param in model.named_parameters():
            if "weight" not in name:
                continue
            # get magnitudes of weight matrices. ignores bias.
            weight_magnitudes = param.flatten().cpu().detach().numpy().astype(np.float64)
            weight_magnitudes = np.random.normal(scale=1e-45, size=weight_magnitudes.shape)
            weight_magnitudes = np.abs(weight_magnitudes)

            # gets the kth weight
            num_weights = len(weight_magnitudes)
            k = int(num_weights*prune_rate)
            kth_weight = np.partition(weight_magnitudes, k)[k]

            # updating mask
            mask[name] = (param.abs() > kth_weight).cpu()
            num_equal = (param.abs() == kth_weight).sum()
            if num_equal > 100:
                raise Exception(f"{num_equal} parameters have the same magnitude {kth_weight} - use iter prune strategy")
            elif num_equal > 1:
                print(f"warning: {num_equal} parameters have the same magnitude {kth_weight}")
    elif prune_strategy["name"] == "global":
        # get magnitudes of weight matrices. ignores bias.
        apply_mask(model, mask)
        layer_weights = [(name, param) for name, param in model.named_parameters() if "weight" in name]
        weight_magnitudes = [param.flatten().cpu().detach().numpy().astype(np.float64) for name, param in layer_weights]
        weight_magnitudes = np.concatenate(weight_magnitudes)
        weight_magnitudes = np.abs(weight_magnitudes + np.random.normal(scale=1e-39, size=weight_magnitudes.shape))

        # gets the kth weight
        num_weights = len(weight_magnitudes)
        k = int(num_weights*prune_rate)
        kth_weight = np.partition(weight_magnitudes, k)[k]

        # updating mask
        num_equal = 0
        for name, parameter in model.named_parameters():
            if "weight" in name:
                mask[name] = (parameter.abs() > kth_weight).cpu()
                num_equal += (parameter.abs() == kth_weight).sum()
        if num_equal > 100:
            raise Exception(f"{num_equal} parameters have the same magnitude {kth_weight} - use iter prune strategy")
        elif num_equal > 1:
                print(f"warning: {num_equal} parameters have the same magnitude {kth_weight}")
    elif prune_strategy["name"] == "early_bird":
        # get magnitudes of weight matrices. ignores bias.
        apply_mask(model, mask)

        bn_layers = []
        for bn_layer_name, w in model.named_children():
            if isinstance(w, torch.nn.BatchNorm2d):
                bn_layers.append((f"{bn_layer_name}.weight", w.weight))

        weight_magnitudes = [param.flatten().cpu().detach().numpy().astype(np.float64) for name, param in bn_layers]
        weight_magnitudes = np.concatenate(weight_magnitudes)
        weight_magnitudes = np.abs(weight_magnitudes + np.random.normal(scale=1e-39, size=weight_magnitudes.shape))

        # gets the kth weight
        num_weights = len(weight_magnitudes)
        k = int(num_weights*prune_rate)
        kth_weight = np.partition(weight_magnitudes, k)[k]

        # updating mask
        num_equal = 0
        for bn_layer_name, w in model.named_children():
            if isinstance(w, torch.nn.BatchNorm2d):
                mask[f"{bn_layer_name}.weight"] = (w.weight.abs() > kth_weight).cpu()
                num_equal += (w.weight.abs() == kth_weight).sum()

        if num_equal > 100:
            raise Exception(f"{num_equal} parameters have the same magnitude {kth_weight} - use iter prune strategy")
        elif num_equal > 1:
                print(f"warning: {num_equal} parameters have the same magnitude {kth_weight}")
    else:
        raise Exception(f"prune strategy {prune_strategy} not found")

def train(model, mask, train_data, val_data, expr_params, writer):
    apply_mask(model, mask)
    val_accs = []
    best_val_acc, best_model_state = 0, {}
    optimizer = initialize_optimizer(expr_params, model)
    pass_name = f"{expr_params['prune_iter']}, {(1-expr_params['pruned_rate'])*100:.3f}% left"
    if expr_params["loss_func"] == "cross_entropy":
        loss_func = nn.CrossEntropyLoss()
    else:
        raise Exception(f"loss func {expr_params['loss_func']} not implemeted")
    
    for epoch in tqdm(range(expr_params["training_iterations"])):
        loss, train_acc = train_one_epoch(model, mask, train_data, optimizer, expr_params, loss_func)
        val_acc = test(model, mask, val_data, expr_params)
        writer.add_scalars("train passes", {pass_name: train_acc}, epoch)
        writer.add_scalars("vals passes", {pass_name: val_acc}, epoch)
        writer.flush()

        # record best score
        val_accs.append(val_acc)
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_model_state = {n: w.cpu().detach() for n, w in model.state_dict().items()}
        if ["early_stopping"] and len(val_accs) >= 5:
            for i in range(-2, -6, -1):
                if val_accs[i] < val_accs[-1]:
                    break
            else:
                print(f"early stopping triggered at epoch {epoch}")
                break

    return val_accs, best_model_state

def train_one_epoch(model, mask, train_data, optimizer, expr_params, loss_func):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train() # setting model to train mode
    correct = 0
    for batch_idx, (X, y_true) in enumerate(train_data):
        optimizer.zero_grad() # zeroing out the gradients
        X, y_true = X.to(device), y_true.to(device) # sending Xs and ys to gpu

        apply_mask(model, mask)
        y_pred = model(X) # pred ys
        train_loss = loss_func(y_pred, y_true) # getting loss

        train_loss.backward() # backpropagating the loss
        optimizer.step() # incrementing step in optimizer

        y_pred = y_pred.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += y_pred.eq(y_true.data.view_as(y_pred)).sum().item()
    accuracy = correct / len(train_data.dataset)
    return train_loss.item(), accuracy

def test(model, mask, test_data, expr_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval() # test mode
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y_true in test_data:
            X, y_true = X.to(device), y_true.to(device)
            apply_mask(model, mask)
            y_pred = model(X)
            y_pred = y_pred.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += y_pred.eq(y_true.data.view_as(y_pred)).sum().item()
        accuracy = correct / len(test_data.dataset)
    return accuracy
