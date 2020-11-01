import torch.nn.functional as F
from tqdm.notebook import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def run(model, dataset, lottery_ticket_params):
    """
    executes a lottery ticket hypothesis run (repeatedly trains, prunes, reinitializes)
    """
    # unpacking inputs
    prune_strategy = lottery_ticket_params["prune_strategy"]
    prune_rate = prune_strategy["rate"]
    train_data, val_data, test_data = dataset

    # saving initial weights
    initial_weights = {n: w.cpu().detach() for n, w in model.named_parameters()}
    mask = build_mask(model)

    # setting up logging
    experiment_logs = []
    writer = SummaryWriter(f'tensorboard/{lottery_ticket_params["expr_id"]}')

    for prune_iter in range(prune_strategy["iterations"]):
        print(f"starting prune iteration {prune_iter}")
        # reinitializing weights
        model.load_state_dict(initial_weights)

        # updating pruning rate and training network to completion
        pruned_rate = 1-(1-prune_rate)**(prune_iter)
        expr_params = {
            "prune_iter": prune_iter, 
            "pruned_rate": pruned_rate, 
            **lottery_ticket_params["model_train_params"]
        }
        val_accs = train(model, mask, train_data, val_data, expr_params, writer)
        
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

    writer.close()
    return mask, experiment_logs

def initialize_optimizer(expr_params, model):
    name, params = expr_params["optimizer"]

    if name == "adam":
        return torch.optim.Adam(model.parameters(), **params)
    else:
        raise Exception(f"optimizer {name} not implemented")

def build_mask(model):
    mask = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, param in model.named_parameters():
        if "weight" not in name:
            continue
        mask[name] = torch.ones_like(param, requires_grad=False, dtype=torch.bool, device="cpu")
    return mask

def apply_mask(model, mask):
    model_named_parameters = dict(model.named_parameters())
    device = next(model.parameters()).device
    for name, param_mask in mask.items():
        model_named_parameters[name] = model_named_parameters[name] * param_mask.to(device)
    model.load_state_dict(model_named_parameters)

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
    else:
        raise Exception(f"prune strategy {prune_strategy} not found")

def train(model, mask, train_data, val_data, expr_params, writer):
    val_accs = []
    optimizer = initialize_optimizer(expr_params, model)
    pass_name = f"{expr_params['prune_iter']}, {1-expr_params['pruned_rate']*100:.3f}% left"
    
    for epoch in tqdm(range(expr_params["training_iterations"])):
        loss, train_acc = train_one_epoch(model, mask, train_data, optimizer, expr_params)
        val_acc = test(model, mask, val_data, expr_params)
        writer.add_scalars("train passes", {pass_name: train_acc}, epoch)
        writer.add_scalars("vals passes", {pass_name: val_acc}, epoch)
        writer.flush()

        val_accs.append(val_acc)
        if len(val_accs) > 1 and val_acc < val_accs[-2] and expr_params["early_stopping"]:
            print(f"early stopping triggered at epoch {epoch}")
            break

    return val_accs

def train_one_epoch(model, mask, train_data, optimizer, expr_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train() # setting model to train mode
    correct = 0
    for batch_idx, (X, y_true) in enumerate(train_data):
        optimizer.zero_grad() # zeroing out the gradients
        X, y_true = X.to(device), y_true.to(device) # sending Xs and ys to gpu

        apply_mask(model, mask)
        y_pred = model(X) # pred ys

        train_loss = expr_params["loss_func"](y_pred, y_true) # getting loss
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
