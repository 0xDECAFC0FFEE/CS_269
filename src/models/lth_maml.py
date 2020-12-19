from tqdm import tqdm
import numpy as np
np.core.arrayprint._line_width = 270
import torch
torch.set_printoptions(linewidth=270)
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from unittest.mock import Mock as SummaryWriter
import copy
from ..utils import Logger
from .mask_ops import build_mask, apply_mask, update_mask
from .lth import detect_early_bird
from src.models.meta import Meta
from pathlib import Path
import pickle
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(dataset, lottery_ticket_params):
    """
    executes a lottery ticket hypothesis run (repeatedly trains n epochs, prunes, reinitializes)
    """

    # unpacking inputs
    prune_strategy = lottery_ticket_params["prune_strategy"]
    training_params = lottery_ticket_params["model_training_params"]
    dataset_params = lottery_ticket_params["dataset_params"]
    prune_rate = prune_strategy["rate"]
    train_data, val_data, test_data = dataset

    # building model
    # torch.set_default_tensor_type(get_dtype(training_params))
    images_spt, labels_spt, images_qry, labels_qry  = next(iter(train_data))
    input_shape = images_spt.shape[2:]
    
    model = build_model(training_params, dataset_params, input_shape)
    initial_weights = {n: w.cpu().detach() for n, w in model.state_dict().items()}
    mask = build_mask(model.net, prune_strategy)

    # setting up logging
    masks, model_state_dicts = [], []
    train_accs_per_prune_iter = []
    val_accs_per_prune_iter = []
    test_accs_per_prune_iter = []
    epoch_runtimes_per_prune_iter = []

    project_dir = Path(lottery_ticket_params["project_dir"])
    logger = Logger(project_dir, project_dir/"logs")
    
    writer = SummaryWriter(log_dir=f'tensorboard/{lottery_ticket_params["expr_id"]}')
    
    for prune_iter in range(prune_strategy["iterations"]):
        print(f"========================\nstarting prune iteration {prune_iter}\n========================")
        # reinitializing weights
        model.load_state_dict(initial_weights)

        # getting current pruned rate and training network to completion
        pruned_rate = 1-(1-prune_rate)**(prune_iter)
        expr_params = {
            "prune_iter": prune_iter, 
            "pruned_rate": pruned_rate, 
            **training_params
        }
        train_accs, val_accs, best_mask_model, epoch_runtimes = train(model, mask, train_data, val_data, expr_params, writer, prune_iter)
        train_accs_per_prune_iter.append({"prune_iter": prune_iter, "prune_rate": pruned_rate, "train_accs": train_accs})
        epoch_runtimes_per_prune_iter.append(epoch_runtimes)
        model.load_state_dict(best_mask_model)
        masks.append(copy.deepcopy(mask))
        model_state_dicts.append(best_mask_model)
        val_accs_per_prune_iter.append({"prune_iter": prune_iter, "prune_rate": pruned_rate, "val_accs": list(val_accs)})

        # scoring masked model
        test_accs = test(model, mask, test_data, training_params)
        
        max_test_acc, max_test_acc_epoch = 0, 0
        for i, test_acc in enumerate(test_accs):
            if test_acc > max_test_acc:
                max_test_acc_epoch = i
                max_test_acc = test_acc

            writer.add_scalars("test acc", {f"prune iteration {prune_iter}": test_acc}, i)
        
        writer.add_scalars("max test epoch per prune iter", {lottery_ticket_params["expr_id"]: max_test_acc_epoch}, prune_iter)
        writer.add_scalars("max test acc per prune iter", {lottery_ticket_params["expr_id"]: max_test_acc}, prune_iter)
        
        early_stop_epoch = 0
        for i in range(len(test_accs)-1):
            if test_accs[i] > test_accs[i+1]:
                early_stop_epoch = i
                early_stop_acc = test_accs[i]
                break
        else:
            early_stop_acc = test_accs[-1]
            early_stop_epoch = len(test_accs)-1

        writer.add_scalars("early stop epoch", {lottery_ticket_params["expr_id"]: early_stop_epoch}, prune_iter)
        writer.add_scalars("early stop acc", {lottery_ticket_params["expr_id"]: early_stop_acc}, prune_iter)

        writer.flush()
        test_accs_per_prune_iter.append({"prune_iter": prune_iter, "prune_rate": pruned_rate, "test_accs": list(test_accs)})

        logger.snapshot(
            expr_id=lottery_ticket_params["expr_id"], 
            initial_weights=initial_weights,
            masks=masks,
            model_state_dicts=model_state_dicts,
            expr_params_JSON=lottery_ticket_params,
            train_accs_JSON=train_accs_per_prune_iter,
            test_accs_JSON=test_accs_per_prune_iter,
            val_accs_JSON=val_accs_per_prune_iter,
            epoch_runtimes_TXT=str(epoch_runtimes_per_prune_iter),
            prune_iterations_TXT=str(prune_iter+1)
        )

        # pruning weights
        next_pass_prune_rate = 1-(1-prune_rate)**(1+prune_iter)

        update_mask(model.net, mask, next_pass_prune_rate, prune_strategy)

        print(f"{prune_iter}. perc_left: {1-pruned_rate}, test_acc {test_accs}")

    writer.close()
    return mask

def get_dtype(training_params):
    assert(training_params["dtype"] == "float32")
    dtype = training_params.get("dtype", "float32")
    if dtype == "float64":
        return torch.DoubleTensor
    elif dtype == "float32":
        return torch.FloatTensor
    elif dtype == "float16":
        return torch.HalfTensor
    return None

def build_model(training_params, dataset_params, data_input_shape):
    model_name = training_params["model_name"]
    dataset_name = dataset_params["dataset_name"]
    layer_definitions = training_params.get("layer_definitions", None)

    if model_name == "MAML" and dataset_name == "mini_imagenet":
        return Meta(training_params, dataset_params, layer_definitions).to(device)
    if model_name == "MAML" and dataset_name == "mini_mini_imagenet":
        return Meta(training_params, dataset_params, layer_definitions).to(device)
    else:
        raise Exception(f"model {model_name} and dataset {dataset_name} not implemented yet")

def train(model, mask, train_data, val_data, expr_params, writer, prune_iter):
    train_accs, val_accs = [], []
    epoch_runtimes = []
    best_val_acc, best_model_state = 0, {}
    prev_acc = 0

    for epoch in list(range(expr_params["meta_training_epochs"])):
        # fetch meta_batchsz num of episode each time
        print(f"train epoch {epoch}")
        # pbar = tqdm(total=len(train_data), leave=False) # not wrapping the train_data in tqdm as it causes a threading error
        start_time = datetime.now()
        epoch_val_accs = []
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(train_data):

            x_spt, y_spt = x_spt.to(device), y_spt.to(device)
            x_qry, y_qry = x_qry.to(device), y_qry.to(device)

            accs = model(mask, x_spt, y_spt, x_qry, y_qry)
            # writer.add_scalars(f"prune {prune_iter} train passes", {f"epoch {epoch}": max(accs)}, step)

            if step % 30 == 0:
                train_accs.append({"epoch": epoch, "step": step, "accs": list(accs)})
                print(f" step: {step} \ttraining acc: {accs}")

            if step % 500 == 0:  # evaluation
                print("validating model...")

                accs = test(model, mask, val_data, expr_params)

                print('val acc:', accs)
                writer.add_scalars(f"prune {prune_iter} val passes", {f"epoch {epoch}": max(accs)}, step)
                epoch_val_accs.append(max(accs))
                if max(accs) > best_val_acc:
                    best_val_acc = max(accs)
                    best_model_state = {n: w.cpu().detach() for n, w in model.state_dict().items()}
        val_accs.extend(epoch_val_accs)

        runtime = datetime.now()-start_time
        epoch_runtimes.append(runtime.total_seconds())
        print(f"epoch time length: {runtime}")

        if expr_params.get("meta_training_early_stopping", False) and prev_acc > max(epoch_val_accs):
            print("early stopping triggered; stopping")
            break
        else:
            prev_acc = max(epoch_val_accs)

    return train_accs, val_accs, best_model_state, epoch_runtimes

def test(model, mask, test_data, expr_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accs_all_test = []

    for x_spt, y_spt, x_qry, y_qry in test_data:
        x_spt, y_spt = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device)
        x_qry, y_qry = x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

        accs = model.finetunning(mask, x_spt, y_spt, x_qry, y_qry)
        accs_all_test.append(accs)

    # [b, update_step+1]
    accs = np.array(accs_all_test).mean(axis=0)
    return accs

def test_finetuning(test_data, lottery_ticket_params, log_dir):
    prune_strategy = lottery_ticket_params["prune_strategy"]
    training_params = lottery_ticket_params["model_training_params"]
    dataset_params = dataset_params["dataset_params"]

    # building model
    images_spt, labels_spt, images_qry, labels_qry  = next(iter(test_data))
    input_shape = images_spt.shape[2:]
    
    model = build_model(training_params, dataset_params, input_shape)

    with open(log_dir/"model_state_dicts.pkl", "rb") as weight_handle:
        state_dicts = pickle.load(weight_handle)
    with open(log_dir/"masks.pkl", "rb") as mask_handle:
        masks = pickle.load(mask_handle)

    accs = []
    for state_dict, mask in tqdm(zip(state_dicts, masks)):
        model.load_state_dict(state_dict)
        acc = test(model, mask, test_data, lottery_ticket_params["model_training_params"])
        print(acc)
        accs.append(accs)
    
    return accs