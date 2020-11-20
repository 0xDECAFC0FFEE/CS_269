from itertools import chain
import torch.nn.functional as F
import torch.nn  as nn
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import copy
from ..utils import Logger
from .mask_ops import build_mask, apply_mask, update_mask
from .lth import detect_early_bird
from src.models.meta import Meta

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
    images_spt, labels_spt, images_qry, labels_qry  = next(iter(train_data))
    input_shape = images_spt.shape[2:]
    
    model = build_model(training_params, input_shape)
    initial_weights = {n: w.cpu().detach() for n, w in model.state_dict().items()}
    mask = build_mask(model.net, prune_strategy)

    # setting up logging
    masks, model_state_dicts = [], []
    val_accs_per_prune_iter = []
    test_accs_per_prune_iter = []

    logger = Logger("/workspace", "/workspace/logs")
    logger.snapshot(
        expr_id=lottery_ticket_params["expr_id"], 
        expr_params_JSON=lottery_ticket_params,
        initial_weights=initial_weights,
        masks=masks,
        model_state_dicts=model_state_dicts,
    )
    writer = SummaryWriter(f'tensorboard/{lottery_ticket_params["expr_id"]}')

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
        val_accs, best_mask_model = train(model, mask, train_data, val_data, expr_params, writer, prune_iter)
        model.load_state_dict(best_mask_model)
        masks.append(copy.deepcopy(mask))
        model_state_dicts.append(best_mask_model)
        val_accs_per_prune_iter.append(val_accs)

        # scoring masked model
        test_accs = test(model, mask, test_data, expr_params)
        for i, test_acc in enumerate(test_accs):
            writer.add_scalars("test acc", {f"prune iteration {prune_iter}": test_acc}, i)
        writer.flush()
        test_accs_per_prune_iter.append(test_accs)

        logger.snapshot(
            expr_id=lottery_ticket_params["expr_id"], 
            initial_weights=initial_weights,
            masks=masks,
            model_state_dicts=model_state_dicts,
            expr_params_JSON=lottery_ticket_params,
            test_accs_TXT="\n".join([str(accs) for accs in test_accs_per_prune_iter]),
            val_accs_TXT="\n".join([str(accs) for accs in val_accs_per_prune_iter]),
        )

        # pruning weights
        next_pass_prune_rate = 1-(1-prune_rate)**(1+prune_iter)

        update_mask(model.net, mask, next_pass_prune_rate, prune_strategy)

        print(f"{prune_iter}. perc_left: {1-pruned_rate}, test_acc {test_accs}")

    writer.close()
    return mask


def build_model(training_params, data_input_shape):
    model_name = training_params["model_name"]
    dataset_name = training_params["dataset_name"]
    layer_definitions = training_params.get("layer_definitions", None)

    if model_name == "MAML" and dataset_name == "mini_imagenet":
        return Meta(training_params, layer_definitions).to(device)
    else:
        raise Exception(f"model {model_name} and dataset {dataset_name} not implemented yet")

def train(model, mask, train_data, val_data, expr_params, writer, prune_iter):
    assert(expr_params["dataset_name"] == "mini_imagenet")
    
    val_accs = []
    best_val_acc, best_model_state = 0, None

    for epoch in list(range(expr_params["training_iterations"]//10000)):
        # fetch meta_batchsz num of episode each time
        print(f"train epoch {epoch}")
        # pbar = tqdm(total=len(train_data), leave=False) # not wrapping the train_data in tqdm as it causes a threading error
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(train_data):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            accs = model(mask, x_spt, y_spt, x_qry, y_qry)
            writer.add_scalars(f"prune {prune_iter} train passes", {f"epoch {epoch}": max(accs)}, step)

            if step % 30 == 0:
                print(f" step: {step} \ttraining acc: {accs}")

            if step % 300 == 0:  # evaluation
                print("validating model...")
                
                accs = test(model, mask, val_data, expr_params)

                print('val acc:', accs)
                writer.add_scalars(f"prune {prune_iter} val passes", {f"epoch {epoch}": max(accs)}, step)
                val_accs.append(max(accs))
                if max(accs) > best_val_acc:
                    best_val_acc = max(accs)
                    best_model_state = {n: w.cpu().detach() for n, w in model.state_dict().items()}

    return val_accs, best_model_state

def test(model, mask, test_data, expr_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accs_all_test = []
    for x_spt, y_spt, x_qry, y_qry in test_data:
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                    x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

        accs = model.finetunning(mask, x_spt, y_spt, x_qry, y_qry)
        accs_all_test.append(accs)

    # [b, update_step+1]
    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
    return accs