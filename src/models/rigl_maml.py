from itertools import chain
import torch.nn.functional as F
import torch.nn  as nn
from torch import optim
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import copy
from ..utils import Logger
from .mask_ops import build_mask, apply_mask, update_mask
from .lth import detect_early_bird
from src.models.rigl_meta import RigLMeta
from rigl_torch.RigL import RigLScheduler
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(dataset, rigl_params):
    # Unpack parameters and split data
    prune_strategy = rigl_params['prune_strategy']
    training_params = rigl_params['model_training_params']
    train_data, val_data, test_data = dataset
    
    # Get model and optimizer
    model, optimizer = build_model(training_params)
    
    # Train the model
    return train(model, optimizer, train_data, val_data, prune_strategy, training_params)
    
    
def train(model, optimizer, train_data, val_data, prune_strategy, training_params):
    # Define RigL pruner
    pruner = RigLScheduler(model,
                       optimizer,
                       dense_allocation=prune_strategy['dense_allocation'],
                       sparsity_distribution=prune_strategy['sparsity_distribution'],
                       T_end=int(0.75 * training_params['training_iterations']),
                       delta=prune_strategy['delta'],
                       alpha=prune_strategy['alpha'],
                       grad_accumulation_n=prune_strategy['grad_accumulation_n'],
                       static_topo=prune_strategy['static_topo'],
                       ignore_linear_layers=prune_strategy['ignore_linear_layers'],
                       state_dict=prune_strategy['state_dict']) 
    val_accs = []
    best_val_acc, best_model_state = 0, None
    
    for epoch in range(training_params['training_iterations']//10000):
        print(f"train epoch {epoch}")
        print(f"Time {datetime.datetime.now()}")
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(train_data):
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            
            optimizer.zero_grad()
            accs = model(x_spt, y_spt, x_qry, y_qry)
            if step % 30 == 0:
                print(f" step: {step} \ttraining acc: {accs}")
                
            if step % 300 == 0:  # evaluation
                print("validating model...")
                accs = test(model, val_data)
                print('val acc:', accs)
                val_accs.append(max(accs))
            if max(accs) > best_val_acc:
                best_val_acc = max(accs)
                best_model_state = {n: w.cpu().detach() for n, w in model.state_dict().items()}
            if pruner():
                optimizer.step()
    return val_accs, best_model_state

def test(model, test_data):
    accs_all_test = []
    for x_spt, y_spt, x_qry, y_qry in test_data:
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                    x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

        accs = model.finetunning(x_spt, y_spt, x_qry, y_qry)
        accs_all_test.append(accs)

    # [b, update_step+1]
    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
    return accs

def build_model(training_params):
    layer_definitions = training_params.get("layer_definitions", None)
    model = RigLMeta(training_params, layer_definitions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer