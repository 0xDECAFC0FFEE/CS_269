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
from src.models.sparse_meta import SparseMeta
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(dataset, expr_params):
    # Unpack parameters and split data
    training_params = expr_params['model_training_params']
    train_data, val_data, test_data = dataset
    
    # Get model
    model = build_model(training_params)
    
    # Train the model
    return train(model, train_data, val_data, training_params)
    
    
def train(model, train_data, val_data, training_params):
    val_accs = []
    best_val_acc, best_model_state = 0, None
    
    for epoch in range(training_params['training_iterations']//10000):
        print(f"train epoch {epoch}")
        print(f"Time {datetime.datetime.now()}")
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(train_data):
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            
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
    return val_accs, best_model_state


def test(model, test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model = SparseMeta(training_params, layer_definitions).to(device)
    return model