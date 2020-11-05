import torch
import numpy as np
from collections import OrderedDict

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
    named_parameters = model.named_parameters()
    named_parameters = apply_mask_state_dict(named_parameters, mask)
    model.load_state_dict(named_parameters, strict=False) # strict is false as we don't want to add buffers

def apply_mask_state_dict(named_parameters, mask):
    named_parameters = OrderedDict(named_parameters)
    device = None
    for name, param_mask in mask.items():
        if device == None:
            device = named_parameters[name].device
        named_parameters[name] = named_parameters[name] * param_mask.to(device)
    return named_parameters

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
