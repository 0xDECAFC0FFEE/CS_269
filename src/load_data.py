import torchvision
import torch
from pathlib import Path
from .MiniImagenet import MiniImagenet
import gdown
import os
import shutil
import pickle
from tqdm import tqdm
import numpy as np
from .utils import fs_greedy_load

def cifar10(args):
    dataset_path = Path(args.get("dataset_location", 'data/CFAR10'))

    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train = torchvision.datasets.CIFAR10(dataset_path, train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(dataset_path, train=False, download=True, transform=transform)

    n_train = int(len(train)*.8)
    n_val = len(train)-n_train
    train, val = torch.utils.data.random_split(train, [n_train, n_val])

    train = torch.utils.data.DataLoader(train, batch_size=args["batch_size"], shuffle=True, num_workers=0,drop_last=False, pin_memory=True)
    val = torch.utils.data.DataLoader(val, batch_size=args["batch_size"], shuffle=True, num_workers=0,drop_last=False, pin_memory=True)
    test = torch.utils.data.DataLoader(test, batch_size=args["batch_size"], shuffle=False, num_workers=0,drop_last=True, pin_memory=True)

    return train, val, test

def mnist(args):
    dataset_path = Path(args.get("dataset_location", 'data/MNIST'))

    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train = torchvision.datasets.MNIST(dataset_path, train=True, download=True,transform=transform)
    test = torchvision.datasets.MNIST(dataset_path, train=False, download=True,transform=transform)

    n_train = int(len(train)*.8)
    n_val = len(train)-n_train
    train, val = torch.utils.data.random_split(train, [n_train, n_val])
    
    train = torch.utils.data.DataLoader(train, batch_size=args["batch_size"], shuffle=True, num_workers=0,drop_last=False, pin_memory=True)
    val = torch.utils.data.DataLoader(val, batch_size=args["batch_size"], shuffle=True, num_workers=0,drop_last=False, pin_memory=True)
    test = torch.utils.data.DataLoader(test, batch_size=args["batch_size"], shuffle=False, num_workers=0,drop_last=True, pin_memory=True)

    return train, val, test

def download_raw(dataset_path, saved_image_zip_file=None):
    """
    downloads raw maml jpeg images and csv train-val-test split files
    """
    print("downloading mini imagenet dataset (2.9 Gb)... ")

    os.makedirs(dataset_path, exist_ok=True)
    if saved_image_zip_file == None:
        url = "https://drive.google.com/u/0/uc?id=1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk"
        import gdown
        with open(dataset_path/"raw.zip", "wb+") as handle:
            gdown.download(url, handle, quiet=False)
    else:
        os.system(f"cp {saved_image_zip_file} {dataset_path/'raw.zip'}")
    
    if not (dataset_path/'raw.zip').exists():
        raise Exception(f"{dataset_path/'raw.zip'} doesn't exist")

    os.system(f"unzip {dataset_path/'raw.zip'} -d {dataset_path}")

    for filename in ["train.csv", "val.csv", "test.csv"]:
        os.system(f"wget https://raw.githubusercontent.com/twitter/meta-learning-lstm/master/data/miniImagenet/{filename} -O {dataset_path/filename}")
    os.system(f"rm -rf {dataset_path/'meta-learning-lstm'}")
    os.system(f"rm {dataset_path/'raw.zip'}")

def build_meta_learning_tasks(dataset_path, args, disable_training=False):
    if not disable_training:
        train = MiniImagenet(dataset_path, mode='train', n_way=args["n_way"], k_shot=args["k_spt"],
                            k_query=args["k_qry"],
                            batchsz=args.get("train_bs", 10000), resize=args["imgsz"])
        train = torch.utils.data.DataLoader(train, args["task_num"], shuffle=False, num_workers=0, pin_memory=True)

        val = MiniImagenet(dataset_path, mode='val', n_way=args["n_way"], k_shot=args["k_spt"],
                                k_query=args["k_qry"],
                                batchsz=args.get("test_bs", 100), resize=args["imgsz"])
        val = torch.utils.data.DataLoader(val, 1, shuffle=False, num_workers=0, pin_memory=True)

    else:
        train, val = None, None

    test = MiniImagenet(dataset_path, mode='test', n_way=args["n_way"], k_shot=args["k_spt"],
                             k_query=args["k_qry"],
                             batchsz=args.get("test_bs", 100), resize=args["imgsz"])
    test = torch.utils.data.DataLoader(test, 1, shuffle=False, num_workers=0, pin_memory=True)

    return train, val, test

def mini_imagenet(args, redownload=False, disable_training=False):
    dataset_path = Path(args.get("dataset_location", "data/miniimagenet/"))
    cache = dataset_path/"cache"
    if redownload:
        print(f"deleting dataset from {dataset_path}")
        shutil.rmtree(dataset_path)

    if not dataset_path.exists():
        download_raw(dataset_path)

    return build_meta_learning_tasks(dataset_path, args, disable_training=disable_training)

def dataset(args, **kwargs):
    dataset_name = args["dataset_name"]
    
    if dataset_name == "cifar10":
        return cifar10(args, **kwargs)
    elif dataset_name == "mnist":
        return mnist(args, **kwargs)
    elif dataset_name == "mini_imagenet":
        return mini_imagenet(args, **kwargs)
    else:
        raise NotImplementedError(f"dataset {dataset_name} not implemented")