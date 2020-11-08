import torchvision
import torch
from pathlib import Path
from .MiniImagenet import MiniImagenet
import gdown
import os
import shutil

def cifar10(args):
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train = torchvision.datasets.CIFAR10('data/CFAR10', train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10('data/CFAR10', train=False, download=True, transform=transform)

    n_train = int(len(train)*.8)
    n_val = len(train)-n_train
    train, val = torch.utils.data.random_split(train, [n_train, n_val])

    train = torch.utils.data.DataLoader(train, batch_size=args["batch_size"], shuffle=True, num_workers=0,drop_last=False, pin_memory=True)
    val = torch.utils.data.DataLoader(val, batch_size=args["batch_size"], shuffle=True, num_workers=0,drop_last=False, pin_memory=True)
    test = torch.utils.data.DataLoader(test, batch_size=args["batch_size"], shuffle=False, num_workers=0,drop_last=True, pin_memory=True)

    return train, val, test

def mnist(args):
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train = torchvision.datasets.MNIST('data/MNIST', train=True, download=True,transform=transform)
    test = torchvision.datasets.MNIST('data/MNIST', train=False, download=True,transform=transform)

    n_train = int(len(train)*.8)
    n_val = len(train)-n_train
    train, val = torch.utils.data.random_split(train, [n_train, n_val])
    
    train = torch.utils.data.DataLoader(train, batch_size=args["batch_size"], shuffle=True, num_workers=0,drop_last=False, pin_memory=True)
    val = torch.utils.data.DataLoader(val, batch_size=args["batch_size"], shuffle=True, num_workers=0,drop_last=False, pin_memory=True)
    test = torch.utils.data.DataLoader(test, batch_size=args["batch_size"], shuffle=False, num_workers=0,drop_last=True, pin_memory=True)

    return train, val, test

def mini_imagenet(args, redownload=False):
    dataset_path = Path("/workspace/data/miniimagenet_3/")
    if redownload:
        print(f"deleting dataset from {dataset_path}")
        shutil.rmtree(dataset_path)

    if not dataset_path.exists():
        print("downloading mini imagenet dataset (2.9 Gb)... ")
        url = "https://drive.google.com/u/0/uc?id=1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk"

        os.makedirs(dataset_path, exist_ok=True)
        with open(dataset_path/"raw.zip", "wb+") as handle:
            gdown.download(url, handle, quiet=False)
        os.system(f"unzip {dataset_path/'raw.zip'} -d {dataset_path}")
        os.system(f"git clone https://github.com/twitter/meta-learning-lstm.git {dataset_path/'meta-learning-lstm'}")
        os.system(f"mv {dataset_path/'meta-learning-lstm/data/miniImagenet/*'} {dataset_path}")
        os.system(f"rm -rf {dataset_path/'meta-learning-lstm'}")
        os.system(f"rm {dataset_path/'raw.zip'}")

    train = MiniImagenet(dataset_path, mode='train', n_way=args["n_way"], k_shot=args["k_spt"],
                        k_query=args["k_qry"],
                        batchsz=10000, resize=args["imgsz"])
    
    val = MiniImagenet(dataset_path, mode='val', n_way=args["n_way"], k_shot=args["k_spt"],
                             k_query=args["k_qry"],
                             batchsz=100, resize=args["imgsz"])

    test = MiniImagenet(dataset_path, mode='test', n_way=args["n_way"], k_shot=args["k_spt"],
                             k_query=args["k_qry"],
                             batchsz=100, resize=args["imgsz"])

    train = torch.utils.data.DataLoader(train, args["task_num"], shuffle=True, num_workers=2, pin_memory=True)
    val = torch.utils.data.DataLoader(val, 1, shuffle=True, num_workers=2, pin_memory=True)
    test = torch.utils.data.DataLoader(test, 1, shuffle=True, num_workers=2, pin_memory=True)

    return train, val, test