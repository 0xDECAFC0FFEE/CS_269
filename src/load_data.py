import torchvision
import torch

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

def mini_imagenet(args):
    from .MiniImagenet import MiniImagenet
    train = MiniImagenet('data/miniimagenet/', mode='train', n_way=args["n_way"], k_shot=args["k_spt"],
                        k_query=args["k_qry"],
                        batchsz=10000, resize=args["imgsz"])
    
    val = MiniImagenet('data/miniimagenet/', mode='val', n_way=args["n_way"], k_shot=args["k_spt"],
                             k_query=args["k_qry"],
                             batchsz=100, resize=args["imgsz"])

    test = MiniImagenet('data/miniimagenet/', mode='test', n_way=args["n_way"], k_shot=args["k_spt"],
                             k_query=args["k_qry"],
                             batchsz=100, resize=args["imgsz"])

    train = torch.utils.data.DataLoader(train, args["task_num"], shuffle=True, num_workers=2, pin_memory=True)
    val = torch.utils.data.DataLoader(val, 1, shuffle=True, num_workers=2, pin_memory=True)
    test = torch.utils.data.DataLoader(test, 1, shuffle=True, num_workers=2, pin_memory=True)

    return train, val, test