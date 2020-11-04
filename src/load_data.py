import torchvision
import torch

def cifar10(args):
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train = torchvision.datasets.CIFAR10('data/CFAR10', train=True, download=True,transform=transform)
    test = torchvision.datasets.CIFAR10('data/CFAR10', train=False, download=True,transform=transform)

    n_train = int(len(train)*.8)
    n_val = len(train)-n_train
    train, val = torch.utils.data.random_split(train, [n_train, n_val])

    train = torch.utils.data.DataLoader(train, batch_size=args["batch_size"], shuffle=True, num_workers=0,drop_last=False)
    val = torch.utils.data.DataLoader(val, batch_size=args["batch_size"], shuffle=True, num_workers=0,drop_last=False)
    test = torch.utils.data.DataLoader(test, batch_size=args["batch_size"], shuffle=False, num_workers=0,drop_last=True)

    return train, val, test

def mnist(args):
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train = torchvision.datasets.MNIST('data/MNIST', train=True, download=True,transform=transform)
    test = torchvision.datasets.MNIST('data/MNIST', train=False, download=True,transform=transform)

    n_train = int(len(train)*.8)
    n_val = len(train)-n_train
    train, val = torch.utils.data.random_split(train, [n_train, n_val])
    
    train = torch.utils.data.DataLoader(train, batch_size=args["batch_size"], shuffle=True, num_workers=0,drop_last=False)
    val = torch.utils.data.DataLoader(val, batch_size=args["batch_size"], shuffle=True, num_workers=0,drop_last=False)
    test = torch.utils.data.DataLoader(test, batch_size=args["batch_size"], shuffle=False, num_workers=0,drop_last=True)

    return train, val, test