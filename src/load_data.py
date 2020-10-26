import torchvision
import torch

def cfar10(args):
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    traindataset = torchvision.datasets.CIFAR10('data/CFAR10', train=True, download=True,transform=transform)
    testdataset = torchvision.datasets.CIFAR10('data/CFAR10', train=False, download=True,transform=transform)
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args["batch_size"], shuffle=True, num_workers=0,drop_last=False)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args["batch_size"], shuffle=False, num_workers=0,drop_last=True)

    return train_loader, test_loader, (traindataset, testdataset)

def mnist(args):
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    traindataset = torchvision.datasets.MNIST('data/MNIST', train=True, download=True,transform=transform)
    testdataset = torchvision.datasets.MNIST('data/MNIST', train=False, download=True,transform=transform)
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args["batch_size"], shuffle=True, num_workers=0,drop_last=False)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args["batch_size"], shuffle=False, num_workers=0,drop_last=True)

    return train_loader, test_loader, (traindataset, testdataset)