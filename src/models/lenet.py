from . import lth
import torch
import torch.nn as nn
import torch.nn.init as init

class LeNet_mnist(nn.Module):
    def __init__(self):
        super(LeNet_mnist, self).__init__()
        self.layer1 = nn.Linear(784, 300)
        init.xavier_normal_(self.layer1.weight.data)
        init.zeros_(self.layer1.bias.data)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer2 = nn.Linear(300, 100)
        init.xavier_normal_(self.layer2.weight.data)
        init.zeros_(self.layer2.bias.data)
        self.relu2 = nn.ReLU(inplace=True)

        self.layer3 = nn.Linear(100, 10)
        init.xavier_normal_(self.layer3.weight.data)
        init.zeros_(self.layer3.bias.data)

    def forward(self, x):
        x = torch.flatten(x, 1)
        l1_out = self.relu1(self.layer1(x))
        l2_out = self.relu2(self.layer2(l1_out))
        logit = self.layer3(l2_out)
        return logit

# class LeNet_cifar10(nn.Module):
#     def __init__(self):
#         super(LeNet_cifar10, self).__init__()
#         self.layer1 = nn.Linear(3072, 300)
#         init.xavier_normal_(self.layer1.weight.data)
#         init.normal_(self.layer1.bias.data)
#         self.relu1 = nn.ReLU(inplace=True)

#         self.layer2 = nn.Linear(300, 100)
#         init.xavier_normal_(self.layer2.weight.data)
#         init.normal_(self.layer2.bias.data)
#         self.relu2 = nn.ReLU(inplace=True)

#         self.layer3 = nn.Linear(100, 10)
#         init.xavier_normal_(self.layer3.weight.data)
#         init.normal_(self.layer3.bias.data)
#         self.mask = lth.build_mask(self)

#     def forward(self, x):
#         x = torch.flatten(x, 1)
#         l1_out = self.relu1(self.layer1(x))
#         l2_out = self.relu2(self.layer2(l1_out))
#         logit = self.layer3(l2_out)
#         return logit