from . import lth
import torch
import torch.nn as nn
import torch.nn.init as init

class VGG_cifar10(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG_cifar10, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1)
        init.xavier_normal_(self.conv1.weight.data)
        init.normal_(self.conv1.bias.data)
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
        init.xavier_normal_(self.conv2.weight.data)
        init.normal_(self.conv2.bias.data)
        self.conv2_act = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        init.xavier_normal_(self.conv3.weight.data)
        init.normal_(self.conv3.bias.data)
        self.conv3_act = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1)
        init.xavier_normal_(self.conv4.weight.data)
        init.normal_(self.conv4.bias.data)
        self.conv4_act = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        init.xavier_normal_(self.conv5.weight.data)
        init.normal_(self.conv5.bias.data)
        self.conv5_act = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        init.xavier_normal_(self.conv6.weight.data)
        init.normal_(self.conv6.bias.data)
        self.conv6_act = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.layer1 = nn.Linear(4096, 256)
        init.xavier_normal_(self.layer1.weight.data)
        init.normal_(self.layer1.bias.data)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer2 = nn.Linear(256, 256)
        init.xavier_normal_(self.layer2.weight.data)
        init.normal_(self.layer2.bias.data)
        self.relu2 = nn.ReLU(inplace=True)

        self.layer3 = nn.Linear(256, 10)
        init.xavier_normal_(self.layer3.weight.data)
        init.normal_(self.layer3.bias.data)
        self.mask = lth.build_mask(self)
    
    def forward(self, x):
        c1_out = self.conv1_act(self.conv1(x))
        c2_out = self.conv2_act(self.conv2(c1_out))
        pool1_out = self.pool1(c2_out)

        c3_out = self.conv3_act(self.conv3(pool1_out))
        c4_out = self.conv4_act(self.conv4(c3_out))
        pool2_out = self.pool2(c4_out)

        c5_out = self.conv5_act(self.conv5(pool2_out))
        c6_out = self.conv6_act(self.conv6(c5_out))
        pool3_out = self.pool3(c6_out)

        pool3_out = torch.flatten(pool3_out, 1)
        l1_out = self.relu1(self.layer1(pool3_out))
        l2_out = self.relu2(self.layer2(l1_out))
        logit = self.layer3(l2_out)
        return logit