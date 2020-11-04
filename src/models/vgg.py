import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

class VGG_cifar10(nn.Module):
    def __init__(self, layers_desc, data_input_shape=(3, 32, 32)):
        super(VGG_cifar10, self).__init__()

        in_shape = data_input_shape

        self.layers = []
        for layer_desc in layers_desc:
            name, params = layer_desc
            if name == "conv":
                out_channels = params["out"]
                batchnorm = params.get("bn", False)
                stride_h, stride_w = params.get("stride", (3, 3))
                in_channels, h, w = in_shape

                conv = nn.Conv2d(in_channels, out_channels, kernel_size=(stride_h, stride_w), stride=1, padding=1)
                init.xavier_normal_(conv.weight.data)
                init.normal_(conv.bias.data)
                self.layers.append(conv)

                if batchnorm:
                    self.layers.append(nn.BatchNorm2d(out_channels))
                in_shape = (out_channels, h+2-stride_h+1, w+2-stride_h+1)

            elif name =="relu":
                self.layers.append(nn.ReLU())

            elif name =="pool":
                self.layers.append(nn.MaxPool2d(kernel_size=2))
                in_channels, h, w = in_shape
                in_shape = (in_channels, int(h/2), int(w/2))

            elif name == "flatten":
                self.layers.append(nn.Flatten(start_dim=1))
                in_shape = (np.prod(in_shape), )

            elif name == "linear":
                out_channels = params["out"]
                [in_channels] = in_shape

                matmul = nn.Linear(in_channels, out_channels)
                init.xavier_normal_(matmul.weight.data)
                init.normal_(matmul.bias.data)

                self.layers.append(matmul)
                in_shape = [out_channels]

            else:
                raise Exception(f"layer {layer_desc} not implemented")

        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
