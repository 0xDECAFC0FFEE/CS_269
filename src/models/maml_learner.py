import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np
from collections import OrderedDict

class Learner(nn.Module):
    """
    """

    def __init__(self, config, imgsz):
        """
        :param config: network config file, type:list of (string, list)
        :param imgsz:  28 or 84
        """
        super().__init__()


        self.config = config

        for i, (name, param) in enumerate(self.config):
            if name == 'conv2d':
                w = nn.Parameter(torch.ones(*param[:4]))
                torch.nn.init.xavier_uniform_(w)
                self.register_parameter(f"{i}_weight", w)
                self.register_parameter(f"{i}_bias", nn.Parameter(torch.zeros(param[0])))

            elif name == 'convt2d':
                w = nn.Parameter(torch.ones(*param[:4]))
                torch.nn.init.xavier_uniform_(w)
                self.register_parameter(f"{i}_weight", w)
                self.register_parameter(f"{i}_bias", nn.Parameter(torch.zeros(param[1])))

            elif name == 'linear':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.xavier_uniform_(w)
                self.register_parameter(f"{i}_weight", w)
                self.register_parameter(f"{i}_bias", nn.Parameter(torch.zeros(param[0])))

            elif name == 'bn':
                w = nn.Parameter(torch.ones(param[0]))
                self.register_parameter(f"{i}_weight", w)
                self.register_parameter(f"{i}_bias", nn.Parameter(torch.zeros(param[0])))

                self.register_buffer(f"{i}_running_mean", torch.zeros(param[0]))
                self.register_buffer(f"{i}_running_var", torch.ones(param[0]))

            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid', 'dropout']:
                continue
            else:
                raise NotImplementedError

    def forward(self, mask, x, vars=None, training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :return: x, loss, likelihood, kld
        """

        if vars == None:
            # vars = self.vars
            vars = OrderedDict(self.named_parameters())

        bn_vars = OrderedDict(self.named_buffers())

        idx = 0
        bn_idx = 0

        for idx, (name, param) in enumerate(self.config):
            try:
                if name == 'conv2d':
                    w, b = vars[f"{idx}_weight"], vars[f"{idx}_bias"]
                    x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                elif name == 'convt2d':
                    w, b = vars[f"{idx}_weight"], vars[f"{idx}_bias"]
                    x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                elif name == 'linear':
                    w, b = vars[f"{idx}_weight"], vars[f"{idx}_bias"]
                    x = F.linear(x, w, b)
                elif name == 'bn':
                    w, b = vars[f"{idx}_weight"], vars[f"{idx}_bias"]
                    running_mean, running_var = bn_vars[f"{idx}_running_mean"], bn_vars[f"{idx}_running_var"]
                    x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=training)
                    bn_idx += 2

                elif name == 'flatten':
                    # print(x.shape)
                    x = x.view(x.size(0), -1)
                elif name == 'reshape':
                    # [b, 8] => [b, 2, 2, 2]
                    x = x.view(x.size(0), *param)
                elif name == 'relu':
                    x = F.relu(x, inplace=param[0])
                elif name == 'leakyrelu':
                    x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
                elif name == 'tanh':
                    x = F.tanh(x)
                elif name == 'sigmoid':
                    x = torch.sigmoid(x)
                elif name == 'upsample':
                    x = F.upsample_nearest(x, scale_factor=param[0])
                elif name == 'max_pool2d':
                    x = F.max_pool2d(x, param[0], param[1], param[2])
                elif name == 'avg_pool2d':
                    x = F.avg_pool2d(x, param[0], param[1], param[2])
                elif name == 'dropout':
                    x = F.dropout(x, param[0], training=training)
                else:
                    raise NotImplementedError
            except Exception as e:
                print("layer input shape:", x.shape)
                raise e


        # make sure variable is used properly
        # assert idx == len(vars)
        # assert bn_idx == len(self.vars_bn)


        return x


    def zero_grad(self, vars=None):
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars == None:
                for p in self.vars:
                    if p.grad != None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad != None:
                        p.grad.zero_()

    # def parameters(self):
    #     """
    #     override this function since initial parameters will return with a generator.
    #     :return:
    #     """
    #     return self.vars.values()
