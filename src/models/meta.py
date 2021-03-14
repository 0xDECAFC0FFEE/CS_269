import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    src.models.maml_learner import Learner
from    copy import deepcopy
from .mask_ops import apply_mask
from collections import OrderedDict
from tqdm import tqdm

def update_weights(named_parameters, loss, lr, first_order):
    names, params = list(zip(*named_parameters))
    if not first_order:
        grad = torch.autograd.grad(loss, params)
    else:
        grad = torch.autograd.grad(loss, params, create_graph=True)
        grad = [g.detach() for g in grad]

    fast_weights = [p - lr * g for p, g in zip(params, grad)]
    fast_weights = OrderedDict(zip(names, fast_weights))
    return fast_weights

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, training_params, dataset_params, config):
        super(Meta, self).__init__()

        self.update_lr = training_params["update_lr"]
        self.meta_lr = training_params["meta_lr"]
        self.n_way = dataset_params["n_way"]
        self.k_spt = dataset_params["k_spt"]
        self.k_qry = dataset_params["k_qry"]
        self.task_num = dataset_params["task_num"]
        self.update_step = training_params["update_step"]
        self.finetune_step = training_params["finetune_step"]
        self.first_order = training_params["first_order"]


        self.net = Learner(config, dataset_params["imgsz"])
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)


    def forward(self, mask, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [task_num, setsz, c_, h, w]
        :param y_spt:   [task_num, setsz]
        :param x_qry:   [task_num, querysz, c_, h, w]
        :param y_qry:   [task_num, querysz]
        :return:
        """
        # print("x_spt.shape", x_spt.shape)
        # print("y_spt.shape", y_spt.shape)
        # print("x_qry.shape", x_qry.shape)
        # print("y_qry.shape", y_qry.shape)
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]


        for i in range(task_num):
            # this is the loss and accuracy before first update
            fast_weights = OrderedDict(self.net.named_parameters())
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(mask, x_qry[i], fast_weights, training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] += correct

            for k in range(self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(mask, x_spt[i], fast_weights, training=True)
                loss = F.cross_entropy(logits, y_spt[i])

                fast_weights = update_weights(fast_weights.items(), loss, self.update_lr, self.first_order)

                logits_q = self.net(mask, x_qry[i], fast_weights, training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[k + 1] += correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()


        accs = np.array(corrects) / (querysz * task_num)

        return accs


    def finetunning(self, mask, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        corrects = []

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        # apply_mask(model, mask)
        net = deepcopy(self.net)

        # this is the loss and accuracy before first update
        fast_weights = OrderedDict(net.named_parameters())

        with torch.no_grad():
            logits_q = net(mask, x_qry, fast_weights, training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            corrects.append(torch.eq(pred_q, y_qry).sum().item()/x_qry.size(0))

        for k in range(self.finetune_step):
            logits = net(mask, x_spt, fast_weights, training=True)
            loss = F.cross_entropy(logits, y_spt)
            fast_weights = update_weights(fast_weights.items(), loss, self.update_lr, False)

            with torch.no_grad():
                logits_q = net(mask, x_qry, fast_weights, training=True)
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                corrects.append(torch.eq(pred_q, y_qry).sum().item()/x_qry.size(0))

        del net

        return corrects




def main():
    pass


if __name__ == '__main__':
    main()
