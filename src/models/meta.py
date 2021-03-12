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

SOFT_CE = True

def update_weights(named_parameters, loss, lr):
    names, params = list(zip(*named_parameters))
    grad = torch.autograd.grad(loss, params)
    #grad = torch.autograd.grad(loss.sum(), params)

    fast_weights = [p - lr * g for p, g in zip(params, grad)]
    #fast_weights = [p - lr * torch.clamp(g, min=-100, max=100) for p, g in zip(params, grad)]
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


        self.net = Learner(config, dataset_params["imgsz"])
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
    

    def _cross_entropy(self, pred_prob, label_prob, reduction='average'):
        '''
        :param pred_prob : k-shot(batch) x classes
        :param label_prob: k-shot(batch)
        :param reduction: sum or average
        '''
        log_likelihood = -1 *torch.nn.functional.log_softmax(pred_prob, dim=1)
        if reduction == "average":
            loss = torch.sum(torch.mul(log_likelihood,label_prob))/label_prob.size(0)
        else:
            loss = torch.sum(torch.mul(log_likelihood,label_prob))
        return loss

    def forward(self, mask, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)
        #print("task_num", task_num)
        #print("querysz",querysz)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):
            # this is the loss and accuracy before first update
            fast_weights = OrderedDict(self.net.named_parameters())
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(mask, x_qry[i], fast_weights, training=False)
                
		
                #print("i:",i,"logits",logits_q)
               
                #print("y_qry:",y_qry.shape)   
                 
                #===========================
                if not SOFT_CE:
                    C=5
                    log_prob = torch.nn.functional.log_softmax(logits_q, dim=1)
                    loss_onehot= -torch.sum(log_prob)*y_qry[i]
                    loss_q = loss_onehot
                else:
                    loss_q = self._cross_entropy(logits_q, y_qry[i])
                #===========================
                #loss_q = F.cross_entropy(logits_q, y_qry[i])

                losses_q[0] += loss_q

                # print("loss_q",loss_q)
                #print("losses_q:",losses_q[0])
    

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i].argmax(dim=1)).sum().item()
                corrects[0] += correct

            for k in range(self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(mask, x_spt[i], fast_weights, training=True)
                #print("logits",logits)
              
                #===========================
                if not SOFT_CE:
                    C=5
                    log_prob_1 = torch.nn.functional.log_softmax(logits, dim=1)
                    loss_onehot_1= -torch.sum(log_prob_1)*y_spt[i]
                    loss = loss_onehot_1
                else:
                    loss = self._cross_entropy(logits, y_spt[i])
                #===========================
            
                #loss = F.cross_entropy(logits, y_spt[i])
                #print('i',i,'k',k,'loss in for k',loss)

            
                fast_weights = update_weights(fast_weights.items(), loss, self.update_lr)

                logits_q = self.net(mask, x_qry[i], fast_weights, training=False)
                #===========================
                if not SOFT_CE:
                    C=5
                    log_prob_2 = torch.nn.functional.log_softmax(logits_q, dim=1)
                    loss_onehot_2= -torch.sum(log_prob_2)*y_qry[i]
                    loss_q = loss_onehot_2           
                else:                
                    loss_q = self._cross_entropy(logits_q, y_qry[i])
            
                #===========================
                #loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        
                    correct = torch.eq(pred_q, y_qry[i].argmax(dim=1)).sum().item()
                    corrects[k + 1] += correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward(loss_q.clone().detach())
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()


        accs = np.array(corrects) / (querysz * task_num)
        #print("np.array(corrects):",np.array(corrects))
        #print("querysz",querysz)
        #print("task_num",task_num)
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
            logits_q = net(mask, x_qry, fast_weights, training=False)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            corrects.append(torch.eq(pred_q, y_qry.argmax(dim=1)).sum().item()/x_qry.size(0))

        for k in range(self.finetune_step):
            logits = net(mask, x_spt, fast_weights, training=True)
            

            #===========================
            if not SOFT_CE:
                C=5
                log_prob = torch.nn.functional.log_softmax(logits, dim=1)
                loss_onehot= -torch.sum(log_prob)*y_spt
                loss = loss_onehot                
            else:
                loss = self._cross_entropy(logits, y_spt)
            #===========================       
            #loss = F.cross_entropy(logits, y_spt)
            fast_weights = update_weights(fast_weights.items(), loss, self.update_lr)

            with torch.no_grad():
                logits_q = net(mask, x_qry, fast_weights, training=False)
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                corrects.append(torch.eq(pred_q, y_qry.argmax(dim=1)).sum().item()/x_qry.size(0))

        del net

        return corrects




def main():
    pass


if __name__ == '__main__':
    main()
