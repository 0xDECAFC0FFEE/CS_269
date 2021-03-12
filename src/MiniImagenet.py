import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random
import torchvision
from torchvision.transforms.functional import resize
if __name__ == "src.MiniImagenet":
    from .utils import fs_greedy_load
if __name__ == "__main__":
    from utils import fs_greedy_load
import pickle
from pathlib import Path
from tqdm import tqdm
import re
import shutil
import torch.nn.functional as F
from sklearn.preprocessing import  OneHotEncoder



class cached_transformer:
    def __init__(self, images_source_path, image_filenames, cache, transform, reload_cache=False):
        if reload_cache:
            shutil.rmtree(cache)
        try:
            with open(cache/"image_path_to_id.pkl", "rb") as handle:
                self.map = pickle.load(handle)

            self.data = fs_greedy_load(cache/"data")
        except FileNotFoundError:
            print("transforming images and loading into memmapped cache")
            os.makedirs(cache, exist_ok=True)

            img_path_to_id = {path: id for id, path in enumerate(image_filenames)}

            tramsformed_images = np.array([transform(images_source_path/path).numpy() for path in tqdm(image_filenames)])

            self.data = fs_greedy_load(cache/"data", lst_array=tramsformed_images)
            self.map = img_path_to_id
            with open(cache/"image_path_to_id.pkl", "wb+") as handle:
                pickle.dump(self.map, handle)
            print("loaded transformed images into cache")

    def __call__(self, path):
        return self.data[self.map[path]]
    
    def __len__(self):
        return len(self.data)

class MiniImagenet(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- images/*.jpg includes all imgeas
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, root, mode, args, startidx=0):
        """

        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """

        self.n_way = args["n_way"]  # n-way
        self.k_shot = args["k_spt"]  # k-shot
        self.k_query = args["k_qry"]  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = args["imgsz"]  # resize to
        self.startidx = startidx  # index label not from 0, but from startidx
        self.mode = mode 
        # print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
        # mode, self.n_way, self.k_shot, self.k_query, self.resize))

        csvdata = self.loadCSV(os.path.join(root, mode + '.csv'))  # csv path
        self.data = []
        self.img2label = {}
        for i, (k, v) in enumerate(csvdata.items()):
            self.data.append(v)  # [[img1, img2, ...], [img111, ...]]
            self.img2label[k] = i + self.startidx  # {"img_name[:9]":label}
        # print(csvdata)
        from itertools import chain
        print("self data", len(set(chain(*self.data))))
        self.cls_num = len(self.data)

        self.path = Path(root)/"images"  # image path
        if mode == 'train':
            self.batchsz = args.get("train_bs", 10000)  # batch of set, not batch of imgs
            # transform = transforms.Compose([
            #     lambda x: Image.open(x).convert('RGB'),
            #     transforms.ToTensor(),
            #     transforms.Resize((self.resize, self.resize)),
            #     transforms.RandomResizedCrop(self.resize),
            #     transforms.ColorJitter(.4, .4, .4, 0),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            # ])
            # self.transform = transform

            tensor_transforms = torch.nn.Sequential(
                transforms.Resize((self.resize, self.resize)),
                transforms.RandomResizedCrop(self.resize),
                transforms.ColorJitter(.4, .4, .4, 0),
                transforms.RandomHorizontalFlip(),
                # utils.mixup()
                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            )
            self.transform = torch.jit.script(tensor_transforms)

            # self.transform = cached_transformer(self.path, set(chain(*self.data)), Path(root)/"cache"/mode/"no_augment", transform)
        else:
            self.batchsz = args.get("test_bs", 100)  # batch of set, not batch of imgs
            # transform = transforms.Compose([
            #     lambda x: Image.open(x).convert('RGB'),
            #     transforms.Resize((self.resize, self.resize)),
            #     #  transforms.RandomResizedCrop(self.resize),
            #     #  transforms.ColorJitter(.4, .4, .4, .4),
            #     #  transforms.RandomHorizontalFlip(),
            #      transforms.ToTensor(),
            #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            # ])
            # self.transform = transform

            tensor_transforms = torch.nn.Sequential(
                transforms.Resize((self.resize, self.resize)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            )
            self.transform = torch.jit.script(tensor_transforms)

            # self.transform = cached_transformer(self.path, set(chain(*self.data)), Path(root)/"cache"/mode, transform)

        self.create_batch(self.batchsz)

        self.counts = {}

    def loadCSV(self, csvf):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets
    #========================================================================
    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        
        #y_a, y_b = y, y[index]
        mixed_y = lam * y + (1 - lam) * y[index, :]     
        return mixed_x, mixed_y, lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    #========================================================================


    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        # support_x shape = [setsz, 3, resize, resize]
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        # support_y shape = [setsz]
        # query_x shape = [querysz, 3, resize, resize]
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        # query_y shape = [querysz]

        flatten_support_x = [os.path.join(self.path, item) for sublist in self.support_x_batch[index] for item in sublist]
        support_y = np.array(
            [self.img2label[item[:9]]  # filename:n0153282900000005.jpg, the first 9 characters treated as label
             for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)

        flatten_query_x = [os.path.join(self.path, item) for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([self.img2label[item[:9]]
                            for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)



        # print('global:', support_y, query_y)
        # support_y: [setsz]
        # query_y: [querysz]
        # unique: [n-way], sorted
        unique = np.unique(support_y)
        random.shuffle(unique)
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(self.querysz)
        

        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx

        # print('relative:', support_y_relative, query_y_relative)

        for i, path in enumerate(flatten_support_x):
            self.counts[path] = self.counts.get(path, 0)+1
            # support_x[i] = torch.FloatTensor(self.transform(path))
            support_x[i] = self.transform(torchvision.transforms.functional.to_tensor(Image.open(path).convert('RGB')))

        for i, path in enumerate(flatten_query_x):
            self.counts[path] = self.counts.get(path, 0)+1
            # query_x[i] = torch.FloatTensor(self.transform(path))
            query_x[i] = self.transform(torchvision.transforms.functional.to_tensor(Image.open(path).convert('RGB')))
        #====================================================================================================================
        #Turn np array to tensor for one hot encoding 
        torch_qyh = torch.from_numpy(query_y_relative)
        onehot_query_y_relative = F.one_hot(torch_qyh.long(), num_classes=5)


        #print("onehot",onehot_query_y_relative)

        """
        print("q_y_r",query_y_relative)
        print("torch_qyh",torch_qyh)
        print("onehot",onehot_query_y_relative)
        """       
        if self.mode == "train":
            #do mixup
            mixed_x , mixed_y, c = self.mixup_data(query_x,onehot_query_y_relative,alpha=1.0)

            #return output
            query_x = mixed_x
            query_y_relative = mixed_y
      
            #print("query_y_relative",query_y)

            #print("query_x",query_x)
            #print("C",c)
        else:
            query_y_relative = torch.FloatTensor(onehot_query_y_relative.numpy())
            #query_y_relative = onehot_query_y_relative
            query_x = query_x
        #====================================================================================================================
        
        #print(self.mode)
        #print("shape",query_y_relative.shape)
        #print("q_y_r",query_y_relative)
        """

        query_y = onehot_query_y_relative
        query_x = query_x
        """

        if random.randint(0, 100) == 50:
            print("average image augmentation calls: ", sum(self.counts.values())/len(self.counts), "total # of images:", len(self.counts))

        #return support_x, torch.LongTensor(support_y_relative), query_x, torch.Tensor(query_y_relative)
        return support_x, torch.FloatTensor(support_y_relative), query_x, query_y_relative



    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz



if __name__ == '__main__':
    # the following episode is to view one set of images via tensorboard.
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    from tensorboardX import SummaryWriter
    import time

    plt.ion()

    tb = SummaryWriter('runs', 'mini-imagenet')
    #mini = MiniImagenet('../mini-imagenet/', mode='train', n_way=5, k_shot=1, k_query=1, batchsz=1000, resize=168)
    mini = MiniImagenet('src/data/miniimagenet/', mode='train', args={'n_way':5, 'k_spt':1, 'k_qry':1, 'batchsz':1000, 'imgsz':168})

    for i, set_ in enumerate(mini):
        # support_x: [k_shot*n_way, 3, 84, 84]
        support_x, support_y, query_x, query_y = set_

        support_x = make_grid(support_x, nrow=2)
        query_x = make_grid(query_x, nrow=2)

        plt.figure(1)
        plt.imshow(support_x.transpose(2, 0).numpy())
        #plt.pause(0.5)
        plt.figure(2)
        plt.imshow(query_x.transpose(2, 0).numpy())
        #plt.pause(0.5)

        tb.add_image('support_x', support_x)
        tb.add_image('query_x', query_x)

        #time.sleep(5)

    tb.close()
