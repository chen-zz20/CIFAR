import numpy as np
import pickle
import os
from torch.utils.data import Dataset
import torch





class Train_Data(Dataset):
    def __init__(self, data_dir:str, mode:str):
        super().__init__()
        self.train_X = None
        self.train_Y = None
        if mode == "cifar10":
            self.train_X , self.train_Y = self.load_cifar10(data_dir)
        elif mode == "cifar100":
            self.train_X , self.train_Y = self.load_cifar100(data_dir)
        else:
            exit("mode should be in ['cifar10', 'cifar100']")
        self.train_len = len(self.train_Y)


    def load_cifar10(self, data_dir):
        data_dir = os.path.join(data_dir, 'cifar-10-batches-py')
        data_list = []
        label_list = []
        for batch_id in range(1,6):
            filename = os.path.join(data_dir, 'data_batch_{}'.format(batch_id))
            with open(filename, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                data_list.append(batch['data'.encode()])
                label_list.append(batch['labels'.encode()])
        trX = np.reshape(np.concatenate(data_list, axis=0), (50000, 3, 32, 32))
        trY = np.reshape(np.concatenate(label_list, axis=-1), (50000, ))

        trX = ((trX - 128.0) / 255.0).astype(np.float32)

        return torch.Tensor(trX), torch.Tensor(trY).int()
    

    def load_cifar100(self, data_dir):
        data_dir = os.path.join(data_dir, 'cifar-100-python')
        filename = os.path.join(data_dir, 'train')
        with open(filename, 'rb') as f:
            dataset = pickle.load(f, encoding='bytes')
        trX = np.reshape(dataset['data'.encode()], (50000, 3, 32, 32))
        trY = np.reshape(dataset['fine_labels'.encode()], (50000, ))

        trX = ((trX - 128.0) / 255.0).astype(np.float32)

        return torch.Tensor(trX), torch.Tensor(trY).int()


    def __getitem__(self, index):
        return self.train_X[index], self.train_Y[index]

 
    def __len__(self):
        return self.train_len


class Test_Data(Dataset):
    def __init__(self, data_dir:str, mode:str):
        super().__init__()
        self.test_X = None
        self.test_Y = None
        if mode == "cifar10":
            self.test_X , self.test_Y = self.load_cifar10(data_dir)
        elif mode == "cifar100":
            self.test_X , self.test_Y = self.load_cifar100(data_dir)
        else:
            exit("mode should be in ['cifar10', 'cifar100']")
        self.test_len = len(self.test_Y)


    def load_cifar10(self, data_dir):
        data_dir = os.path.join(data_dir, 'cifar-10-batches-py')
        filename = os.path.join(data_dir, 'test_batch')
        with open(filename, 'rb') as f:
            dataset = pickle.load(f, encoding='bytes')
        teX = np.reshape(dataset['data'.encode()], (10000, 3, 32, 32))
        teY = np.reshape(dataset['labels'.encode()], (10000, ))

        teX = ((teX - 128.0) / 255.0).astype(np.float32)

        return torch.Tensor(teX), torch.Tensor(teY).int()
    

    def load_cifar100(self, data_dir):
        data_dir = os.path.join(data_dir, 'cifar-100-python')
        filename = os.path.join(data_dir, 'test')
        with open(filename, 'rb') as f:
            dataset = pickle.load(f, encoding='bytes')
        teX = np.reshape(dataset['data'.encode()], (10000, 3, 32, 32))
        teY = np.reshape(dataset['fine_labels'.encode()], (10000, ))

        teX = ((teX - 128.0) / 255.0).astype(np.float32)

        return torch.Tensor(teX), torch.Tensor(teY).int()



    def __getitem__(self, index):
        return self.test_X[index], self.test_Y[index]

 
    def __len__(self):
        return self.test_len

