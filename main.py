import argparse
import os
from tqdm import tqdm
from time import time

from tensorboardX import SummaryWriter

from dataload import Train_Data, Test_Data
from model import Model
from control import train_epoch, test_epoch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=100,
	help='Batch size for mini-batch training and evaluating. Default: 100')
parser.add_argument('--num_epochs', type=int, default=20,
	help='Number of training epoch. Default: 20')
parser.add_argument('-hd', '--hidden_dim', type=int, default=128,
	help='Number of hidden dim. Default: 128')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
	help='Learning rate during optimization. Default: 1e-3')
parser.add_argument('-dr', '--drop_rate', type=float, default=0.5,
	help='Drop rate of the Dropout Layer. Default: 0.5')
parser.add_argument('--test', default=False, action="store_true", 
	help='True to train and False to inference. Default: True')
parser.add_argument('--data_dir', type=str, default='./data',
	help='Data directory. Default: ./data')
parser.add_argument('--train_dir', type=str, default='./train',
	help='Training directory for saving model. Default: ./train')
parser.add_argument('--log_dir', type=str, 
default='./log', 
    help='Log directory. Default: ./log')
parser.add_argument('--mode', type=str, default='cifar10',
	help='Training mode to choose from [cifar10, cifar100]. Default: cifar10')
parser.add_argument('--pretrained', type=str, default=None,
	help='Give a pretrained model. Default: None')
parser.add_argument('--name', type=str, default='test',
	help='Give the model a name. Default: test')
args = parser.parse_args()


if __name__ == '__main__':
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    train_loader = DataLoader(Train_Data(args.data_dir, args.mode), batch_size=args.batch_size, shuffle=True, num_workers=10, pin_memory=is_cuda)
    test_loader = DataLoader(Test_Data(args.data_dir, args.mode), batch_size=args.batch_size, shuffle=True, num_workers=10, pin_memory=is_cuda)
    model = None
    if args.pretrained is not None:
        model_path = os.path.join(args.train_dir, f"model-{args.pretrained}-{args.mode}.pth.tar")
        if os.path.exists(model_path):
            model = torch.load(model_path)
            model = model.to(device)
    if model is None:
        model = Model(mode=args.mode, hidden_dim=args.hidden_dim, drop_rate=args.drop_rate, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if not os.path.exists(args.train_dir):
        os.mkdir(args.train_dir)

    if not args.test:
        log_dir = os.path.join(args.log_dir, 'train')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        train_writer = SummaryWriter(log_dir=log_dir)
        log_dir = os.path.join(args.log_dir, 'test')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        test_writer = SummaryWriter(log_dir=log_dir)

        print(model)
        print("begin trainning")
        begin = time()

        for epoch in tqdm(range(1, args.num_epochs+1)):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer)
            train_writer.add_scalar("loss", train_loss, epoch)
            train_writer.add_scalar("accuracy", train_acc, epoch)

            test_loss, test_acc = test_epoch(model, test_loader)
            test_writer.add_scalar("loss", test_loss, epoch)
            test_writer.add_scalar("accuracy", test_acc, epoch)
        end = time()
        use_time = end - begin
        minutes = use_time // 60
        use_time -= minutes * 60
        use_time = "%.3f" % (use_time)
        print("end trainning")
        print(f"{args.name} training used {minutes} minutes {use_time} seconds.")
        print("begin testing")
        test_loss, test_acc = test_epoch(model, test_loader)
        msg = "The final loss is %.3f" % test_loss
        msg += ", final accuracy is "
        msg += "{:.2%}".format(test_acc)
        print(msg)
        with open(os.path.join(args.train_dir, f"model-{args.name}-{args.mode}.pth.tar"), 'wb') as fout:
            torch.save(model, fout)
        train_writer.close()
        test_writer.close()
    
    else:
        print("begin testing")
        test_loss, test_acc = test_epoch(model, test_loader)
        msg = "The final loss is %.3f" % test_loss
        msg += ", final accuracy is "
        msg += "{:.2%}".format(test_acc)
        print(msg)