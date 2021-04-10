import os
import sys
import math
import time
import logging
import sys
import argparse
import torch
import glob
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from read_data import MyDataset
from model import resnet18
import utils
from utils import adjust_learning_rate, save_checkpoint
import numpy as np
from random import shuffle
from ninpy.datasets import load_toy_dataset, get_cifar10_transforms


parser = argparse.ArgumentParser("ImageNet")
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--report_freq', type=float, default=300, help='report frequency')
parser.add_argument('--epochs', type=int, default=40, help='num of training epochs')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=10, help='gradient clipping')
parser.add_argument('--resume_train', action='store_true', default=False, help='resume training')
parser.add_argument('--resume_dir', type=str, default='./weights/checkpoint.pth.tar', help='save weights directory')
parser.add_argument('--load_epoch', type=int, default=30, help='random seed')
parser.add_argument('--weights_dir', type=str, default='./weights/', help='save weights directory')
parser.add_argument('--learning_step', type=list, default=[20,30,40], help='learning rate steps')


args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
if not os.path.exists(args.save):
    os.makedirs(args.save)
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():

    num_epochs = args.epochs
    batch_size = args.batch_size

    train_transforms, test_transforms = get_cifar10_transforms()
    train_loader, test_loader = load_toy_dataset(batch_size, batch_size, 8, 'cifar10', './dataset', True, train_transforms, test_transforms)

    #num_train = train_dataset.__len__()
    #n_train_batches = math.floor(num_train / batch_size)


    criterion = nn.CrossEntropyLoss().cuda()
    bitW = 1
    bitA = 1
    model = resnet18(bitW, bitA, pretrained=True)
    model = model.cuda()

    #model = utils.dataparallel(model, 4)


    print("Compilation complete, starting training...")

    test_record = []
    train_record = []
    learning_rate = args.learning_rate
    epoch = 0
    step_idx = 0
    best_top1 = 0


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    while epoch < num_epochs:

        logging.info('epoch %d lr %e', epoch, learning_rate)
        epoch = epoch + 1
    # resume training    
        if (args.resume_train) and (epoch == 1):   
            checkpoint = torch.load(args.resume_dir)
            epoch = checkpoint['epoch']
            learning_rate = checkpoint['learning_rate']
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['state_dict'])
            test_record = list(
                np.load(args.weights_dir + 'test_record.npy'))
            train_record = list(
                np.load(args.weights_dir + 'train_record.npy'))

    # training
        train_acc_top1, train_acc_top5, train_obj = train(train_loader, model, criterion, optimizer, learning_rate)
        logging.info('train_acc %f', train_acc_top1)
        train_record.append([train_acc_top1, train_acc_top5])
        np.save(args.weights_dir + 'train_record.npy', train_record)   

    # test
        test_acc_top1, test_acc_top5, test_obj = infer(test_loader, model, criterion)
        is_best = test_acc_top1 > best_top1
        if is_best:
            best_top1 = test_acc_top1

        logging.info('test_acc %f', test_acc_top1)
        test_record.append([test_acc_top1, test_acc_top5])
        np.save(args.weights_dir + 'test_record.npy', test_record)

        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'best_top1': best_top1,
                'learning_rate': learning_rate,
                }, args, is_best)

        step_idx, learning_rate = utils.adjust_learning_rate(args, epoch, step_idx,
                                           learning_rate)

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate


def train(train_queue, model, criterion, optimizer, lr):

    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    model.train()

    for step, (input, target) in enumerate(train_queue):
   
        n = input.size(0)
        input = input.cuda()
        target = target.cuda()


        logits = model(input)
        loss = criterion(logits, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg



def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()

            logits = model(input)
            loss = criterion(logits, target)
 
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg
 

if __name__ == '__main__':
    utils.create_folder(args)       
    main()

