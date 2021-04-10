import os
import sys

from tqdm import tqdm
import logging
import sys
import argparse
import torch
from torch.autograd import Variable
from torchvision import transforms, datasets
from new_layers import self_conv
import torch.nn as nn
import torch.nn.functional as F
from model import resnet18
import utils
from utils import adjust_learning_rate, save_checkpoint
import numpy as np
from multiprocessing import cpu_count
from ninpy.datasets import load_toy_dataset, get_cifar10_transforms

parser = argparse.ArgumentParser("ImageNet")
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.05, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--epochs', type=int, default=150, help='num of training epochs')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2021, help='random seed')
parser.add_argument('--grad_clip', type=float, default=10, help='gradient clipping')
parser.add_argument('--resume_train', action='store_true', default=False, help='resume training')
parser.add_argument('--resume_dir', type=str, default='./weights/checkpoint.pth.tar', help='save weights directory')
parser.add_argument('--load_epoch', type=int, default=2021, help='random seed')
parser.add_argument('--weights_dir', type=str, default='./weights/', help='save weights directory')
parser.add_argument('--learning_step', type=list, default=[50, 150], help='learning rate steps')
parser.add_argument('--num_worker', type=int, default=cpu_count(), help='num worker')
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
    train_loader, test_loader = load_toy_dataset(
        batch_size, batch_size, 8, 'cifar10', './dataset', True,
        train_transforms, test_transforms)
    criterion = nn.CrossEntropyLoss().cuda()
    bitW = 32
    bitA = 32
    model = resnet18(bitW, bitA)
    #model = utils.dataparallel(model, 3)


    print("Compilation complete, starting training...")

    test_record = []
    train_record = []
    learning_rate = args.learning_rate
    epoch = 0
    step_idx = 0
    best_top1 = 0


    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)

    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, self_conv):
            c = float(m.weight.data[0].nelement())
            torch.nn.init.xavier_uniform(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data = m.weight.data.zero_().add(1.0)
    model = model.cuda()

    pbar = tqdm(total=num_epochs - epoch)
    while epoch < num_epochs:
        epoch = epoch + 1
    # resume training
        if (args.resume_train) and (epoch == 1):
            checkpoint = torch.load(args.resume_dir)
            epoch = checkpoint['epoch']
            learning_rate = checkpoint['learning_rate']
            optimizer.load_state_dict(checkpoint['optimizer'])
            step_idx = checkpoint['step_idx']
            model.load_state_dict(checkpoint['state_dict'])
            test_record = list(
                np.load(args.weights_dir + 'test_record.npy'))
            train_record = list(
                np.load(args.weights_dir + 'train_record.npy'))

        logging.info('epoch %d lr %e', epoch, learning_rate)

    # training
        train_acc_top1, train_acc_top5, train_obj = train(train_loader, model, criterion, optimizer)
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
                'step_idx': step_idx,
                'learning_rate': learning_rate,
                }, args, is_best)

        step_idx, learning_rate = utils.adjust_learning_rate(
            args, epoch, step_idx, learning_rate)

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        pbar.update(1)

def train(train_queue, model, criterion, optimizer):

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

    logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    utils.create_folder(args)
    main()

