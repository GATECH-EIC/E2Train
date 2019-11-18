from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import os
import shutil
import argparse
import time
import logging
import models
import random
import numpy as np
from data import *
from functools import reduce
from tensorboardX import SummaryWriter
from meters import accuracy

def str2bool(s):
    return s.lower() in ['yes', '1', 'true', 'y']

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name])
                     )


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR10 training')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('arch', metavar='ARCH',
                        default='cifar10_rnn_gate_74',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: cifar10_rnn_gate_74)')
    parser.add_argument('--dataset', '-d', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset type')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')
    parser.add_argument('--iters', default=64000, type=int,
                        help='number of total iterations (default: 64,000)')
    parser.add_argument('--start-iter', default=0, type=int,
                        help='manual iter number (useful on restarts)')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', default=30, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to  latest checkpoint (default: None)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--step-ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--warm-up', action='store_true',
                        help='for n = 18, the model needs to warm up for 400 '
                             'iterations')
    parser.add_argument('--save-folder', default='save_checkpoints',
                        type=str,
                        help='folder to save the checkpoints')
    parser.add_argument('--eval-every', default=1000, type=int,
                        help='evaluate model every (default: 1000) iterations')
    parser.add_argument('--verbose', action="store_true",
                        help='print layer skipping ratio at training')
    parser.add_argument('--energy', default=1, type=int,
                        help='using energy as regularization term')
    parser.add_argument('--beta', default=1e-5, type=float,
                        help='coefficient')
    parser.add_argument('--minimum', default=100, type=float,
                        help='minimum')
    # Quantization of input, weight, bias and grad
    parser.add_argument('--num_bits', default=8, type=int,
                        help='precision of input/activation')
    parser.add_argument('--num_bits_weight', default=8, type=int,
                        help='precision of weight')
    parser.add_argument('--num_bits_grad', default=32, type=int,
                        help='precision of (layer) gradients')
    parser.add_argument('--biprecision', default=False, type=str2bool,
                        help='use biprecision or not')
    # Predictive (sign) SGD arguments
    parser.add_argument('--predictive_forward', default=False, type=str2bool,
                        help='use predictive net in forward pass')
    parser.add_argument('--predictive_backward', default=True, type=str2bool,
                        help='use predictive net in backward pass')
    parser.add_argument('--msb_bits', default=4, type=int,
                        help='precision of msb part of input')
    parser.add_argument('--msb_bits_weight', default=4, type=int,
                        help='precision of msb part of weight')
    parser.add_argument('--msb_bits_grad', default=16, type=int,
                        help='precision of msb part of (layer) gradient')
    parser.add_argument('--threshold', default=5e-5, type=float,
                        help='threshold to use full precision gradient calculation')
    parser.add_argument('--sparsify', default=False, type=str2bool,
                        help='sparsify the gradients using predictive net method')
    parser.add_argument('--sign', default=True, type=str2bool,
                        help='take sign before applying gradient')
    args = parser.parse_args()
    return args

training_cost = 0
skip_count = 0

def main():
    args = parse_args()

    descriptions = [
        args.arch,
        'g:%d' % args.num_bits_grad,
        'mg:%d' % args.msb_bits_grad,
        'th:%f' % args.threshold,
        'minimum:%f' % args.minimum,
        'lr:%f' % args.lr,
        'wd:%f' % args.weight_decay,
        'sr:%f' % args.step_ratio,
    ]
    args.exp_desc = '-'.join(filter(None, descriptions))

    save_path = args.save_path = os.path.join(args.save_folder, args.arch, args.exp_desc)
    os.makedirs(save_path, exist_ok=True)

    # config logger file
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    if args.cmd == 'train':
        logging.info('start training {}'.format(args.arch))
        run_training(args)

    elif args.cmd == 'test':
        logging.info('start evaluating {} with checkpoints from {}'.format(
            args.arch, args.resume))
        test_model(args)


def run_training(args):

    writer_path = os.path.join('runs', args.exp_desc + '-' + time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime()))
    writer = SummaryWriter(writer_path)

    signsgd_config = {
        'num_bits': args.num_bits,
        'num_bits_weight': args.num_bits_weight,
        'num_bits_grad': args.num_bits_grad,
        'biprecision': args.biprecision,
        'predictive_forward': args.predictive_forward,
        'predictive_backward': args.predictive_backward,
        'msb_bits': args.msb_bits,
        'msb_bits_weight': args.msb_bits_weight,
        'msb_bits_grad': args.msb_bits_grad,
        'threshold': args.threshold,
        'sparsify': args.sparsify,
        'sign': args.sign,
        'writer': writer,
    }

    # create model
    model = models.__dict__[args.arch](args.pretrained, **signsgd_config)
    model.install_gate()
    model = torch.nn.DataParallel(model).cuda()
    best_prec1 = 0

    # optionally resume from a checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume)
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))

            args.start_iter = 0
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            # translate(model, checkpoint)
            logging.info('=> loaded checkpoint `{}` (iter: {})'.format(
                args.resume, checkpoint['iter']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = True
    train_loader = prepare_train_data(dataset=args.dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                       model.parameters()), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    cp_energy_record = AverageMeter()
    skip_ratios = ListAverageMeter()

    end = time.time()
    dataloader_iterator = iter(train_loader)

    for i in range(0, args.iters):

        rand_flag = random.uniform(0, 1) > 0.5
        model.train()
        adjust_learning_rate(args, optimizer, i)

        try:
            input, target = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(train_loader)
            input, target = next(dataloader_iterator)

        # measuring data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = Variable(input, requires_grad=True).cuda()
        target_var = Variable(target).cuda()

        # compute output
        if rand_flag:
            optimizer.zero_grad()
            optimizer.step()
            global skip_count
            skip_count += 1
            continue

        output, masks, _ = model(input_var)

        energy_parameter = np.ones(35,)
        energy_parameter /= energy_parameter.max()

        energy_cost = 0
        energy_all = 0
        for layer in range(len(energy_parameter)):
            energy_cost += masks[layer].sum() * energy_parameter[layer]
            energy_all += reduce((lambda x, y: x * y), masks[layer].shape) * energy_parameter[layer]

        cp_energy = (energy_cost.item() / energy_all.item()) * 100
        global training_cost
        training_cost += (cp_energy / 100) * 0.51 * args.batch_size
        energy_cost *= args.beta
        if cp_energy <= args.minimum:
            reg = -1
        else:
            reg = 1
        if args.energy:
            loss = criterion(output, target_var) + energy_cost * reg
        else:
            loss = criterion(output, target_var)

        # collect skip ratio of each layer
        skips = [mask.data.le(0.5).float().mean() for mask in masks]
        if skip_ratios.len != len(skips):
            skip_ratios.set_len(len(skips))

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        writer.add_scalar('data/train_error', 100 - prec1, i-skip_count)
        writer.add_scalar('data/train_comp_using', cp_energy, i-skip_count)
        writer.add_scalar('data/train_cost_Gops', training_cost, i-skip_count)
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        cp_energy_record.update(cp_energy, 1)
        skip_ratios.update(skips, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # repackage hidden units for RNN Gate
        model.module.control.repackage_hidden()

        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        if i % args.print_freq == 0 or i == (args.iters - 1):
            logging.info("Iter: [{0}/{1}]\t"
                         "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                         "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                         "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                         "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                         'Energy_ratio: {cp_energy_record.val:.3f}({cp_energy_record.avg:.3f})\t'.format(
                            i,
                            args.iters,
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                            top1=top1,
                            cp_energy_record=cp_energy_record)
            )

        # evaluate every 1000 steps
        if (i % args.eval_every == 0 and i > 0) or (i == (args.iters-1)):
            prec1 = validate(args, test_loader, model, criterion)
            writer.add_scalar('data/test_error', 100 - prec1, i-skip_count)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            checkpoint_path = os.path.join(args.save_path,
                                           'checkpoint_{:05d}.pth.tar'.format(
                                               i))
            save_checkpoint({
                'iter': i,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            },
                is_best, filename=checkpoint_path)
            shutil.copyfile(checkpoint_path, os.path.join(args.save_path,
                                                          'checkpoint_latest'
                                                          '.pth.tar'))


def validate(args, test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    skip_ratios = ListAverageMeter()
    cp_energy_record = AverageMeter()

    # switch to evaluation mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            if i == len(test_loader) - 1:
                break
            target = target.cuda()
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()
            # compute output
            output, masks, logprobs = model(input_var)

            energy_parameter = np.ones(35, )
            energy_parameter /= energy_parameter.max()

            energy_cost = 0
            energy_all = 0
            for layer in range(len(energy_parameter)):
                energy_cost += masks[layer].sum() * energy_parameter[layer]
                energy_all += reduce((lambda x, y: x * y), masks[layer].shape) * energy_parameter[layer]
            cp_energy = (energy_cost.item() / energy_all.item()) * 100

            skips = [mask.data.le(0.5).float().mean().item() for mask in masks]

            if skip_ratios.len != len(skips):
                skip_ratios.set_len(len(skips))
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1,5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            skip_ratios.update(skips, input.size(0))
            losses.update(loss.data.item(), input.size(0))
            batch_time.update(time.time() - end)
            cp_energy_record.update(cp_energy, 1)
            end = time.time()

            if i % args.print_freq == 0 or (i == (len(test_loader) - 1)):
                logging.info(
                    'Test: [{}/{}]\t'
                    'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                    'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                    'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'
                    'Prec@5: {top1.val:.3f}({top5.avg:.3f})\t'
                    'Energy_ratio: {cp_energy_record.val:.3f}({cp_energy_record.avg:.3f})\t'.format(
                        i, len(test_loader), batch_time=batch_time,
                        loss=losses,
                        top1=top1, top5=top5,
                        cp_energy_record=cp_energy_record,
                    )
                )
        logging.info(' * Prec@1 {top1.avg:.3f}, Loss {loss.avg:.3f}'.format(
            top1=top1, loss=losses))

        skip_summaries = []
        for idx in range(skip_ratios.len):
            skip_summaries.append(1-skip_ratios.avg[idx])
        # compute `computational percentage`
        cp = ((sum(skip_summaries) + 1) / (len(skip_summaries) + 1)) * 100
        logging.info('*** Computation Percentage: {:.3f} %'.format(cp))

    return top1.avg


def test_model(args):
    signsgd_config = {
        'num_bits': args.num_bits,
        'num_bits_weight': args.num_bits_weight,
        'num_bits_grad': args.num_bits_grad,
        'biprecision': args.biprecision,
        'predictive_forward': args.predictive_forward,
        'predictive_backward': args.predictive_backward,
        'msb_bits': args.msb_bits,
        'msb_bits_weight': args.msb_bits_weight,
        'msb_bits_grad': args.msb_bits_grad,
        'threshold': args.threshold,
        'sparsify': args.sparsify,
        'sign': args.sign,
        'writer': None,
    }

    # create model
    model = models.__dict__[args.arch](args.pretrained, **signsgd_config)
    model.install_gate()
    model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            # translate(model, checkpoint)
            logging.info('=> loaded checkpoint `{}` (iter: {})'.format(
                args.resume, checkpoint['iter']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))
    cudnn.benchmark = False
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)
    criterion = nn.CrossEntropyLoss().cuda()

    validate(args, test_loader, model, criterion)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(save_path,
                                               'model_best_eic.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ListAverageMeter(object):
    """Computes and stores the average and current values of a list"""
    def __init__(self):
        self.len = 10000  # set up the maximum length
        self.reset()

    def reset(self):
        self.val = [0] * self.len
        self.avg = [0] * self.len
        self.sum = [0] * self.len
        self.count = 0

    def set_len(self, n):
        self.len = n
        self.reset()

    def update(self, vals, n=1):
        assert len(vals) == self.len, 'length of vals not equal to self.len'
        self.val = vals
        for i in range(self.len):
            self.sum[i] += self.val[i] * n
        self.count += n
        for i in range(self.len):
            self.avg[i] = self.sum[i] / self.count


def adjust_learning_rate(args, optimizer, _iter):
    """divide lr by 10 at 32k and 48k """
    if args.warm_up and (_iter < 400):
        lr = 0.01
    elif int(32000 * 4/3) <= _iter < int(48000 * 4/3):
        lr = args.lr * (args.step_ratio ** 1)
    elif _iter >= int(48000 * 4/3):
        lr = args.lr * (args.step_ratio ** 2)
    else:
        lr = args.lr

    # if _iter % args.eval_every == 0:
    #     logging.info('Iter [{}] learning rate = {}'.format(_iter, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
