"""
@Author:        禹棋赢
@StartTime:     2021/8/5 19:23
@Filename:      main.py
"""
import argparse
import os

import torch
from easydict import EasyDict
from tqdm import tqdm

import sys

sys.path.append("../..")  # 为了引入上上级目录下的文件
sys.path.append("..")  # 为了引入上级目录下的文件
from CV.datasets.cifar import load_cifar10
from CV.CIFAR.models.resnet import ResNet50


class Processor(object):

    def __init__(self, model, dataset, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = model
        self.model.to(self.device)
        if self.device == 'cuda':
            self.model = torch.nn.DataParallel(self.model)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

        self.args = args
        self.dataset = dataset

        self.start_epoch = 1
        self.epoch = 0
        self.best_acc = 0.0

        if args.resume:
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.pth')
            self.model.load_state_dict(checkpoint['model'])
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']

    def process(self):
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            print("\nEpoch: %d/%d" % (epoch, self.args.epochs))
            self.epoch = epoch
            self.train()
            self.test()

    def train(self):
        self.model.train()
        train_loss = 0.0
        metrics = EasyDict(total=0, correct=0)
        for x, y in tqdm(self.dataset.train):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.loss_fn(outputs, y)
            loss.backward()
            self.optimizer.step()

            _, pred = outputs.max(1)
            metrics.total += y.shape[0]
            metrics.correct += torch.eq(pred, y).sum().item()
            train_loss += loss.item()

        print("train loss is {:2f}, train acc is {:2f}".format(train_loss, 100.*metrics.correct/metrics.total))

    def test(self):
        self.model.eval()
        metrics = EasyDict(total=0, correct=0)
        for x, y in tqdm(self.dataset.test):
            x, y = x.to(self.device), y.to(self.device)
            _, pred = self.model(x).max(1)
            metrics.total += y.shape[0]
            metrics.correct += torch.eq(pred, y).sum().item()

        acc = 100.0 * metrics.correct / metrics.total
        print("test acc is {:2f}".format(acc))
        if acc > self.best_acc:
            self.best_acc = acc
            print("This epoch's test acc is better, Saving...")
            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")
            torch.save({
                'model': self.model.state_dict(),
                'acc': acc,
                'epoch': self.epoch,
            }, "./checkpoint/ckpt.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="learning rate")
    parser.add_argument("--bsz", type=int, default=128, help="batch size")
    parser.add_argument("--resume", '-r', action='store_true', help="resume training from checkpoint")
    args = parser.parse_args()

    model = ResNet50()
    cifar10 = load_cifar10(args.bsz)
    p = Processor(model, cifar10, args)
    p.process()
