import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.models import resnet18
import numpy as np

from torch.utils.tensorboard import SummaryWriter, writer


class CNN(torch.nn.Module):

    def __init__(self, in_channels=1):
        super(CNN, self).__init__()

        self.linear = nn.Linear(1, 2)  # 10 classes

    def forward(self, x):
        x = x.reshape(-1, 1)
        return self.linear(x)


class mydata(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super(mydata).__init__()

        self.data = torch.rand([512, 1])
        self.targets = torch.randint(1, 2, [512])

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        return img, target

    def __len__(self) -> int:
        return len(self.data)


parser = argparse.ArgumentParser(description='scheduler')
parser.add_argument('--scheduler', default="step", type=str, help='scheduler name')
args = parser.parse_args()

writer = SummaryWriter("logger/"+args.scheduler)

train_dataset = mydata()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)

model = CNN().cuda()
criterion = nn.CrossEntropyLoss()


def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0
    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (
                    1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))
    return lr


sc = args.scheduler
epochs = 200
lr = 0.5
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
if sc == "cosine":
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(step,
                                                epochs * len(train_loader),
                                                1,  # since lr_lambda computes multiplicative factor
                                                1e-6 / lr,
                                                warmup_steps=10 * len(train_loader))
    )
elif sc == "cosine_warmrestarts":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2)
elif sc == "step":
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 60, 0.1)

for epoch in range(epochs):
    for i, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()

        output = model(x)

        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if sc == "cosine_warmrestarts":
            scheduler.step(epoch + i / len(train_loader))
        else:
            scheduler.step()

    writer.add_scalar('loss', loss, epoch)
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
