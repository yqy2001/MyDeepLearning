"""
@Author:        禹棋赢
@StartTime:     2021/8/15 15:27
@Filename:      FashionMNIST.py
"""
import torch
from torch.utils import data
import torchvision
from torchvision import transforms

from easydict import EasyDict
from d2l import torch as d2l


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    d2l.plt.show()
    return axes


def get_fashion_mnist_labels(labels):
    """Return text labels for the Fashion-MNIST dataset. Input is a integer list"""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def load_MNIST(bsz):
    """
    FashionMNIST: train_dataset[0] have two elements, the first is a (1, 28, 28) image,
    the second is the corresponding label
    """
    transform = transforms.Compose(
        [
            transforms.Resize(28),
            transforms.ToTensor(),
        ]
    )
    train_data = torchvision.datasets.MNIST(root="~/data", train=True, transform=transform, download=True)
    test_data = torchvision.datasets.MNIST(root="~/data", train=False, transform=transform, download=True)
    train_loader = data.DataLoader(train_data, batch_size=bsz, shuffle=True, num_workers=2)
    test_loader = data.DataLoader(test_data, batch_size=bsz, shuffle=False, num_workers=2)

    dataloader = EasyDict(train=train_loader, test=test_loader)
    # X, y = next(iter(dataloader.train))
    # show_images(X.reshape(bsz, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

    print(len(train_data))
    for i in range(len(train_data)):
        if i < 5:
            print(train_data[i][0].shape)
        else:
            break

    return dataloader


if __name__ == '__main__':
    dataloader = load_MNIST(32)
    timer = d2l.Timer()
    for X, y in dataloader.train:
        continue
    print(f'{timer.stop():.2f} sec')
