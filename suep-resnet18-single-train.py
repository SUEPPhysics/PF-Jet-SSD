import argparse
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn.functional as F
import yaml

from Disco import distance_corr
from ssd.genclassifier import SUEPDataset
from tqdm import trange
from torch.utils.data import DataLoader
from torchvision import models
from utils import IsValidFile


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt

    def reset(self):
        self.count = 0
        self.average = 0
        self.sum = 0

    def update(self, value, num=1):
        self.count += num
        self.sum += value * num
        self.average = self.sum / self.count

    def __str__(self):
        fmtstr = '{average' + self.fmt + '} ({name})'
        return fmtstr.format(**self.__dict__)


class Plotting():
    """Plots the training and test loss and accuracy per epoch"""

    def __init__(self, fname):
        self.fname = fname

    def draw(self, train_loss, train_accuracy, test_loss, test_accuracy):
        fig, axs = plt.subplots(2, 2, figsize=(25, 20))

        axs[0, 0].set_title('Train Loss')
        axs[0, 1].set_title('Training Accuracy')
        axs[1, 0].set_title('Test Loss')
        axs[1, 1].set_title('Test Accuracy')

        axs[0, 0].plot(train_loss)
        axs[0, 1].plot(train_accuracy)
        axs[1, 0].plot(test_loss)
        axs[1, 1].plot(test_accuracy)

        fig.savefig(self.fname)


def collate_fn(batch):
    transposed_data = list(zip(*batch))
    inp = torch.stack(transposed_data[0], 0)
    tgt = torch.stack(transposed_data[1], 0)
    tgt2 = torch.stack(transposed_data[2], 0)
    return inp, tgt, tgt2


def get_data_loader(hdf5_source_path,
                    batch_size,
                    num_workers,
                    input_dimensions,
                    rank=0,
                    flip_prob=None,
                    shuffle=True):
    dataset = SUEPDataset(torch.device(rank),
                          hdf5_source_path,
                          input_dimensions,
                          flip_prob=flip_prob)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      collate_fn=collate_fn,
                      num_workers=num_workers,
                      shuffle=shuffle)


def run_epoch(rank,
              epoch,
              net,
              criterion,
              optimizer,
              loader,
              prefix='Training',
              backprop=True,
              fp16=False):

    tr = trange(len(loader), file=sys.stdout)
    loss = AverageMeter('Loss', ':1.5f')
    accuracy = AverageMeter('Accuracy', ':1.5f')
    loss.reset()
    accuracy.reset()
    for (inputs, tracks, targets) in loader:
        inputs, tracks, targets = inputs.to(rank), tracks.to(rank), targets.to(rank)
        if backprop:
            optimizer.zero_grad()
        pos = targets == 0
        outputs = net(inputs)
        batch_loss = criterion(outputs, targets)
        corr = distance_corr(F.softmax(outputs, dim=1)[:, 0][pos],
                             tracks[pos], 1)
        if torch.isnan(corr):
            corr = 0
        batch_loss = batch_loss + corr
        if backprop:
            batch_loss.backward()
            optimizer.step()
        loss.update(batch_loss.item(), targets.size(0))
        _, predicted = outputs.max(1)
        accuracy.update(predicted.eq(targets).sum().item()/targets.size(0),
                        targets.size(0))
        tr.set_description('{0} epoch {1}: {2}, {3}'.format(
            prefix, epoch, loss, accuracy))
        tr.update(1)
    tr.close()

    return loss.average, accuracy.average


def main(config, name):
    epochs = config['training_pref']['max_epochs']
    rank = 0

    # Set model
    resnet18 = models.resnet18(pretrained=True)
    resnet18.conv1 = torch.nn.Conv2d(1,
                                     64,
                                     kernel_size=7,
                                     stride=2,
                                     padding=3,
                                     bias=False)
    resnet18.fc = torch.nn.Linear(512,
                                  2,
                                  bias=True)
    resnet18 = resnet18.to(rank)

    # Set loss
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet18.parameters(), lr=0.01)

    # Set data
    train_loader = get_data_loader(
        config['dataset']['train'][rank],
        config['training_pref']['batch_size_train'],
        config['training_pref']['workers'],
        config['ssd_settings']['input_dimensions'],
        rank,
        flip_prob=0.5,
        shuffle=True
    )

    val_loader = get_data_loader(
        config['dataset']['validation'][rank],
        config['training_pref']['batch_size_validation'],
        config['training_pref']['workers'],
        config['ssd_settings']['input_dimensions'],
        rank,
        shuffle=False
    )

    train_accuracy, val_accuracy = [], []
    train_loss, val_loss = [], []

    best_accuracy = 0.

    for epoch in range(epochs):

        loss, accuracy = run_epoch(rank,
                                   epoch,
                                   resnet18,
                                   criterion,
                                   optimizer,
                                   train_loader,
                                   backprop=True)
        train_loss.append(loss)
        train_accuracy.append(accuracy)

        loss, accuracy = run_epoch(rank,
                                   epoch,
                                   resnet18,
                                   criterion,
                                   optimizer,
                                   val_loader,
                                   backprop=False)
        val_loss.append(loss)
        val_accuracy.append(accuracy)

        if accuracy > best_accuracy:
            torch.save(resnet18.state_dict(),
                       'models/{}.pth'.format(name))
            best_accuracy = accuracy

    plot = Plotting('plots/{}.png'.format(name))
    plot.draw(train_loss,
              train_accuracy,
              val_loss,
              val_accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Train SUEP Detection Model, ResNet18: Single Disco')
    parser.add_argument('name',
                        help='Model name',
                        type=str)
    parser.add_argument('-c',
                        '--config',
                        action=IsValidFile,
                        default='ssd-config.yml',
                        help='Path to config file')
    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        help='Output verbosity')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    main(config, args.name)
