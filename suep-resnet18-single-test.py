import argparse
import numpy as np
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


def to_numpy(t):
    return t.detach().cpu().numpy() if t.requires_grad else t.cpu().numpy()


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
              net,
              loader,
              prefix='Testing'):
    results = np.empty(3)
    tr = trange(len(loader), file=sys.stdout)
    accuracy = AverageMeter('Accuracy', ':1.5f')
    accuracy.reset()
    for (inputs, tracks, targets) in loader:
        inputs, tracks, targets = inputs.to(rank), tracks.to(rank), targets.to(rank)
        pos = targets == 0
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        outputs = F.softmax(outputs, dim=1)[:, 1]
        accuracy.update(predicted.eq(targets).sum().item()/targets.size(0),
                        targets.size(0))
        batched_results = np.hstack((to_numpy(outputs).reshape(-1, 1),
                                     to_numpy(tracks).reshape(-1, 1),
                                     to_numpy(targets).reshape(-1, 1)))
        tr.set_description('{0}: {1}'.format(prefix, accuracy))
        tr.update(1)
        results = np.vstack((results, batched_results))
    tr.close()
    print("Accuracy: {}".format(accuracy.average))
    return results


def main(config, name):
    rank = 0

    # Set model
    state_dict = torch.load('models/{}.pth'.format(name),
                            map_location=lambda s, loc: s)
    resnet18 = models.resnet18(pretrained=False)
    resnet18.conv1 = torch.nn.Conv2d(1,
                                     64,
                                     kernel_size=7,
                                     stride=2,
                                     padding=3,
                                     bias=False)
    resnet18.fc = torch.nn.Linear(512,
                                  2,
                                  bias=True)
    resnet18.load_state_dict(state_dict,
                             strict=True)
    resnet18 = resnet18.to(rank)

    # Set data
    test_loader = get_data_loader(config['dataset']['train'][rank],
                                  config['evaluation_pref']['batch_size'],
                                  config['evaluation_pref']['workers'],
                                  config['ssd_settings']['input_dimensions'],
                                  rank,
                                  flip_prob=.0,
                                  shuffle=False)

    results = run_epoch(rank, resnet18, test_loader)
    np.save('models/{}-results'.format(name), results)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        'Test SUEP Detection Model, ResNet18: Single Disco')
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
