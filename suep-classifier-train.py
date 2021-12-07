import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.models as models
import tqdm
import warnings
import yaml
import torch.cuda as tcuda

from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import DistributedOptimizer
from torch.cuda.amp import GradScaler, autocast
from tqdm import trange
from ssd.checkpoints import EarlyStopping
from ssd.layers.functions import PriorBox
from ssd.layers.modules import MultiBoxLoss
from ssd.layers.regularizers import FLOPRegularizer
from ssd.generator import CalorimeterJetDataset
from ssd.net import build_ssd
from ssd.qutils import get_delta, get_alpha, to_ternary
from utils import AverageMeter, IsValidFile, Plotting, get_data_loader, \
    set_logging


warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    module=r'.*'
)


def execute(rank,
            world_size,
            name,
            dataset,
            output,
            training_pref,
            ssd_settings,
            verbose):

    setup(rank, world_size)

    if rank == 0:
        logname = '{}/{}.log'.format(output['model'], name)
        logger = set_logging('Train_ResNet50', logname, verbose)

    plot = Plotting(save_dir=output['plots'])

    # Initialize dataset
    train_loader = get_data_loader(dataset['train'][rank],
                                   training_pref['batch_size_train'],
                                   training_pref['workers'],
                                   ssd_settings['input_dimensions'],
                                   ssd_settings['object_size'],
                                   rank,
                                   is_classifier=True,
                                   flip_prob=0.5,
                                   shuffle=True,
                                   return_pt=True)

    val_loader = get_data_loader(dataset['validation'][rank],
                                 training_pref['batch_size_validation'],
                                 training_pref['workers'],
                                 ssd_settings['input_dimensions'],
                                 ssd_settings['object_size'],
                                 rank,
                                 is_classifier=True,
                                 shuffle=False,
                                 return_pt=True)

    # Build SSD network
    model = models.resnet50(pretrained=True, progress=True)
    model.conv1 = nn.Conv2d(1,
                            64,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False)
    model.fc = nn.Linear(2048, ssd_settings['n_classes'], bias=True)
    model.fc.apply(weights_init)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(rank)
    if rank == 0:
        logger.debug('ResNet50 architecture:\n{}'.format(str(model)))

    # Data parallelization
    cudnn.benchmark = True
    net = DDP(model, device_ids=[rank])

    # Set training objective parameters
    optimizer = optim.SGD(net.parameters(),
                          lr=training_pref['learning_rate'],
                          momentum=training_pref['momentum'],
                          weight_decay=training_pref['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[20, 30, 50, 60,
                                                           70, 80, 90],
                                               gamma=0.5)
    if rank == 0:
        cp_es = EarlyStopping(patience=training_pref['patience'],
                              save_path='%s/%s.pth' % (output['model'], name))
    criterion = nn.CrossEntropyLoss().to(rank)
    scaler = GradScaler()
    verobse = verbose and rank == 0
    train_loss, val_loss = torch.empty(1, 0), torch.empty(1, 0)

    loss = AverageMeter('Loss', ':1.5f')
    acc = AverageMeter('Accuracy', ':1.5f')
    m = nn.Softmax(dim=1)

    for epoch in range(1, training_pref['max_epochs']+1):
        # Start model training
        net.train()
        loss.reset()

        if verbose:
            tr = trange(len(train_loader), file=sys.stdout)

        for images, targets in train_loader:
            count = len(targets)
            targets = tcuda.LongTensor(targets, device=rank)

            outputs = net(images)
            classification = torch.argmax(m(outputs), dim=1)

            l = criterion(outputs, targets)
            a = (targets == classification).sum() / count

            loss.update(l)
            acc.update(a)
            scaler.scale(l).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            info = 'Epoch {}, {}, {}'.format(epoch, loss, acc)
            if verbose:
                tr.set_description(info)
                tr.update(1)

        if rank == 0:
            logger.debug(info)
        tloss = torch.tensor([loss.avg]).unsqueeze(1)
        train_loss = torch.cat((train_loss, tloss), 1)
        if verbose:
            tr.close()

        net.eval()
        loss.reset()
        acc.reset()

        # Start model validation
        if verbose:
            tr = trange(len(val_loader), file=sys.stdout)

        with torch.no_grad():

            for images, targets in val_loader:
                count = len(targets)
                targets = tcuda.LongTensor(targets, device=rank)
                outputs = net(images)
                classification = torch.argmax(m(outputs), dim=1)
                l = criterion(outputs, targets)
                l = reduce_tensor(l.data)
                a = (targets == classification).sum() / count
                a = reduce_tensor(a.data)

                loss.update(l)
                acc.update(a)

                info = 'Validation, {}, {}'.format(loss, acc)
                if verbose:
                    tr.set_description(info)
                    tr.update(1)

            if rank == 0:
                logger.debug(info)
            vloss = torch.tensor([loss.avg]).unsqueeze(1)
            val_loss = torch.cat((val_loss, vloss), 1)
            if verbose:
                tr.close()

            plot.draw_loss(train_loss.cpu().numpy(),
                           val_loss.cpu().numpy(),
                           name)

            if rank == 0 and cp_es(vloss.sum(0), model):
                break

            dist.barrier()

        scheduler.step()
    cleanup()


def reduce_tensor(loss):
    loss = loss.clone()
    dist.all_reduce(loss)
    loss /= int(os.environ['WORLD_SIZE'])
    return loss


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '11223'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Train SUEP ResNet50 Classifier')
    parser.add_argument('name', type=str, help='Model name')
    parser.add_argument('-c', '--config',
                        action=IsValidFile,
                        type=str,
                        help='Path to config file',
                        default='ssd-config.yml')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Output verbosity')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    world_size = torch.cuda.device_count()

    mp.spawn(execute,
             args=(world_size,
                   args.name,
                   config['dataset'],
                   config['output'],
                   config['training_pref'],
                   config['ssd_settings'],
                   args.verbose),
             nprocs=world_size,
             join=True)
