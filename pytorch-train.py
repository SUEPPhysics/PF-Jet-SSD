import argparse
import h5py
import numpy as np
import pathlib
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import tqdm
import yaml

from tqdm import trange
from torch.autograd import Variable

from ssd.checkpoints import EarlyStopping
from ssd.layers.modules import MultiBoxLoss
from ssd.generator import CalorimeterJetDataset
from ssd.net import build_ssd
from utils import Plotting


def adjust_learning_rate(optimizer, epoch, learning_rates):
    for step in learning_rates:
        if step['epoch'] == epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = step['rate']


def collate_fn(batch):
    transposed_data = list(zip(*batch))
    inp = torch.stack(transposed_data[0], 0)
    tgt = list(transposed_data[1])
    return inp, tgt


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()


def get_data_loader(source_path, batch_size, num_workers, input_dimensions,
                    object_size, shuffle=True):
    h5 = h5py.File(source_path, 'r')
    generator = CalorimeterJetDataset(input_dimensions, object_size,
                                      hdf5_dataset=h5)
    return torch.utils.data.DataLoader(generator,
                                       batch_size=batch_size,
                                       collate_fn=collate_fn,
                                       shuffle=shuffle,
                                       num_workers=num_workers), h5


def batch_step(x, y, optimizer, net, criterion):
    x = Variable(x.cuda())
    y = [Variable(ann.cuda()) for ann in y]
    if optimizer:
        optimizer.zero_grad()
    output = net(x)
    return criterion(output, y)


def execute(model_name, qtype, dataset, output, training_pref, ssd_settings,
            trained_model_path=None):

    ssd_settings['n_classes'] += 1
    quantized = (qtype == 'binary') or (qtype == 'ternary')
    plot = Plotting(save_dir=output['plots'])

    # Initialize dataset
    train_loader, h5t = get_data_loader(dataset['train'],
                                        training_pref['batch_size'],
                                        training_pref['workers'],
                                        ssd_settings['input_dimensions'],
                                        ssd_settings['object_size'])
    val_loader, h5v = get_data_loader(dataset['validation'],
                                      training_pref['batch_size'],
                                      training_pref['workers'],
                                      ssd_settings['input_dimensions'],
                                      ssd_settings['object_size'],
                                      shuffle=False)

    # Build SSD network
    ssd_net = build_ssd('train', ssd_settings, qtype=qtype)
    print(ssd_net)

    with open('{}/{}.txt'.format(output['model'], model_name), 'w') as f:
        f.write(str(ssd_net))

    # Initialize weights
    if trained_model_path:
        ssd_net.load_weights(trained_model_path)
    else:
        ssd_net.vgg.apply(weights_init)
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    # Data parallelization
    cudnn.benchmark = True
    net = torch.nn.DataParallel(ssd_net)
    net = net.cuda()

    # Print total number of parameters
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total network parameters: %s' % total_params)

    # Set training objective parameters
    optimizer = optim.SGD(net.parameters(), lr=1e-3,
                          momentum=training_pref['momentum'],
                          weight_decay=training_pref['weight_decay'])
    cp_es = EarlyStopping(patience=training_pref['patience'],
                          save_path='%s/%s.pth' % (output['model'], model_name))
    criterion = MultiBoxLoss(ssd_settings['n_classes'],
                             min_overlap=ssd_settings['overlap_threshold'])

    train_loss, val_loss = torch.empty(3, 0), torch.empty(3, 0)

    for epoch in range(1, training_pref['max_epochs']+1):

        adjust_learning_rate(optimizer, epoch, training_pref['learning_rates'])

        # Start model training
        tr = trange(len(train_loader)*training_pref['batch_size'],
                    file=sys.stdout)
        tr.set_description('Epoch {}'.format(epoch))
        all_epoch_loss = torch.zeros(3)
        net.train()

        for batch_index, (images, targets) in enumerate(train_loader):

            l, c, r = batch_step(images, targets, optimizer, net, criterion)
            loss = l + c + r
            loss.backward()

            if quantized:
                for p in list(net.parameters()):
                    if hasattr(p, 'org'):
                        p.data.copy_(p.org)

            optimizer.step()

            if quantized:
                for p in list(net.parameters()):
                    if hasattr(p, 'org'):
                        p.org.copy_(p.data.clamp_(-1, 1))

            all_epoch_loss += torch.tensor([l.item(), c.item(), r.item()])
            av_epoch_loss = all_epoch_loss / (batch_index + 1)

            tr.set_description(
                ('Epoch {} Loss {:.5f} Localization {:.5f} ' +
                 'Classification {:.5f} Regresion {:.5f}').format(
                 epoch, av_epoch_loss.sum(), av_epoch_loss[0],
                 av_epoch_loss[1], av_epoch_loss[2]))
            tr.update(len(images))

        train_loss = torch.cat((train_loss, av_epoch_loss.unsqueeze(1)), 1)
        tr.close()

        # Start model validation
        tr = trange(len(val_loader)*training_pref['batch_size'],
                    file=sys.stdout)
        tr.set_description('Validation')
        all_epoch_loss = torch.zeros(3)
        net.eval()

        with torch.no_grad():
            for batch_index, (images, targets) in enumerate(val_loader):

                l, c, r = batch_step(images, targets, None, net, criterion)
                all_epoch_loss += torch.tensor([l.item(), c.item(), r.item()])
                av_epoch_loss = all_epoch_loss / (batch_index + 1)

                tr.set_description(
                    ('Validation Loss {:.5f} Localization {:.5f} ' +
                     'Classification {:.5f} Regresion {:.5f}').format(
                     av_epoch_loss.sum(), av_epoch_loss[0], av_epoch_loss[1],
                     av_epoch_loss[2]))
                tr.update(len(images))

            val_loss = torch.cat((val_loss, av_epoch_loss.unsqueeze(1)), 1)
            tr.close()

            plot.draw_loss(train_loss.cpu().numpy(),
                           val_loss.cpu().numpy(),
                           type=qtype)

            if cp_es(av_epoch_loss.sum(0), ssd_net):
                break

    h5t.close()
    h5v.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Train Single Shot Jet Detection Model')
    parser.add_argument('name', type=str, help='Model name')
    parser.add_argument('qtype', type=str,
                        choices={'full', 'ternary', 'binary'},
                        help='Type of quantization')
    parser.add_argument('config', type=str, help="Path to config file")
    parser.add_argument('-m', '--pre-trained', type=str,
                        default=None, help='Path to pre-trained model',
                        dest='trained_model_path')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    execute(args.name,
            args.qtype,
            config['dataset'],
            config['output'],
            config['training_pref'],
            config['ssd_settings'],
            trained_model_path=args.trained_model_path)
