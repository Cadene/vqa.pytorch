import argparse
import os
import time
import h5py
import numpy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import vqa.datasets.coco as coco
from vqa.lib.dataloader import DataLoader
from vqa.models.utils import ResNet
from vqa.lib.logger import AvgMeter

model_names = sorted(name for name in models.__dict__
    if name.islower() and name.startswith("resnet")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Extract')
parser.add_argument('--dir_data', default='data/coco', metavar='DIR',
                    help='dir dataset: mscoco or visualgenome')
parser.add_argument('--data_split', default='train', type=str,
                    help='Options: (default) train | val | test')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet152',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet152)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--batch_size', '-b', default=80, type=int, metavar='N',
                    help='mini-batch size (default: 80)')
parser.add_argument('--mode', default='both', type=str,
                    help='Options: att | noatt | (default) both')


def main():
    args = parser.parse_args()

    print("=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=True)
    model = ResNet(model, False)
    model = nn.DataParallel(model).cuda()

    #extract_name = 'arch,{}_layer,{}_resize,{}'.format()
    extract_name = 'arch,{}'.format(args.arch)

    #dir_raw = os.path.join(args.dir_data, 'raw')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = coco.COCOImages(args.data_split, dict(dir=args.dir_data), 
        transform=transforms.Compose([
            transforms.Scale(448),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            normalize,
        ]))

    data_loader = DataLoader(dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    dir_extract = os.path.join(args.dir_data, 'extract', extract_name)
    path_file = os.path.join(dir_extract, args.data_split + 'set')
    os.system('mkdir -p ' + dir_extract)

    extract(data_loader, model, path_file, args.mode)


def extract(data_loader, model, path_file, mode):
    path_hdf5 = path_file + '.hdf5'
    path_txt = path_file + '.txt'
    hdf5_file = h5py.File(path_hdf5, 'w')

    nb_images = len(data_loader.dataset)
    if mode == 'both' or mode == 'att':
        shape_att = (nb_images, 2048, 14, 14)
        hdf5_att = hdf5_file.create_dataset('att', shape_att,
                                            dtype='f')#, compression='gzip')
    if mode == 'both' or mode == 'noatt':
        shape_noatt = (nb_images, 2048)
        hdf5_noatt = hdf5_file.create_dataset('noatt', shape_noatt,
                                              dtype='f')#, compression='gzip')

    model.eval()

    batch_time = AvgMeter()
    data_time  = AvgMeter()
    begin = time.time()
    end = time.time()

    idx = 0
    for i, input in enumerate(data_loader):
        input_var = torch.autograd.Variable(input['visual'], volatile=True)
        output_att = model(input_var)

        nb_regions = output_att.size(2) * output_att.size(3)
        output_noatt = output_att.sum(3).sum(2).div(nb_regions).view(-1, 2048)
        
        batch_size = output_att.size(0)
        if mode == 'both' or mode == 'att':
            hdf5_att[idx:idx+batch_size]   = output_att.data.cpu().numpy()
        if mode == 'both' or mode == 'noatt':
            hdf5_noatt[idx:idx+batch_size] = output_noatt.data.cpu().numpy()
        idx += batch_size

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 1 == 0:
            print('Extract: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                   i, len(data_loader),
                   batch_time=batch_time,
                   data_time=data_time,))
            
    hdf5_file.close()

    # Saving image names in the same order than extraction
    with open(path_txt, 'w') as handle:
        for name in data_loader.dataset.dataset.imgs:
            handle.write(name + '\n')

    end = time.time() - begin
    print('Finished in {}m and {}s'.format(int(end/60), int(end%60)))


if __name__ == '__main__':
    main()