import os
# from mpi4py import MPI
import numpy as np
import h5py
import torch.utils.data as data
import torchvision.transforms as transforms

from ..lib import utils
from .images import ImagesFolder, AbstractImagesDataset, default_loader
from .features import FeaturesDataset

def split_name(data_split):
    if data_split in ['train', 'val']:
        return data_split + '2014'
    elif data_split == 'test':
        return data_split + '2015'
    else:
        assert False, 'data_split {} not exists'.format(data_split)


class COCOImages(AbstractImagesDataset):

    def __init__(self, data_split, opt, transform=None, loader=default_loader):
        self.split_name = split_name(data_split)
        super(COCOImages, self).__init__(data_split, opt, transform, loader)
        self.dir_split = self.get_dir_data()
        self.dataset = ImagesFolder(self.dir_split, transform=self.transform, loader=self.loader)
        self.name_to_index = self._load_name_to_index()

    def get_dir_data(self):
        return os.path.join(self.dir_raw, self.split_name)

    def _raw(self):
        if self.data_split in ['train', 'val']:
            os.system('wget http://msvocds.blob.core.windows.net/coco2014/{}.zip -P {}'.format(self.split_name, self.dir_raw))
        elif self.data_split == 'test':
            os.system('wget http://msvocds.blob.core.windows.net/coco2015/test2015.zip -P '+self.dir_raw)
        else:
            assert False, 'data_split {} not exists'.format(self.data_split)
        os.system('unzip '+os.path.join(self.dir_raw, self.split_name+'.zip')+' -d '+self.dir_raw)

    def _load_name_to_index(self):
        self.name_to_index = {name:index for index, name in enumerate(self.dataset.imgs)}
        return self.name_to_index

    def __getitem__(self, index):
        item = self.dataset[index]
        item['name'] = os.path.join(self.split_name, item['name'])
        return item

    def __len__(self):
        return len(self.dataset)


class COCOTrainval(data.Dataset):

    def __init__(self, trainset, valset):
        self.trainset = trainset
        self.valset = valset

    def __getitem__(self, index):
        if index < len(self.trainset):
            item = self.trainset[index]
        else:
            item = self.valset[index - len(self.trainset)]
        return item

    def get_by_name(self, image_name):
        if image_name in self.trainset.name_to_index:
            index = self.trainset.name_to_index[image_name]
            item = self.trainset[index]
            return item
        elif image_name in self.valset.name_to_index:
            index = self.valset.name_to_index[image_name]
            item = self.valset[index]
            return item
        else:
            raise ValueError

    def __len__(self):
        return len(self.trainset) + len(self.valset)


def default_transform(size):
    transform = transforms.Compose([
        transforms.Scale(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # resnet imagnet
                             std=[0.229, 0.224, 0.225])
    ])
    return transform

def factory(data_split, opt, transform=None):
    if data_split == 'trainval':
        trainset = factory('train', opt, transform)
        valset = factory('val', opt, transform)
        return COCOTrainval(trainset, valset)
    elif data_split in ['train', 'val', 'test']:
        if opt['mode'] == 'img':
            if transform is None:
                transform = default_transform(opt['size'])
            return COCOImages(data_split, opt, transform)
        elif opt['mode'] in ['noatt', 'att']:
            return FeaturesDataset(data_split, opt)
        else:
            raise ValueError
    else:
        raise ValueError
