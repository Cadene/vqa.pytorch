import os
import torch
import torch.utils.data as data
import copy

from .images import ImagesFolder, AbstractImagesDataset, default_loader
from .features import FeaturesDataset
from .vgenome_interim import vgenome_interim
from .vgenome_processed import vgenome_processed
from .coco import default_transform
from .utils import AbstractVQADataset

def raw(dir_raw):
    dir_img = os.path.join(dir_raw, 'images')
    os.system('wget http://visualgenome.org/static/data/dataset/image_data.json.zip -P '+dir_raw)
    os.system('wget http://visualgenome.org/static/data/dataset/question_answers.json.zip -P '+dir_raw)
    os.system('wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -P '+dir_raw)
    os.system('wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -P '+dir_raw)

    os.system('unzip '+os.path.join(dir_raw, 'image_data.json.zip')+' -d '+dir_raw)
    os.system('unzip '+os.path.join(dir_raw, 'question_answers.json.zip')+' -d '+dir_raw)
    os.system('unzip '+os.path.join(dir_raw, 'images.zip')+' -d '+dir_raw)
    os.system('unzip '+os.path.join(dir_raw, 'images2.zip')+' -d '+dir_raw)

    os.system('mv '+os.path.join(dir_raw, 'VG_100K')+' '+dir_img)

    #os.system('mv '+os.path.join(dir_raw, 'VG_100K_2', '*.jpg')+' '+dir_img)
    os.system('find '+os.path.join(dir_raw, 'VG_100K_2')+' -type f -name \'*\' -exec mv {} '+dir_img+' \\;')
    os.system('rm -rf '+os.path.join(dir_raw, 'VG_100K_2'))

    # remove images with 0 octet in a ugly but efficient way :')
    #print('for f in $(ls -lh '+dir_img+' | grep " 0 " | cut -s -f14 --delimiter=" "); do rm '+dir_img+'/${f}; done;')
    os.system('for f in $(ls -lh '+dir_img+' | grep " 0 " | cut -s -f14 --delimiter=" "); do echo '+dir_img+'/${f}; done;')
    os.system('for f in $(ls -lh '+dir_img+' | grep " 0 " | cut -s -f14 --delimiter=" "); do rm '+dir_img+'/${f}; done;')


class VisualGenome(AbstractVQADataset):

    def __init__(self, data_split, opt, dataset_img=None):
        super(VisualGenome, self).__init__(data_split, opt, dataset_img)

    def __getitem__(self, index):
        item_qa = self.dataset[index]
        item = {}
        if self.dataset_img is not None:
            item_img = self.dataset_img.get_by_name(item_qa['image_name'])
            item['visual'] = item_img['visual']
            # DEBUG
            #item['visual_debug'] = item_qa['image_name']
        item['question'] = torch.LongTensor(item_qa['question_wids'])
        # DEBUG
        #item['question_debug'] = item_qa['question']
        item['question_id'] = item_qa['question_id']
        item['answer'] = item_qa['answer_aid']
        # DEBUG
        #item['answer_debug'] = item_qa['answer']
        return item

    def _raw(self):
        raw(self.dir_raw)

    def _interim(self):
        vgenome_interim(self.opt)

    def _processed(self):
        vgenome_processed(self.opt)

    def __len__(self):
        return len(self.dataset)


class VisualGenomeImages(AbstractImagesDataset):

    def __init__(self, data_split, opt, transform=None, loader=default_loader):
        super(VisualGenomeImages, self).__init__(data_split, opt, transform, loader)
        self.dir_img = os.path.join(self.dir_raw, 'images')
        self.dataset = ImagesFolder(self.dir_img, transform=self.transform, loader=self.loader)
        self.name_to_index = self._load_name_to_index()

    def _raw(self):
        raw(self.dir_raw)

    def _load_name_to_index(self):
        self.name_to_index = {name:index for index, name in enumerate(self.dataset.imgs)}
        return self.name_to_index

    def __getitem__(self, index):
        item = self.dataset[index]
        return item

    def __len__(self):
        return len(self.dataset)


def factory(opt, vqa=False, transform=None):

    if vqa:
        dataset_img = factory(opt, vqa=False, transform=transform)
        return VisualGenome('train', opt, dataset_img)

    if opt['mode'] == 'img':
        if transform is None:
            transform = default_transform(opt['size'])

    elif opt['mode'] in ['noatt', 'att']:
        return FeaturesDataset('train', opt)
        
    else:
        raise ValueError


