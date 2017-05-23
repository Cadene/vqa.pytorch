import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    for fname in os.listdir(dir):
        if is_image_file(fname):
            images.append(fname)
    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImagesFolder(data.Dataset):

    def __init__(self, root, transform=None, loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        item = {}
        item['name'] = self.imgs[index] 
        item['path'] = os.path.join(self.root, item['name'])
        if self.loader is not None:
            item['visual']  = self.loader(item['path'])
            if self.transform is not None:
                item['visual'] = self.transform(item['visual'])
        return item

    def __len__(self):
        return len(self.imgs)


class AbstractImagesDataset(data.Dataset):

    def __init__(self, data_split, opt, transform=None, loader=default_loader):
        self.data_split = data_split
        self.opt = opt
        self.transform = transform
        self.loader = loader

        self.dir_raw = os.path.join(self.opt['dir'], 'raw')
        if not os.path.exists(self.dir_raw):
            self._raw()

    def get_by_name(self, image_name):
        index = self.name_to_index[image_name]
        return self[index]

    def _raw(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError