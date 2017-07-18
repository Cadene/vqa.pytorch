import os
import numpy as np
import h5py
import torch
import torch.utils.data as data

class FeaturesDataset(data.Dataset):

    def __init__(self, data_split, opt):
        self.data_split = data_split
        self.opt = opt
        self.dir_extract = os.path.join(self.opt['dir'],
                                      'extract',
                                      'arch,' + self.opt['arch'])
        if 'size' in opt:
            self.dir_extract += '_size,' + str(opt['size'])
        self.path_hdf5 = os.path.join(self.dir_extract,
                                      data_split + 'set.hdf5')
        assert os.path.isfile(self.path_hdf5), \
               'File not found in {}, you must extract the features first with extract.py'.format(self.path_hdf5)
        self.hdf5_file = h5py.File(self.path_hdf5, 'r')#, driver='mpio', comm=MPI.COMM_WORLD)
        self.dataset_features = self.hdf5_file[self.opt['mode']]
        self.index_to_name, self.name_to_index = self._load_dicts()

    def _load_dicts(self):
        self.path_fname = os.path.join(self.dir_extract,
                                       self.data_split + 'set.txt')
        with open(self.path_fname, 'r') as handle:
            self.index_to_name = handle.readlines()
        self.index_to_name = [name[:-1] for name in self.index_to_name] # remove char '\n'
        self.name_to_index = {name:index for index,name in enumerate(self.index_to_name)}
        return self.index_to_name, self.name_to_index

    def __getitem__(self, index):
        item = {}
        item['name'] = self.index_to_name[index]
        item['visual'] = self.get_features(index)
        #item = torch.Tensor(self.get_features(index))
        return item

    def get_features(self, index):
        return torch.Tensor(self.dataset_features[index])

    def get_features_old(self, index):
        try:
            self.features_array
        except AttributeError:
            if self.opt['mode'] == 'att':
                self.features_array = np.zeros((2048,14,14), dtype='f')
            elif self.opt['mode'] == 'noatt':
                self.features_array = np.zeros((2048), dtype='f')

        if self.opt['mode'] == 'att':
            self.dataset_features.read_direct(self.features_array,
                                              np.s_[index,:2048,:14,:14],
                                              np.s_[:2048,:14,:14])
        elif self.opt['mode'] == 'noatt':
            self.dataset_features.read_direct(self.features_array,
                                              np.s_[index,:2048],
                                              np.s_[:2048])
        return self.features_array
        

    def get_by_name(self, image_name):
        index = self.name_to_index[image_name]
        return self[index]

    def __len__(self):
        return self.dataset_features.shape[0]
