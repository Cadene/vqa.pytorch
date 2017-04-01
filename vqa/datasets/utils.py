import os
import pickle
import torch
import torch.utils.data as data
import copy

class AbstractVQADataset(data.Dataset):

    def __init__(self, data_split, opt, dataset_img=None):
        self.data_split = data_split
        self.opt = copy.copy(opt)
        self.dataset_img = dataset_img

        self.dir_raw = os.path.join(self.opt['dir'], 'raw')
        if not os.path.exists(self.dir_raw):
            self._raw()
  
        self.dir_interim = os.path.join(self.opt['dir'], 'interim')
        if not os.path.exists(self.dir_interim):
            self._interim()
  
        self.dir_processed = os.path.join(self.opt['dir'], 'processed')
        self.subdir_processed = self.subdir_processed()
        if not os.path.exists(self.subdir_processed):
            self._processed()

        path_wid_to_word = os.path.join(self.subdir_processed, 'wid_to_word.pickle')
        path_word_to_wid = os.path.join(self.subdir_processed, 'word_to_wid.pickle')
        path_aid_to_ans  = os.path.join(self.subdir_processed, 'aid_to_ans.pickle')
        path_ans_to_aid  = os.path.join(self.subdir_processed, 'ans_to_aid.pickle')
        path_dataset     = os.path.join(self.subdir_processed, self.data_split+'set.pickle')
        
        with open(path_wid_to_word, 'rb') as handle:
            self.wid_to_word = pickle.load(handle)
  
        with open(path_word_to_wid, 'rb') as handle:
            self.word_to_wid = pickle.load(handle)
  
        with open(path_aid_to_ans, 'rb') as handle:
            self.aid_to_ans = pickle.load(handle)
  
        with open(path_ans_to_aid, 'rb') as handle:
            self.ans_to_aid = pickle.load(handle)
 
        with open(path_dataset, 'rb') as handle:
            self.dataset = pickle.load(handle)

    def _raw(self):
        raise NotImplementedError

    def _interim(self):
        raise NotImplementedError

    def _processed(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def subdir_processed(self):
        subdir = 'nans,' + str(self.opt['nans']) \
              + '_maxlength,' + str(self.opt['maxlength']) \
              + '_minwcount,' + str(self.opt['minwcount']) \
              + '_nlp,' + self.opt['nlp'] \
              + '_pad,' + self.opt['pad'] \
              + '_trainsplit,' + self.opt['trainsplit']
        subdir = os.path.join(self.dir_processed, subdir)
        return subdir