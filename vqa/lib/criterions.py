import torch
import torch.nn as nn

def factory(opt, cuda=True):
    criterion = nn.CrossEntropyLoss()
    if cuda:
        criterion = criterion.cuda()
    return criterion