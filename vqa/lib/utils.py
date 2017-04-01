import itertools
import collections
import torch
import numpy as np

def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key] 
    return dict_to

def merge_dict(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        d = dict(a)
        d.update({k: merge_dict(a.get(k, None), b[k]) for k in b})
    if isinstance(a, list) and isinstance(b, list):
        return b
        #return [merge_dict(x, y) for x, y in itertools.zip_longest(a, b)]
    return a if b is None else b

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    if target.dim() == 2: # multians option
        _, target = torch.max(target, 1)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def params_count(model):
    count = 0
    for p in model.parameters():
        c = 1
        for i in range(p.dim()):
            c *= p.size(i)
        count += c
    return count

def str2bool(v):
    if v is None:
        return v
    elif type(v) == bool:
        return v
    elif type(v) == str:
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def create_n_hot(idxs, N):
    out = np.zeros(N)
    for i in idxs:
        out[i] += 1
    return torch.Tensor(out/out.sum())

