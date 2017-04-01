import copy
import time
import json
import numpy as np
import os
from collections import defaultdict

class Experiment(object):

    def __init__(self, name, options=dict()):
        """ Create an experiment
        """
        super(Experiment, self).__init__()

        self.name = name
        self.options = options
        self.date_and_time = time.strftime('%d-%m-%Y--%H-%M-%S')

        self.info = defaultdict(dict)
        self.logged = defaultdict(dict)
        self.meters = defaultdict(dict)

    def add_meters(self, tag, meters_dict):
        assert tag not in (self.meters.keys())
        for name, meter in meters_dict.items():
            self.add_meter(tag, name, meter)

    def add_meter(self, tag, name, meter):
        assert name not in list(self.meters[tag].keys()), \
            "meter with tag {} and name {} already exists".format(tag, name)
        self.meters[tag][name] = meter

    def update_options(self, options_dict):
        self.options.update(options_dict)

    def log_meter(self, tag, name, n=1):
        meter = self.get_meter(tag, name)
        if name not in self.logged[tag]:
            self.logged[tag][name] = {}
        self.logged[tag][name][n] = meter.value()

    def log_meters(self, tag, n=1):
        for name, meter in self.get_meters(tag).items():
            self.log_meter(tag, name, n=n)

    def reset_meters(self, tag):
        meters = self.get_meters(tag)
        for name, meter in meters.items():
            meter.reset()
        return meters

    def get_meters(self, tag):
        assert tag in list(self.meters.keys())
        return self.meters[tag]

    def get_meter(self, tag, name):
        assert tag in list(self.meters.keys())
        assert name in list(self.meters[tag].keys())
        return self.meters[tag][name]

    def to_json(self, filename):
        os.system('mkdir -p ' + os.path.dirname(filename))
        var_dict = copy.copy(vars(self))
        var_dict.pop('meters')
        for key in ('viz', 'viz_dict'):
            if key in list(var_dict.keys()):
                var_dict.pop(key)
        with open(filename, 'w') as f:
            json.dump(var_dict, f)

    def from_json(filename):
        with open(filename, 'r') as f:
            var_dict = json.load(f)
        xp = Experiment('')
        xp.date_and_time = var_dict['date_and_time']
        xp.logged        = var_dict['logged']
        # TODO: Remove
        if 'info' in var_dict:
            xp.info          = var_dict['info']
        xp.options       = var_dict['options']
        xp.name          = var_dict['name']
        return xp


class AvgMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def value(self):
        return self.avg


class SumMeter(object):
    """Computes and stores the sum and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def value(self):
        return self.sum


class ValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = val

    def value(self):
        return self.val