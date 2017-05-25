import os
import time
import argparse
import json
import random
import numpy as np
import colorlover as cl

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, plot
from plotly import tools

from vqa.lib.logger import Experiment

def load_accs_oe(path_logger):
    dir_xp = os.path.dirname(path_logger)
    epochs = []
    for name in os.listdir(dir_xp):
        if name.startswith('epoch'):
            epochs.append(name)
    epochs = sorted(epochs, key=lambda x: float(x.split('_')[1]))
    accs = {}
    for i, epoch in enumerate(epochs):
        epoch_id = i+1
        path_acc = os.path.join(dir_xp, epoch, 'OpenEnded_mscoco_val2014_model_accuracy.json')
        if os.path.exists(path_acc):
            with open(path_acc, 'r') as f:
                data = json.load(f)
                accs[epoch_id] = data['overall']
    return accs

def sort(dict_):
    return [v for k,v in sorted(dict_.items(), \
                           key=lambda x: float(x[0]))]

def reduce(list_, num=15):
    tmp = []
    for i, val in enumerate(list_):
        if i < num:
            tmp.append(val)
    return tmp

# Display accuracy & loss of one exp
def visu_one_exp(path_logger, path_visu, auto_open=True):
    xp = Experiment.from_json(path_logger)
    xp.logged['val']['acc1_oe'] = load_accs_oe(path_logger)
    train_acc1 = sort(xp.logged['train']['acc1'])
    val_acc1   = sort(xp.logged['val']['acc1_oe'])
    train_loss = sort(xp.logged['train']['loss'])
    val_loss   = sort(xp.logged['val']['loss'])
    train_data_x = list(range(1, len(train_acc1)+1))
    val_data_x   = list(range(1, len(val_acc1)+1))
  
    fig = tools.make_subplots(rows=1,
                              cols=2,
                              subplot_titles=('Accuracy top1', 'Loss'))
    # blue rgb(31, 119, 180)
    # orange rgb(255, 127, 14)

    train_acc1_trace = go.Scatter(
        x=train_data_x, 
        y=train_acc1,
        name='train accuracy top1'
    )
    val_acc1_trace = go.Scatter(
        x=val_data_x, 
        y=val_acc1,
        name='val accuracy top1',
        line = dict(
            color = ('rgb(255, 127, 14)'),
        )
    )
    best_val_acc1_trace = go.Scatter(
        x=[np.argmax(val_acc1)+1], 
        y=[max(val_acc1)],
        mode='markers',
        name='best val accuracy top1',
        marker = dict(
            color = 'rgb(255, 127, 14)',
            size = 10
        )
    )

    val_loss_trace = go.Scatter(
        x=val_data_x, 
        y=val_loss,
        name='val loss'
    )
    train_loss_trace = go.Scatter(
        x=train_data_x, 
        y=train_loss,
        name='train loss'
    )

    fig.append_trace(train_acc1_trace, 1, 1)
    fig.append_trace(val_acc1_trace, 1, 1)
    fig.append_trace(best_val_acc1_trace, 1, 1)
    
    fig.append_trace(train_loss_trace, 1, 2)
    fig.append_trace(val_loss_trace, 1, 2)

    plot(fig, filename=path_visu, auto_open=auto_open)

    return train_acc1, val_acc1

# Display accuracy & loss of one exp
def visu_exps(list_path_logger, path_visu, auto_open=True):
    fig = tools.make_subplots(rows=2,
                          cols=2,
                          subplot_titles=('Val accuracy top1',
                                          'Val loss',
                                          'Train accuracy top1',
                                          'Train loss'))
    num_xp = len(list_path_logger)
    if num_xp < 3: # cl.scales not accept
        num_xp = 3
    list_color = cl.scales[str(num_xp)]['qual']['Paired']

    for i, path_logger in enumerate(list_path_logger):
        name = path_logger.split('/')[-2]

        xp = Experiment.from_json(path_logger)
        xp.logged['val']['acc1_oe'] = load_accs_oe(path_logger)
        train_acc1 = sort(xp.logged['train']['acc1'])
        val_acc1   = sort(xp.logged['val']['acc1_oe'])
        train_loss = sort(xp.logged['train']['loss'])
        val_loss   = sort(xp.logged['val']['loss'])
        train_data_x = list(range(1, len(train_acc1)+1))
        val_data_x   = list(range(1, len(val_acc1)+1))

        train_acc1_trace = go.Scatter(
            x=train_data_x, 
            y=train_acc1,
            name='train acc: '+name,
            line=dict(
                color=list_color[i]
            )
        )
        val_acc1_trace = go.Scatter(
            x=val_data_x, 
            y=val_acc1,
            name='val acc: '+name,
            line=dict(
                color=list_color[i]
            )
        )
        best_val_acc1_trace = go.Scatter(
            x=[np.argmax(val_acc1)+1], 
            y=[max(val_acc1)],
            mode='markers',
            name='best val acc: '+name,
            marker = dict(
                color = list_color[i],
                size = 10
            )
        )

        val_loss_trace = go.Scatter(
            x=val_data_x, 
            y=val_loss,
            name='val loss: '+name,
            line=dict(
                color=list_color[i]
            )
        )
        train_loss_trace = go.Scatter(
            x=train_data_x, 
            y=train_loss,
            name='train loss: '+name,
            line=dict(
                color=list_color[i]
            )
        )

        fig.append_trace(val_acc1_trace, 1, 1)
        fig.append_trace(best_val_acc1_trace, 1, 1)
        fig.append_trace(train_acc1_trace, 2, 1)

        fig.append_trace(val_loss_trace, 1, 2)
        fig.append_trace(train_loss_trace, 2, 2)

    plot(fig, filename=path_visu, auto_open=auto_open)

def main_one_exp(dir_logs, path_visu=None, refresh_freq=60):
    if path_visu is None:
        path_visu = os.path.join(dir_logs, 'visu.html')
    
    path_logger = os.path.join(dir_logs, 'logger.json')

    i = 1
    print('Create visu to ' + path_visu)
    while True:
        train_acc1, val_acc1 = visu_one_exp(path_logger, path_visu, auto_open=(i==1))
        print('# Visu iteration (refresh every {} sec): {}'.format(refresh_freq, i))
        print('Max Val OpenEnded-Accuracy Top1: {}'.format(max(val_acc1)))
        print('Max Train Accuracy Top1: {}'.format(max(train_acc1)))
        i += 1
        time.sleep(refresh_freq)

def main_exps(list_dir_logs, path_visu=None, refresh_freq=60):
    if path_visu is None:
        path_visu = os.path.join(os.path.dirname(list_dir_logs[0]), 'visu.html')

    list_path_logger = []
    for dir_logs in list_dir_logs:
        list_path_logger.append(os.path.join(dir_logs, 'logger.json'))

    i = 1
    print('Create visu to ' + path_visu)
    while True:
        visu_exps(list_path_logger, path_visu, auto_open=(i==1))
        print('# Visu iteration (refresh every {} sec): {}'.format(refresh_freq, i))
        i += 1
        time.sleep(refresh_freq)


##########################################################################
# Main
##########################################################################

parser = argparse.ArgumentParser(
    description='Create html visu files',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir_logs', type=str,
                    help='''First mode: dir to logs of an experiment (ex: logs/vqa/mutan)'''
                         '''Second mode: add several dirs to create a comparativ visualisation (ex: logs/vqa/mutan,logs/vqa/mlb)''')
parser.add_argument('--refresh_freq', '-f', default=60, type=int,
                    help='refresh frequency in seconds')
parser.add_argument('--path_visu', default=None,
                    help='path to the html file (default: visu.html in dir_logs)')

def main():
    global args
    args = parser.parse_args()

    list_dir_logs = args.dir_logs.split(',')

    if len(list_dir_logs) == 1:
        main_one_exp(args.dir_logs,
                     path_visu=args.path_visu,
                     refresh_freq=args.refresh_freq)
    else:
        main_exps(list_dir_logs,
                  path_visu=args.path_visu,
                  refresh_freq=args.refresh_freq)

if __name__ == '__main__':
    main()
