import argparse
import os
import shutil
import yaml
import json
import click
from pprint import pprint

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import vqa.lib.engine as engine
import vqa.lib.utils as utils
import vqa.lib.logger as logger
import vqa.lib.criterions as criterions
import vqa.datasets as datasets
import vqa.models as models

parser = argparse.ArgumentParser(
    description='Train/Evaluate models',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
##################################################
# yaml options file contains all default choices #
parser.add_argument('--path_opt', default='options/vqa/default.yaml', type=str, 
                    help='path to a yaml options file')
################################################
# change cli options to modify default choices #
# logs options
parser.add_argument('--dir_logs', type=str, help='dir logs')
# data options
parser.add_argument('--vqa_trainsplit', type=str, choices=['train','trainval'])
# model options
parser.add_argument('--arch', choices=models.model_names,
                    help='vqa model architecture: ' +
                        ' | '.join(models.model_names))
parser.add_argument('--st_type',
                    help='skipthoughts type')
parser.add_argument('--st_dropout', type=float)
parser.add_argument('--st_fixed_emb', default=None, type=utils.str2bool,
                    help='backprop on embedding')
# optim options
parser.add_argument('-lr', '--learning_rate', type=float,
                    help='initial learning rate')
parser.add_argument('-b', '--batch_size', type=int,
                    help='mini-batch size')
parser.add_argument('--epochs', type=int,
                    help='number of total epochs to run')
# options not in yaml file          
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint')
parser.add_argument('--save_model', default=True, type=utils.str2bool,
                    help='able or disable save model and optim state')
parser.add_argument('--save_all_from', type=int,
                    help='''delete the preceding checkpoint until an epoch,'''
                         ''' then keep all (useful to save disk space)')''')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation and test set')
parser.add_argument('-j', '--workers', default=2, type=int,
                    help='number of data loading workers')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    help='print frequency')
################################################
parser.add_argument('-ho', '--help_opt', dest='help_opt', action='store_true',
                    help='show selected options before running')

best_acc1 = 0

def main():
    global args, best_acc1
    args = parser.parse_args()

    #########################################################################################
    # Create options
    #########################################################################################

    options = {
        'vqa' : {
            'trainsplit': args.vqa_trainsplit
        },
        'logs': {
            'dir_logs': args.dir_logs
        },
        'model': {
            'arch': args.arch,
            'seq2vec': {
                'type': args.st_type,
                'dropout': args.st_dropout,
                'fixed_emb': args.st_fixed_emb
            }
        },
        'optim': {
            'lr': args.learning_rate,
            'batch_size': args.batch_size,
            'epochs': args.epochs
        }
    }
    if args.path_opt is not None:
        with open(args.path_opt, 'r') as handle:
            options_yaml = yaml.load(handle)
        options = utils.update_values(options, options_yaml)
    print('## args'); pprint(vars(args))
    print('## options'); pprint(options)
    if args.help_opt:
        return

    # Set datasets options
    if 'vgenome' not in options:
        options['vgenome'] = None

    #########################################################################################
    # Create needed datasets
    #########################################################################################

    trainset = datasets.factory_VQA(options['vqa']['trainsplit'],
                                    options['vqa'],
                                    options['coco'],
                                    options['vgenome'])
    train_loader = trainset.data_loader(batch_size=options['optim']['batch_size'],
                                        num_workers=args.workers,
                                        shuffle=True)                                      

    if options['vqa']['trainsplit'] == 'train':
        valset = datasets.factory_VQA('val', options['vqa'], options['coco'])
        val_loader = valset.data_loader(batch_size=options['optim']['batch_size'],
                                        num_workers=args.workers)

    if options['vqa']['trainsplit'] == 'trainval' or args.evaluate:
        testset = datasets.factory_VQA('test', options['vqa'], options['coco'])
        test_loader = testset.data_loader(batch_size=options['optim']['batch_size'],
                                          num_workers=args.workers)
    
    #########################################################################################
    # Create model, criterion and optimizer
    #########################################################################################
    
    model = models.factory(options['model'],
                           trainset.vocab_words(), trainset.vocab_answers(),
                           cuda=True, data_parallel=True)
    criterion = criterions.factory(options['vqa'], cuda=True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            options['optim']['lr'])
    
    #########################################################################################
    # args.resume: resume from a checkpoint OR create logs directory
    #########################################################################################

    exp_logger = None
    if args.resume:
        args.start_epoch, best_acc1, exp_logger = load_checkpoint(model.module, optimizer,
            os.path.join(options['logs']['dir_logs'], args.resume))
    else:
        # Or create logs directory
        if os.path.isdir(options['logs']['dir_logs']):
            if click.confirm('Logs directory already exists in {}. Erase?'
                .format(options['logs']['dir_logs'], default=False)):
                os.system('rm -r ' + options['logs']['dir_logs'])
            else:
                return
        os.system('mkdir -p ' + options['logs']['dir_logs'])
        path_new_opt = os.path.join(options['logs']['dir_logs'],
                       os.path.basename(args.path_opt))
        path_args = os.path.join(options['logs']['dir_logs'], 'args.yaml')
        with open(path_new_opt, 'w') as f:
            yaml.dump(options, f, default_flow_style=False)
        with open(path_args, 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)
        
    if exp_logger is None:
        # Set loggers
        exp_name = os.path.basename(options['logs']['dir_logs']) # add timestamp
        exp_logger = logger.Experiment(exp_name, options)
        exp_logger.add_meters('train', make_meters())
        exp_logger.add_meters('test', make_meters())
        if options['vqa']['trainsplit'] == 'train':
            exp_logger.add_meters('val', make_meters())
        exp_logger.info['model_params'] = utils.params_count(model)
        print('Model has {} parameters'.format(exp_logger.info['model_params']))

    #########################################################################################
    # args.evaluate: on valset OR/AND on testset
    #########################################################################################

    if args.evaluate:
        path_logger_json = os.path.join(options['logs']['dir_logs'], 'logger.json')

        if options['vqa']['trainsplit'] == 'train':
            acc1, val_results = engine.validate(val_loader, model, criterion,
                                                exp_logger, args.start_epoch, args.print_freq)
            # save results and compute OpenEnd accuracy
            exp_logger.to_json(path_logger_json)
            save_results(val_results, args.start_epoch, valset.split_name(),
                         options['logs']['dir_logs'], options['vqa']['dir'])
        
        test_results, testdev_results = engine.test(test_loader, model, exp_logger,
                                                    args.start_epoch, args.print_freq)
        # save results and DOES NOT compute OpenEnd accuracy
        exp_logger.to_json(path_logger_json)
        save_results(test_results, args.start_epoch, testset.split_name(),
                     options['logs']['dir_logs'], options['vqa']['dir'])
        save_results(testdev_results, args.start_epoch, testset.split_name(testdev=True),
                     options['logs']['dir_logs'], options['vqa']['dir'])
        return

    #########################################################################################
    # Begin training on train/val or trainval/test
    #########################################################################################

    for epoch in range(args.start_epoch+1, options['optim']['epochs']):
        #adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        engine.train(train_loader, model, criterion, optimizer, 
                     exp_logger, epoch, args.print_freq)
        
        if options['vqa']['trainsplit'] == 'train':
            # evaluate on validation set
            acc1, val_results = engine.validate(val_loader, model, criterion,
                                                exp_logger, epoch, args.print_freq)
            # remember best prec@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            save_checkpoint({
                    'epoch': epoch,
                    'arch': options['model']['arch'],
                    'best_acc1': best_acc1,
                    'exp_logger': exp_logger
                },
                model.module.state_dict(),
                optimizer.state_dict(),
                options['logs']['dir_logs'],
                args.save_model,
                args.save_all_from,
                is_best)

            # save results and compute OpenEnd accuracy
            save_results(val_results, epoch, valset.split_name(),
                         options['logs']['dir_logs'], options['vqa']['dir'])
        else:
            test_results, testdev_results = engine.test(test_loader, model, exp_logger,
                                                        epoch, args.print_freq)

            # save checkpoint at every timestep
            save_checkpoint({
                    'epoch': epoch,
                    'arch': options['model']['arch'],
                    'best_acc1': best_acc1,
                    'exp_logger': exp_logger
                },
                model.module.state_dict(),
                optimizer.state_dict(),
                options['logs']['dir_logs'],
                args.save_model,
                args.save_all_from)

            # save results and DOES NOT compute OpenEnd accuracy
            save_results(test_results, epoch, testset.split_name(),
                         options['logs']['dir_logs'], options['vqa']['dir'])
            save_results(testdev_results, epoch, testset.split_name(testdev=True),
                         options['logs']['dir_logs'], options['vqa']['dir'])
    

def make_meters():  
    meters_dict = {
        'loss': logger.AvgMeter(),
        'acc1': logger.AvgMeter(),
        'acc5': logger.AvgMeter(),
        'batch_time': logger.AvgMeter(),
        'data_time': logger.AvgMeter(),
        'epoch_time': logger.SumMeter()
    }
    return meters_dict

def save_results(results, epoch, split_name, dir_logs, dir_vqa):
    dir_epoch = os.path.join(dir_logs, 'epoch_' + str(epoch))
    name_json = 'OpenEnded_mscoco_{}_model_results.json'.format(split_name)
    # TODO: simplify formating
    if 'test' in split_name:
        name_json = 'vqa_' + name_json
    path_rslt = os.path.join(dir_epoch, name_json)
    os.system('mkdir -p ' + dir_epoch)
    with open(path_rslt, 'w') as handle:
        json.dump(results, handle)
    if not 'test' in split_name:
        os.system('python2 eval_res.py --dir_vqa {} --dir_epoch {} --subtype {} &'
                  .format(dir_vqa, dir_epoch, split_name))

def save_checkpoint(info, model, optim, dir_logs, save_model, save_all_from=None, is_best=True):
    os.system('mkdir -p ' + dir_logs)
    if save_all_from is None:
        path_ckpt_info  = os.path.join(dir_logs, 'ckpt_info.pth.tar')
        path_ckpt_model = os.path.join(dir_logs, 'ckpt_model.pth.tar')
        path_ckpt_optim = os.path.join(dir_logs, 'ckpt_optim.pth.tar')
        path_best_info  = os.path.join(dir_logs, 'best_info.pth.tar')
        path_best_model = os.path.join(dir_logs, 'best_model.pth.tar')
        path_best_optim = os.path.join(dir_logs, 'best_optim.pth.tar')
        # save info & logger
        path_logger = os.path.join(dir_logs, 'logger.json')
        info['exp_logger'].to_json(path_logger)
        torch.save(info, path_ckpt_info)
        if is_best:
            shutil.copyfile(path_ckpt_info, path_best_info)
        # save model state & optim state
        if save_model:
            torch.save(model, path_ckpt_model)
            torch.save(optim, path_ckpt_optim)
            if is_best:
                shutil.copyfile(path_ckpt_model, path_best_model)
                shutil.copyfile(path_ckpt_optim, path_best_optim)
    else:
        is_best = False # because we don't know the test accuracy
        path_ckpt_info  = os.path.join(dir_logs, 'ckpt_epoch,{}_info.pth.tar')
        path_ckpt_model = os.path.join(dir_logs, 'ckpt_epoch,{}_model.pth.tar')
        path_ckpt_optim = os.path.join(dir_logs, 'ckpt_epoch,{}_optim.pth.tar')
        # save info & logger
        path_logger = os.path.join(dir_logs, 'logger.json')
        info['exp_logger'].to_json(path_logger)
        torch.save(info, path_ckpt_info.format(info['epoch']))
        # save model state & optim state
        if save_model:
            torch.save(model, path_ckpt_model.format(info['epoch']))
            torch.save(optim, path_ckpt_optim.format(info['epoch']))
        if  info['epoch'] > 1 and info['epoch'] < save_all_from + 1:
            os.system('rm ' + path_ckpt_info.format(info['epoch'] - 1))
            os.system('rm ' + path_ckpt_model.format(info['epoch'] - 1))
            os.system('rm ' + path_ckpt_optim.format(info['epoch'] - 1))
    if not save_model:
        print('Warning train.py: checkpoint not saved')

def load_checkpoint(model, optimizer, path_ckpt):
    path_ckpt_info  = path_ckpt + '_info.pth.tar'
    path_ckpt_model = path_ckpt + '_model.pth.tar'
    path_ckpt_optim = path_ckpt + '_optim.pth.tar'
    if os.path.isfile(path_ckpt_info):
        info = torch.load(path_ckpt_info)
        start_epoch = 0
        best_acc1   = 0
        exp_logger  = None
        if 'epoch' in info:
            start_epoch = info['epoch']
        else:
            print('Warning train.py: no epoch to resume')
        if 'best_acc1' in info:
            best_acc1 = info['best_acc1']
        else:
            print('Warning train.py: no best_acc1 to resume')
        if 'exp_logger' in info:
            exp_logger = info['exp_logger']
        else:
            print('Warning train.py: no exp_logger to resume')
    else:
        print("Warning train.py: no info checkpoint found at '{}'".format(path_ckpt_info))
    if os.path.isfile(path_ckpt_model):
        model_state = torch.load(path_ckpt_model)
        model.load_state_dict(model_state)
    else:
        print("Warning train.py: no model checkpoint found at '{}'".format(path_ckpt_model))
    if optimizer is not None and os.path.isfile(path_ckpt_optim):
        optim_state = torch.load(path_ckpt_optim)
        optimizer.load_state_dict(optim_state)
    else:
        print("Warning train.py: no optim checkpoint found at '{}'".format(path_ckpt_optim))
    print("=> loaded checkpoint '{}' (epoch {}, best_acc1 {})"
              .format(path_ckpt, start_epoch, best_acc1))
    return start_epoch, best_acc1, exp_logger

if __name__ == '__main__':
    main()
