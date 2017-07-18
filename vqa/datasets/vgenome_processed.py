"""
Preprocess an interim json data files
into one preprocess hdf5/json data files.
Caption: Use nltk, or mcb, or split function to get tokens. 
"""
from random import shuffle, seed
import sys
import os.path
import argparse
import numpy as np
import scipy.io
import pdb
import h5py
from nltk.tokenize import word_tokenize
import json
import csv
import re
import math
import pickle

from .vqa_processed import get_top_answers, remove_examples, tokenize, tokenize_mcb, \
                           preprocess_questions, remove_long_tail_train, \
                           encode_question, encode_answer

def preprocess_answers(examples, nlp='nltk'):
    print('Example of modified answers after preprocessing:')
    for i, ex in enumerate(examples):
        s = ex['answer']
        if nlp == 'nltk':
            ex['answer'] = " ".join(word_tokenize(str(s).lower()))
        elif nlp == 'mcb':
            ex['answer'] = " ".join(tokenize_mcb(s))
        else:
            ex['answer'] = " ".join(tokenize(s))
        if i < 10: print(s, 'became', "->"+ex['answer']+"<-")
        if i>0 and i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(examples), i*100.0/len(examples)) )
            sys.stdout.flush() 
    return examples

def build_csv(path, examples, split='train', delimiter_col='~', delimiter_number='|'):
    with open(path, 'wb') as f:
        writer = csv.writer(f, delimiter=delimiter_col)
        for ex in examples:
            import ipdb; ipdb.set_trace()
            row = []
            row.append(ex['question_id'])
            row.append(ex['question'])
            row.append(delimiter_number.join(ex['question_words_UNK']))
            row.append(delimiter_number.join(ex['question_wids']))

            row.append(ex['image_id'])

            if split in ['train','val','trainval']:
                row.append(ex['answer_aid'])
                row.append(ex['answer'])
            writer.writerow(row)

def vgenome_processed(params):
    
    #####################################################
    ## Read input files
    #####################################################

    path_train = os.path.join(params['dir'], 'interim', 'questions_annotations.json')
 
    # An example is a tuple (question, image, answer)
    # /!\ test and test-dev have no answer
    trainset = json.load(open(path_train, 'r'))

    #####################################################
    ## Preprocess examples (questions and answers)
    #####################################################

    trainset = preprocess_answers(trainset, params['nlp'])

    top_answers = get_top_answers(trainset, params['nans'])
    aid_to_ans = {i+1:w for i,w in enumerate(top_answers)}
    ans_to_aid = {w:i+1 for i,w in enumerate(top_answers)}

    # Remove examples if answer is not in top answers
    #trainset = remove_examples(trainset, ans_to_aid)

    # Add 'question_words' to the initial tuple
    trainset = preprocess_questions(trainset, params['nlp'])

    # Also process top_words which contains a UNK char
    trainset, top_words = remove_long_tail_train(trainset, params['minwcount'])
    wid_to_word = {i+1:w for i,w in enumerate(top_words)}
    word_to_wid = {w:i+1 for i,w in enumerate(top_words)}

    #examples_test = remove_long_tail_test(examples_test, word_to_wid)

    trainset = encode_question(trainset, word_to_wid, params['maxlength'], params['pad'])

    trainset = encode_answer(trainset, ans_to_aid)

    #####################################################
    ## Write output files
    #####################################################

    # Paths to output files
    # Ex: data/vqa/preprocess/nans,3000_maxlength,15_..._trainsplit,train_testsplit,val/id_to_word.json
    subdirname = 'nans,'+str(params['nans'])
    for param in ['maxlength', 'minwcount', 'nlp', 'pad', 'trainsplit']:
        subdirname += '_' + param + ',' + str(params[param])
    os.system('mkdir -p ' + os.path.join(params['dir'], 'processed', subdirname))

    path_wid_to_word = os.path.join(params['dir'], 'processed', subdirname, 'wid_to_word.pickle')
    path_word_to_wid = os.path.join(params['dir'], 'processed', subdirname, 'word_to_wid.pickle')
    path_aid_to_ans  = os.path.join(params['dir'], 'processed', subdirname, 'aid_to_ans.pickle')
    path_ans_to_aid  = os.path.join(params['dir'], 'processed', subdirname, 'ans_to_aid.pickle')
    #path_csv_train   = os.path.join(params['dir'], 'processed', subdirname, 'train.csv')
    path_trainset = os.path.join(params['dir'], 'processed', subdirname, 'trainset.pickle')

    print('Write wid_to_word to', path_wid_to_word)
    with open(path_wid_to_word, 'wb') as handle:
        pickle.dump(wid_to_word, handle)

    print('Write word_to_wid to', path_word_to_wid)
    with open(path_word_to_wid, 'wb') as handle:
        pickle.dump(word_to_wid, handle)

    print('Write aid_to_ans to', path_aid_to_ans)
    with open(path_aid_to_ans, 'wb') as handle:
        pickle.dump(aid_to_ans, handle)

    print('Write ans_to_aid to', path_ans_to_aid)
    with open(path_ans_to_aid, 'wb') as handle:
        pickle.dump(ans_to_aid, handle)

    print('Write trainset to', path_trainset)
    with open(path_trainset, 'wb') as handle:
        pickle.dump(trainset, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_vg',
        default='data/visualgenome',
        type=str,
        help='Root directory containing raw, interim and processed directories'
    )
    parser.add_argument('--nans',
        default=10000,
        type=int,
        help='Number of top answers for the final classifications'
    )
    parser.add_argument('--maxlength',
        default=26,
        type=int,
        help='Max number of words in a caption. Captions longer get clipped'
    )
    parser.add_argument('--minwcount',
        default=0,
        type=int,
        help='Words that occur less than that are removed from vocab'
    )
    parser.add_argument('--nlp',
        default='mcb',
        type=str,
        help='Token method ; Options: nltk | mcb | naive'
    )
    parser.add_argument('--pad',
        default='left',
        type=str,
        help='Padding ; Options: right (finish by zeros) | left (begin by zeros)'
    )
    args = parser.parse_args()
    params = vars(args)
    vgenome_processed(params)