"""
Preprocess a train/test pair of interim json data files.
Caption: Use NLTK or split function to get tokens. 
"""
from random import shuffle, seed
import sys
import os.path
import argparse
import numpy as np
import scipy.io
import pdb
import json
import csv
import re
import math
import pickle
#import pprint

def get_top_answers(examples, nans=3000):
    counts = {}
    for ex in examples:
        ans = ex['answer'] 
        counts[ans] = counts.get(ans, 0) + 1

    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    print('Top answer and their counts:'    )
    print('\n'.join(map(str,cw[:20])))

    vocab = []
    for i in range(nans):
        vocab.append(cw[i][1])
    return vocab[:nans]

def remove_examples(examples, ans_to_aid):
    new_examples = []
    for i, ex in enumerate(examples):
        if ex['answer'] in ans_to_aid:
            new_examples.append(ex)
    print('Number of examples reduced from %d to %d '%(len(examples), len(new_examples)))
    return new_examples

def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];

def tokenize_mcb(s):
    t_str = s.lower()
    for i in [r'\?',r'\!',r'\'',r'\"',r'\$',r'\:',r'\@',r'\(',r'\)',r'\,',r'\.',r'\;']:
        t_str = re.sub( i, '', t_str)
    for i in [r'\-',r'\/']:
        t_str = re.sub( i, ' ', t_str)
    q_list = re.sub(r'\?','',t_str.lower()).split(' ')
    q_list = list(filter(lambda x: len(x) > 0, q_list))
    return q_list

def preprocess_questions(examples, nlp='nltk'):
    if nlp == 'nltk':
        from nltk.tokenize import word_tokenize
    print('Example of generated tokens after preprocessing some questions:')
    for i, ex in enumerate(examples):
        s = ex['question']
        if nlp == 'nltk':
            ex['question_words'] = word_tokenize(str(s).lower())
        elif nlp == 'mcb':
            ex['question_words'] = tokenize_mcb(s)
        else:
            ex['question_words'] = tokenize(s)
        if i < 10:
            print(ex['question_words'])
        if i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(examples), i*100.0/len(examples)) )
            sys.stdout.flush() 
    return examples

def remove_long_tail_train(examples, minwcount=0):
    # Replace words which are in the long tail (counted less than 'minwcount' times) by the UNK token.
    # Also create vocab, a list of the final words.

    # count up the number of words
    counts = {}
    for ex in examples:
        for w in ex['question_words']:
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count,w) for w, count in counts.items()], reverse=True)
    print('Top words and their counts:')
    print('\n'.join(map(str,cw[:20])))

    total_words = sum(counts.values())
    print('Total words:', total_words)
    bad_words = [w for w,n in counts.items() if n <= minwcount]
    vocab     = [w for w,n in counts.items() if n > minwcount]
    bad_count = sum(counts[w] for w in bad_words)
    print('Number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print('Number of words in vocab would be %d' % (len(vocab), ))
    print('Number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))

    print('Insert the special UNK token')
    vocab.append('UNK')
    for ex in examples:
        words = ex['question_words']
        question = [w if counts.get(w,0) > minwcount else 'UNK' for w in words]
        ex['question_words_UNK'] = question

    return examples, vocab

def remove_long_tail_test(examples, word_to_wid):
    for ex in examples:
        ex['question_words_UNK'] = [w if w in word_to_wid else 'UNK' for w in ex['question_words']]
    return examples

def encode_question(examples, word_to_wid, maxlength=15, pad='left'):
    # Add to tuple question_wids and question_length
    for i, ex in enumerate(examples):
        ex['question_length'] = min(maxlength, len(ex['question_words_UNK'])) # record the length of this sequence
        ex['question_wids'] = [0]*maxlength
        for k, w in enumerate(ex['question_words_UNK']):
            if k < maxlength:
                if pad == 'right':
                    ex['question_wids'][k] = word_to_wid[w]
                else:   #['pad'] == 'left'
                    new_k = k + maxlength - len(ex['question_words_UNK'])
                    ex['question_wids'][new_k] = word_to_wid[w]
                ex['seq_length'] = len(ex['question_words_UNK'])
    return examples

def encode_answer(examples, ans_to_aid):
    print('Warning: aid of answer not in vocab is 1999')
    for i, ex in enumerate(examples):
        ex['answer_aid'] = ans_to_aid.get(ex['answer'], 1999) # -1 means answer not in vocab
    return examples

def encode_answers_occurence(examples, ans_to_aid):
    for i, ex in enumerate(examples):
        answers = []
        answers_aid = []
        answers_count = []
        for ans in ex['answers_occurence']:
            aid = ans_to_aid.get(ans[0], -1) # -1 means answer not in vocab
            if aid != -1:
                answers.append(ans[0])
                answers_aid.append(aid) 
                answers_count.append(ans[1])
        ex['answers']       = answers
        ex['answers_aid']   = answers_aid
        ex['answers_count'] = answers_count
    return examples

def vqa_processed(params):
    
    #####################################################
    ## Read input files
    #####################################################

    path_train = os.path.join(params['dir'], 'interim', params['trainsplit']+'_questions_annotations.json')
    if params['trainsplit'] == 'train':
        path_val = os.path.join(params['dir'], 'interim', 'val_questions_annotations.json')
    path_test    = os.path.join(params['dir'], 'interim', 'test_questions.json')
    path_testdev = os.path.join(params['dir'], 'interim', 'testdev_questions.json')
 
    # An example is a tuple (question, image, answer)
    # /!\ test and test-dev have no answer
    trainset = json.load(open(path_train, 'r'))
    if params['trainsplit'] == 'train':
        valset = json.load(open(path_val, 'r'))
    testset    = json.load(open(path_test, 'r'))
    testdevset = json.load(open(path_testdev, 'r'))

    #####################################################
    ## Preprocess examples (questions and answers)
    #####################################################

    top_answers = get_top_answers(trainset, params['nans'])
    aid_to_ans = [a for i,a in enumerate(top_answers)]
    ans_to_aid = {a:i for i,a in enumerate(top_answers)}
    # Remove examples if answer is not in top answers
    trainset = remove_examples(trainset, ans_to_aid)

    # Add 'question_words' to the initial tuple
    trainset = preprocess_questions(trainset, params['nlp'])
    if params['trainsplit'] == 'train':
        valset = preprocess_questions(valset, params['nlp'])
    testset    = preprocess_questions(testset, params['nlp'])
    testdevset = preprocess_questions(testdevset, params['nlp'])

    # Also process top_words which contains a UNK char
    trainset, top_words = remove_long_tail_train(trainset, params['minwcount'])
    wid_to_word = {i+1:w for i,w in enumerate(top_words)}
    word_to_wid = {w:i+1 for i,w in enumerate(top_words)}

    if params['trainsplit'] == 'train':
        valset = remove_long_tail_test(valset, word_to_wid)
    testset    = remove_long_tail_test(testset, word_to_wid)
    testdevset = remove_long_tail_test(testdevset, word_to_wid)

    trainset = encode_question(trainset, word_to_wid, params['maxlength'], params['pad'])
    if params['trainsplit'] == 'train':
        valset = encode_question(valset, word_to_wid, params['maxlength'], params['pad'])
    testset    = encode_question(testset, word_to_wid, params['maxlength'], params['pad'])
    testdevset = encode_question(testdevset, word_to_wid, params['maxlength'], params['pad'])

    trainset = encode_answer(trainset, ans_to_aid)
    trainset = encode_answers_occurence(trainset, ans_to_aid)
    if params['trainsplit'] == 'train':
        valset = encode_answer(valset, ans_to_aid)
        valset = encode_answers_occurence(valset, ans_to_aid)

    #####################################################
    ## Write output files
    #####################################################

    # Paths to output files
    # Ex: data/vqa/processed/nans,3000_maxlength,15_..._trainsplit,train_testsplit,val/id_to_word.json
    subdirname = 'nans,'+str(params['nans'])
    for param in ['maxlength', 'minwcount', 'nlp', 'pad', 'trainsplit']:
        subdirname += '_' + param + ',' + str(params[param])
    os.system('mkdir -p ' + os.path.join(params['dir'], 'processed', subdirname))

    path_wid_to_word = os.path.join(params['dir'], 'processed', subdirname, 'wid_to_word.pickle')
    path_word_to_wid = os.path.join(params['dir'], 'processed', subdirname, 'word_to_wid.pickle')
    path_aid_to_ans  = os.path.join(params['dir'], 'processed', subdirname, 'aid_to_ans.pickle')
    path_ans_to_aid  = os.path.join(params['dir'], 'processed', subdirname, 'ans_to_aid.pickle')
    if params['trainsplit'] == 'train':
        path_trainset = os.path.join(params['dir'], 'processed', subdirname, 'trainset.pickle')
        path_valset   = os.path.join(params['dir'], 'processed', subdirname, 'valset.pickle')
    elif params['trainsplit'] == 'trainval':
        path_trainset = os.path.join(params['dir'], 'processed', subdirname, 'trainvalset.pickle')
    path_testset     = os.path.join(params['dir'], 'processed', subdirname, 'testset.pickle')
    path_testdevset  = os.path.join(params['dir'], 'processed', subdirname, 'testdevset.pickle')

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

    if params['trainsplit'] == 'train':
        print('Write valset to', path_valset)
        with open(path_valset, 'wb') as handle:
            pickle.dump(valset, handle)

    print('Write testset to', path_testset)
    with open(path_testset, 'wb') as handle:
        pickle.dump(testset, handle)

    print('Write testdevset to', path_testdevset)
    with open(path_testdevset, 'wb') as handle:
        pickle.dump(testdevset, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname',
        default='data/vqa',
        type=str,
        help='Root directory containing raw, interim and processed directories'
    )
    parser.add_argument('--trainsplit',
        default='train',
        type=str,
        help='Options: train | trainval'
    )
    parser.add_argument('--nans',
        default=2000,
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
    opt_vqa = vars(args)
    vqa_processed(opt_vqa)