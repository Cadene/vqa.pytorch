import argparse
import json
import random
import os
from os.path import join
import sys
#import pickle
helperDir = 'vqa/external/VQA/'
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(helperDir))
sys.path.insert(0, '%s/PythonEvaluationTools/vqaEvaluation' %(helperDir))
from vqa import VQA
from vqaEval import VQAEval


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_vqa',   type=str, default='/local/cadene/data/vqa')
    parser.add_argument('--dir_epoch', type=str, default='logs/16_12_13_20:39:55/epoch,1')
    parser.add_argument('--subtype',  type=str, default='train2014')
    args = parser.parse_args()

    diranno  = join(args.dir_vqa, 'raw', 'annotations')
    annFile  = join(diranno, 'mscoco_%s_annotations.json' % (args.subtype))
    quesFile = join(diranno, 'OpenEnded_mscoco_%s_questions.json' % (args.subtype))
    vqa = VQA(annFile, quesFile)
    
    taskType    = 'OpenEnded'
    dataType    = 'mscoco'
    dataSubType = args.subtype
    resultType  = 'model'
    fileTypes = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType'] 
    
    [resFile, accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = \
        ['%s/%s_%s_%s_%s_%s.json' % (args.dir_epoch, taskType, dataType,
            dataSubType, resultType, fileType) for fileType in fileTypes] 
    vqaRes = vqa.loadRes(resFile, quesFile)
    vqaEval = VQAEval(vqa, vqaRes, n=2)

    quesIds = [int(d['question_id']) for d in json.loads(open(resFile).read())]
    vqaEval.evaluate(quesIds=quesIds)
    
    json.dump(vqaEval.accuracy,     open(accuracyFile,     'w'))
    #json.dump(vqaEval.evalQA,       open(evalQAFile,       'w'))
    #json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, 'w'))
    #json.dump(vqaEval.evalAnsType,  open(evalAnsTypeFile,  'w'))