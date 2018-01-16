import os
import time
import yaml
import json
import argparse
import re
import base64
import torch
from torch.autograd import Variable
from PIL import Image
from io import BytesIO
from pprint import pprint

from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple

import torchvision.transforms as transforms
import vqa.lib.utils as utils
import vqa.datasets as datasets
import vqa.models as models
import vqa.models.convnets as convnets
from vqa.datasets.vqa_processed import tokenize_mcb
from train import load_checkpoint

parser = argparse.ArgumentParser(
    description='Demo server',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir_logs', type=str,
    #default='logs/vqa2/blocmutan_noatt_fbresnet152torchported_save_all',
    default='logs/vqa2/mutan_att_train',
    help='dir logs')
parser.add_argument('--path_opt', type=str,
    #default='logs/vqa2/blocmutan_noatt_fbresnet152torchported_save_all/blocmutan_noatt.yaml',
    default='logs/vqa2/mutan_att_train/mutan_att_train.yaml',
    help='path to a yaml options file')
parser.add_argument('--resume', type=str,
    default='best',
    help='path to latest checkpoint')
parser.add_argument('--cuda', type=bool,
    const=True,
    nargs='?',
    help='path to latest checkpoint')

@Request.application
def application(request):
    print('')
    if 'visual' in request.form and 'question' in request.form:
        visual = process_visual(request.form['visual'])
        question = process_question(request.form['question'])
        answer = process_answer(model(visual, question))
        response = Response(answer)
    
    elif 'question' not in request.form:
        response = Response('Question missing')

    elif 'visual' not in request.form:
        response = Response('Image missing')

    else:
        response = Response('what?')

    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.add('X-XSS-Protection', '0')    
    return response

def process_visual(visual_strb64):
    visual_strb64 = re.sub('^data:image/.+;base64,', '', visual_strb64)
    visual_PIL = Image.open(BytesIO(base64.b64decode(visual_strb64)))
    visual_tensor = transform(visual_PIL)
    visual_data = torch.FloatTensor(1, 3,
                                       visual_tensor.size(1),
                                       visual_tensor.size(2))
    visual_data[0][0] = visual_tensor[0]
    visual_data[0][1] = visual_tensor[1]
    visual_data[0][2] = visual_tensor[2]
    print('visual', visual_data.size(), visual_data.mean())
    if args.cuda:
        visual_data = visual_data.cuda(async=True)
    visual_input = Variable(visual_data, volatile=True)
    visual_features = cnn(visual_input)
    if 'NoAtt' in options['model']['arch']:
        nb_regions = visual_features.size(2) * visual_features.size(3)
        visual_features = visual_features.sum(3).sum(2).div(nb_regions).view(-1, 2048)
    return visual_features

def process_question(question_str):
    question_tokens = tokenize_mcb(question_str)
    question_data = torch.LongTensor(1, len(question_tokens))
    for i, word in enumerate(question_tokens):
        if word in trainset.word_to_wid:
            question_data[0][i] = trainset.word_to_wid[word]
        else:
            question_data[0][i] = trainset.word_to_wid['UNK']
    if args.cuda:
        question_data = question_data.cuda(async=True)
    question_input = Variable(question_data, volatile=True)
    print('question', question_str, question_tokens, question_data)

    return question_input

def process_answer(answer_var):
    answer_sm = torch.nn.functional.softmax(answer_var.data[0].cpu())
    max_, aid = answer_sm.topk(5, 0, True, True)
    ans = []
    val = []
    for i in range(5):
        ans.append(trainset.aid_to_ans[aid.data[i]])
        val.append(max_.data[i])

    att = []
    for x_att in model.list_att:
        img = x_att.view(1,14,14).cpu()
        img = transforms.ToPILImage()(img)
        buffer_ = BytesIO()
        img.save(buffer_, format="PNG")
        img_str = base64.b64encode(buffer_.getvalue()).decode()
        img_str = 'data:image/png;base64,'+img_str
        att.append(img_str)

    answer = {'ans':ans,'val':val,'att':att}
    answer_str = json.dumps(answer)

    return answer_str

def main():
    global args, options, model, cnn, transform, trainset
    args = parser.parse_args()

    options = {
        'logs': {
            'dir_logs': args.dir_logs
        }
    }
    if args.path_opt is not None:
        with open(args.path_opt, 'r') as handle:
            options_yaml = yaml.load(handle)
        options = utils.update_values(options, options_yaml)
    print('## args'); pprint(vars(args))
    print('## options'); pprint(options)

    trainset = datasets.factory_VQA(options['vqa']['trainsplit'],
                                    options['vqa'])
                                    #options['coco'])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Scale(options['coco']['size']),
        transforms.CenterCrop(options['coco']['size']),
        transforms.ToTensor(),
        normalize,
    ])

    opt_factory_cnn = {
        'arch': options['coco']['arch']
    }
    cnn = convnets.factory(opt_factory_cnn, cuda=args.cuda, data_parallel=False)
    model = models.factory(options['model'],
                           trainset.vocab_words(),
                           trainset.vocab_answers(),
                           cuda=args.cuda,
                           data_parallel=False)
    model.eval()
    start_epoch, best_acc1, _ = load_checkpoint(model, None,
            os.path.join(options['logs']['dir_logs'], args.resume))

    my_local_ip = '192.168.0.32'
    my_local_port = 3456
    run_simple(my_local_ip, my_local_port, application)

if __name__ == '__main__':
    main()

