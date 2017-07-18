import json
import os
import argparse

# def get_image_path(subtype='train2014', image_id='1', format='%s/COCO_%s_%012d.jpg'):
#     return format%(subtype, subtype, image_id)

def interim(questions_annotations):
    data = []
    for i in range(len(questions_annotations)):
        qa_img = questions_annotations[i]
        qa_img_id = qa_img['id']
        for j in range(len(qa_img['qas'])):
            qa = qa_img['qas'][j]
            row = {}
            row['question_id'] = qa['qa_id']
            row['image_id'] = qa_img_id
            row['image_name'] = str(qa_img_id) + '.jpg'
            row['question'] = qa['question']
            row['answer'] = qa['answer']
            data.append(row)
    return data

def vgenome_interim(params):
    '''
    Put the VisualGenomme VQA data into single json file in data/interim
    or train, val, trainval : [[question_id, image_id, question, answer] ... ]
    '''
    path_qa = os.path.join(params['dir'], 'interim', 'questions_annotations.json')
    os.system('mkdir -p ' + os.path.join(params['dir'], 'interim'))

    print('Loading annotations and questions...')
    questions_annotations = json.load(open(os.path.join(params['dir'], 'raw', 'question_answers.json'), 'r'))
    
    data = interim(questions_annotations)
    print('Questions number %d'%len(data))
    print('Write', path_qa)
    json.dump(data, open(path_qa, 'w'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_vg', default='data/visualgenome', type=str, help='Path to visual genome data directory')
    args = parser.parse_args()
    params = vars(args)
    vgenome_interim(params)
