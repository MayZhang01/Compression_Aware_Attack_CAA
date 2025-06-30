from llava.eval.utils import load_jsonl, save_jsonl
import os
import json

def cut_str(input_str, input_char_list):
    for input_char in input_char_list:
        idx=input_str.find(input_char)
        if idx>-1:
            # tmp_str=input_str[:idx]
            return input_str[:idx].rstrip()
    return input_str

question_file='../playground/data/eval/MME/llava_mme.jsonl'
ques_list = load_jsonl(question_file)
mme_path='../playground/data/eval/MME/MME_Benchmark'

new_dataset=[]

for ques in ques_list:
    category = ques['question_id'].split('/')[0]
    file = ques['question_id'].split('/')[-1].split('.')[0] + '.txt'
    question = ques['text']
    
    category_dir = os.path.join(mme_path, category)
    if not os.path.isdir(category_dir):
        continue
    qa_path = os.path.join(category_dir, 'questions_answers_YN')
    if not os.path.exists(qa_path):
        qa_path=category_dir
    ans_path=os.path.join(qa_path, file)
    if not os.path.exists(ans_path):
        print(ans_path)
        continue
    for line in open(ans_path):
        ground_ques, answer = line.strip().split('\t')
        question = cut_str(question, ['?', '\nA', ' Please'])
        ground_ques = cut_str(ground_ques, ['?', '\nA', ' Please'])
        if question==ground_ques:
            ques['ground_truth']=answer
    if 'ground_truth' in ques:
        new_dataset.append(ques)
    else:
        print('ERROR', ques['question_id'])
save_jsonl(new_dataset, '../playground/data/eval/MME/llava_mme_ground_truth.jsonl')

