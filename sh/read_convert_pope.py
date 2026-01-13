from llava.eval.utils import load_jsonl, save_jsonl, load_json
import os
import json

def cut_str(input_str, input_char_list):
    for input_char in input_char_list:
        idx=input_str.find(input_char)
        if idx>-1:
            # tmp_str=input_str[:idx]
            return input_str[:idx].rstrip()
    return input_str

question_file='../playground/data/eval/pope/llava_pope_test.jsonl'
ques_list = load_jsonl(question_file)
candidate_answer_path='../playground/data/eval/pope/coco_ground_truth_segmentation.json'
candidate_answer=load_jsonl(candidate_answer_path)
all_answer={}
for ans in candidate_answer:
    image_id = ans['image']
    objects = ans['objects']
    all_answer[image_id]=objects

new_dataset=[]

for ques in ques_list:
    image_id = ques['image']
    qusetion = ques['text']
    candidate = all_answer[image_id]
    answer="No"
    for tmp_c in candidate:
        qusetion=qusetion.lower()
        if qusetion.find(tmp_c.lower())>-1:
            answer='Yes'
    ques['ground_truth']=answer
    if 'ground_truth' in ques:
        new_dataset.append(ques)

save_jsonl(new_dataset, '../playground/data/eval/pope/llava_pope_test_ground_truth.jsonl')

