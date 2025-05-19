import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.model_vqa_loader import eval_model
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = "/mnt/codedisk/model_weight/llava-v1.5-7b"#llava-v1.5-7b


args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "image_folder": '/mnt/codedisk/python/playground/data/eval/pope/val2014',
    "question_file": '/mnt/codedisk/python/playground/data/eval/pope/llava_pope_test.jsonl',
    "answers_file": '/mnt/codedisk/python/playground/data/eval/pope/answers/llava-v1.5-13b.jsonl',
    "temperature": 0,
    "conv_mode": "llava_v1",
    # "temperature": 0,
    "num_chunks":1,
    "chunk_idx":0,
    # "top_p":None,
    "num_beams":1,
    "max_new_tokens":10,
    "layer_list":'[2]',
    "image_token_ratio_list":"[1]",
    # "layer_list":'[2,6,15]',
    # "image_token_ratio_list":"[0.1,0.08,0.06]",
})()


eval_model(args)