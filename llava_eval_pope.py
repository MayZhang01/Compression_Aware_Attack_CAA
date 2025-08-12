import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.model_vqa_loader_adv_i import eval_model
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = "../../model_weight/llava-v1.5-7b"#llava-v1.5-7b
# model_path = "../../model_weight/llava-v1.6-vicuna-7b"#llava-v1.5-7b
# model_path = "../../model_weight/llava-v1.6-mistral-7b"#llava-v1.5-7b


# for adv_steps in [0, 30, 50]:
args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "image_folder": '../playground/data/eval/pope/val2014',
    "question_file": '../playground/data/eval/pope/llava_pope_test_ground_truth.jsonl',
    # "question_file": '../playground/data/eval/pope/llava_pope_filter_rule1.jsonl',
    # "question_file": '../playground/data/eval/pope/llava_pope_test_filter.jsonl',
    "answers_file": '../playground/data/eval/pope/answers/llava-v1.5-13b.jsonl',
    "temperature": 0,
    "conv_mode": "llava_v1",
    "num_chunks":1,
    "chunk_idx":0,
    # "top_p":None,
    "num_beams":1,
    "max_new_tokens":5,
    "layer_list":'[2]',
    "image_token_ratio_list":"[1]",
    "target_layer_idx":2, 
    "adv_steps":0, 
    "adv_lr":1,
    "epsilon":0/255,
    "n_patch_num":50,
    "l_patch_num":100,
    "bpr_ths":0.1,
    "image_size":336,
    "patch_size":14,
})()
eval_model(args)