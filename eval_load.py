import os
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.model_vqa_loader_adv_i_load import eval_model
# import torch
from llava.eval.find_adv_util import get_args

ArgsDefault = type('Args', (), {
    "model_path": "../../model_weight/llava-v1.5-7b",
    "model_base": None,
    # "image_folder": '../playground/data/eval/MME/MME_Benchmark',
    # "question_file": '../playground/data/eval/MME/llava_mme_ground_truth.jsonl',
    # "question_file": "../playground/data/eval/MME/llava_mme_filter_rule1.jsonl",
    # "question_file": "../playground/data/eval/MME/qwen25_7b_llava_mme_filter_rule1.jsonl",
    "temperature": 0,
    "conv_mode": "llava_v1",
    "num_chunks":1,
    "chunk_idx":0,
    # "top_p":None,
    "num_beams":1,
    "max_new_tokens":3,    
    "image_token_ratio_list":"[1, 0.7, 0.5, 0.3, 0.2, 0.1]",
    "benign_flag": True,
    "most_or_least":True,
    "res_file": 'saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.2_0.4_32_150_1.0_8_False_True_1_ca_False_157_False_20_20_10_5_5_10_10_40_0.8.pt',

})

args = get_args(ArgsDefault)
layer_list=[
    [2], 
    # [4], [6], [8],
    # [1],[10],[12]
    # [8,16,24],
    # [2,6,15]
]

model_path = os.path.expanduser(args.model_path)
model_name = get_model_name_from_path(model_path)
pdrop_infer = True  # whether to use pdrop infer
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, 
    args.model_base, 
    model_name,
    pdrop_infer,
    load_4bit=True
)
for ll in layer_list:
    args.layer_list=ll
    eval_model(args, tokenizer, model, image_processor)