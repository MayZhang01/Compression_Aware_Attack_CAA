import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.model_vqa_loader_adv_i import eval_model
from llava.eval.model_vqa_loader_adv_i_transfer import eval_model as eval_model_trf
from llava.eval.model_vqa_loader_adv_i_transfer_comb import eval_model as eval_model_trf_comb
# import torch
from llava.eval.find_adv_util import get_args

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model_path = "../../model_weight/llava-v1.5-7b"#llava-v1.5-7b
# model_path = "../../model_weight/llava-v1.6-vicuna-7b"#llava-v1.5-7b
# model_path = "../../model_weight/llava-v1.6-mistral-7b"#llava-v1.5-7b

ArgsDefault = type('Args', (), {
    "model_path": "../../model_weight/llava-v1.5-7b",
    "model_base": None,
    "image_folder": '../playground/data/eval/pope/val2014',
    # "question_file": '../playground/data/eval/pope/pope_filter_rule1_llava_qwen25_3b.jsonl',
    "question_file": '../playground/data/eval/pope/qwen25_3b_7b_llava_pope_filter_rule1.jsonl',
    "answers_file": '../playground/data/eval/pope/answers/llava-v1.5-13b.jsonl',
    "temperature": 0,
    "conv_mode": "llava_v1",
    "num_chunks":1,
    "chunk_idx":0,
    # "top_p":None,
    "num_beams":1,
    "max_new_tokens":3,
    "layer_list":'[2]',
    "target_layer_idx":20, 
    "image_token_ratio_list":"[1, 0.7, 0.5, 0.3, 0.2, 0.1]",#
    "adv_steps_list":"[10]",
    "adv_lr":1,
    "cut_num":1,
    "n_patch_num":0.2,
    "l_patch_num":0,
    "epsilon_list":"[64/255]",
    "bpr_ths":0.1,
    "image_size":336,
    "patch_size":14,
    "atk":'HJS',
    "patch_or_noise":False,
    "most_or_least":True,
    "re_scale":False,
    "most_flag":False,
    "w_bpr_l":20, "w_bpr_p":20, "w_dir_l":10, "w_dir_p":5, "w_bpr_lp":5,
    "w_ori_l":10, "w_ori_p":10,
    "stop_teps":40, "stop_ratio":0.8,
    "ori_bound": 7,
    "adv_path":'', "rlt_path":'', 
    # "adv_path":"['../qwen2vl_compress/saved_data/Qwen2.5-VL-7B-Instruct_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.15_0.15_128_100_1.0_2_False_True_1_trf_True_57_True_20.0_20.0_5.0_5.0_10.0_1.0_1.0_80_0.8_False.pt', "
    # "'../token-compress-pyramid/saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.15_0.15_128_100_1.0_2_False_True_3_trf_True_57_True_20.0_20.0_5.0_5.0_10.0_1.0_1.0_40_0.8.pt']",
    # "rlt_path": "../qwen2vl_compress/saved_data/Qwen2.5-VL-7B-Instruct_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.15_0.15_128_80_1.0_2_False_True_1_trf_True_15_True_20_20_5.0_5.0_10.0_10.0_20.0_80_0.8_False.pt"
})

args = get_args(ArgsDefault)
model_path = os.path.expanduser(args.model_path)
model_name = get_model_name_from_path(model_path)
if args.layer_list is not None:
    pdrop_infer = True  # whether to use pdrop infer
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, 
    args.model_base, 
    model_name,
    pdrop_infer,
    load_4bit=True
)
epsilon_list=eval(args.epsilon_list)
adv_steps_list=eval(args.adv_steps_list)
for adv_steps in adv_steps_list:
    for epsilon in epsilon_list:
        args.adv_steps=adv_steps
        args.epsilon=epsilon
        print(f"adv_steps: {adv_steps}, epsilon: {epsilon}")
        if len(args.rlt_path) > 0:
            eval_model_trf(args, tokenizer, model, image_processor)
        elif len(args.adv_path) > 0:
            eval_model_trf_comb(args, tokenizer, model, image_processor)
        else:
            eval_model(args, tokenizer, model, image_processor)