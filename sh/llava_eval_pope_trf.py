import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
# from llava.eval.model_vqa_loader_adv_i import eval_model
from llava.eval.model_vqa_loader_adv_i_transfer import eval_model
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
    "question_file": '../playground/data/eval/pope/pope_filter_rule1_llava_qwen25_3b.jsonl',
    "answers_file": '../playground/data/eval/pope/answers/llava-v1.5-13b.jsonl',
    "temperature": 0,
    "conv_mode": "llava_v1",
    "num_chunks":1,
    "chunk_idx":0,
    # "top_p":None,
    "num_beams":1,
    "max_new_tokens":5,
    "layer_list":'[2]',
    "target_layer_idx":2, 
    "image_token_ratio_list":"[1, 0.5, 0.2, 0.15, 0.1]",
    "adv_steps_list":"[100]",
    "adv_lr":1,
    "n_patch_num":0.1,
    "l_patch_num":0.3,
    "epsilon_list":"[64/255]",
    "bpr_ths":0.1,
    "image_size":336,
    "patch_size":14,
    "atk":'ca',
    "patch_or_noise":False,
    "most_or_least":True,
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
    pdrop_infer
)
epsilon_list=eval(args.epsilon_list)
adv_steps_list=eval(args.adv_steps_list)
for adv_steps in adv_steps_list:
    for epsilon in epsilon_list:
        args.adv_steps=adv_steps
        args.epsilon=epsilon
        print(f"adv_steps: {adv_steps}, epsilon: {epsilon}")
        eval_model(args, tokenizer, model, image_processor)