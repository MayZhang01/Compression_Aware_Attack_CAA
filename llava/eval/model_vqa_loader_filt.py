import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import random
# from llava.eval.find_adv import opt_adv_pro, opt_adv_image_token, opt_adv_image_text_pro
# from llava.eval.find_adv_patch import opt_adv_image_patch
from llava.eval.utils import Logger, save_jsonl
from llava.eval.find_adv_util import judge_adv
import numpy as np
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
# from llava.eval.utils import load_jsonl, load_json
from llava.eval.model_vqa_loader_util import split_list, get_chunk, CustomDataset, Counter_acc, Counter, create_data_loader, collate_fn

def eval_model(args):
    logger=Logger()
    # logger.print_log(args)
    for key, value in args.__class__.__dict__.items():
        if not key.startswith('__'):
            logger.print_log(f"{key}: {value}")

    # Model
    disable_torch_init()
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
    model.model.logger=logger

    model_class_name = type(model).__name__

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    random.seed(124)
    random.shuffle(questions)
    questions = questions[0:5000]
    # questions = questions[150:200]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # answers_file = os.path.expanduser(args.answers_file)
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, conv_mode=args.conv_mode)
    count, count2=0, 0
    total=0
    succ_list=[]
    layer_list=eval(args.layer_list)
    
    new_dataset=[]
    # counter=Counter_acc(evaluator=)
    evaluator = TextVQAAccuracyEvaluator()
    count_new=0
    for (input_ids, image_tensor, image_sizes, gt), line in tqdm(zip(data_loader, questions), total=len(questions)):
        if isinstance(gt, tuple):
            gt=gt[0]
        # if gd=='No':
        #     continue
        idx = line["question_id"]
        cur_prompt = line["text"]
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        
        # ori_prompt_len=model.model.prompt_len[0]

        with torch.inference_mode():
            if model_class_name == "LlavaLlamaForCausalLM_PDrop":
                model.model.layer_list = []
                model.model.image_token_ratio_list = []
            # model.model.prompt_len[0]=ori_prompt_len
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                # inputs_embeds=new_input_embeds,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                # temperature=args.temperature,
                # top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )
        outputs_i = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        with torch.inference_mode():
            if model_class_name == "LlavaLlamaForCausalLM_PDrop":
                model.model.layer_list = []
                model.model.image_token_ratio_list = []
            # model.model.prompt_len[0]=ori_prompt_len
            output_ids = model.generate(
                input_ids,
                images=torch.randn_like(image_tensor, dtype=torch.float16, device='cuda'),
                # inputs_embeds=new_input_embeds,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                # temperature=args.temperature,
                # top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )
        outputs_e = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        
        if 'textvqa' in args.question_file:
            if evaluator.eval_pred_per_sample(outputs_e, gt)==0 and evaluator.eval_pred_per_sample(outputs_i, gt)>0:
                new_dataset.append(line)
                count_new+=1
                logger.print_log(idx, ':', cur_prompt.replace('\n', ' '))
                logger.print_log(outputs_e, outputs_i, gt)
                logger.print_log('count_new', count_new)
        else:
            if  judge_adv(outputs_e, outputs_i)==False:# and judge_adv(outputs_i, gt)==True:
                new_dataset.append(line)
                count_new+=1
                logger.print_log(idx, ':', cur_prompt.replace('\n', ' '))
                logger.print_log(outputs_e, outputs_i, gt)
                logger.print_log('count_new', count_new)
    save_jsonl(new_dataset, '../playground/data/eval/textvqa/llava_vqa_filter_rule1.jsonl')
    # save_jsonl(new_dataset, '../playground/data/eval/pope/llava_pope_filter_rule1.jsonl')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--layer_list", type=str, default= None)
    parser.add_argument("--image_token_ratio_list", type=str, default= None)
    parser.add_argument("--pdrop_infer", action='store_true', help="use this to apply pdrop inference to ori llava-1.5")
    args = parser.parse_args()

    eval_model(args)
