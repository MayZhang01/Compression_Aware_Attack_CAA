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
from llava.eval.find_adv_patch import opt_adv_image_patch
from llava.eval.find_adv_trf import opt_adv_image_patch_transf
from llava.eval.find_adv_trf_comb import opt_adv_image_patch_transf_comb
from llava.eval.utils import Logger, extract_name_from_path, print_args, get_arg_or_default
from llava.eval.find_adv_util import judge_adv
import numpy as np
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
from llava.eval.model_vqa_loader_util import split_list, get_chunk, CustomDataset, Counter_acc, Counter, create_data_loader, collate_fn

from llava.eval.find_adv_vnl import opt_adv_image_vanilla

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def get_target_label_from_gt(gt, tokenizer, target_label=[], none_char=[1,29871]):
    tids_list=tokenizer(gt)['input_ids']
    for tl in tids_list:
        for t in tl:
            if t not in target_label and t not in none_char:
                target_label.append(t)
    return target_label

def eval_model(args, tokenizer, model, image_processor):
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    dataset_name=extract_name_from_path(args.res_file)
    most_or_least=getattr(args, 'most_or_least', True)
    benign_flag=getattr(args, 'benign_flag', False)
    dir_name=dataset_name

    logger=Logger(dir_name=dir_name)
    print_args(args, logger)
    logger.print_log('model_name:', model_name)
    logger.print_log('dataset_name:', dataset_name)
    
    model.model.logger=logger

    model_class_name = type(model).__name__

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    
    layer_list=args.layer_list
    cp_list=eval(args.image_token_ratio_list)

    if 'vqa' in args.res_file:
        evaluator = TextVQAAccuracyEvaluator()
        counter=Counter_acc(cp_list, evaluator=evaluator)
    else:
        counter=Counter(cp_list)

    data=torch.load(args.res_file)
    
    for idx in tqdm(data):
        
        input_ids = data[idx]['input_ids'].to(model.device)
        raw_input_embeds = data[idx]['raw_input_embeds'].to(model.device)
        adv_input_embeds = data[idx]['adv_input_embeds'].to(model.device)
        # image_grid_thw = data[idx]['image_grid_thw'].to(model.device)
        # attention_mask = data[idx]['attention_mask'].to(model.device)
        gt = data[idx]['gt']
        cur_prompt = data[idx]['text']
        # image_sizes=data[idx]['adv_image_tensor'].to(model.device)
        ori_prompt_len=input_ids.shape[1]-1
        
        image_index = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1].tolist()
        
        model.model.image_token_posi=[image_index[0]]
        model.model.image_tokens=[576]
        model.model.prompt_len=[ori_prompt_len]
        model.model.most_or_least=most_or_least
        model.model.layer_list = layer_list
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        

        rlt={}
        
        if benign_flag:
            for cp in cp_list:
                with torch.inference_mode():
                    if model_class_name == "LlavaLlamaForCausalLM_PDrop":
                        model.model.layer_list = layer_list
                        model.model.image_token_ratio_list = [cp]
                        model.model.image_token_ratio_list.insert(0, 1.0)
                        model.model.most_or_least=most_or_least
                    model.model.prompt_len[0]=ori_prompt_len
                    raw_output_ids = model.generate(
                        inputs_embeds=raw_input_embeds,
                        do_sample=True if args.temperature > 0 else False,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True
                        # temperature=args.temperature,
                        # top_p=args.top_p,
                    )
                raw_outputs = tokenizer.batch_decode(raw_output_ids, skip_special_tokens=True)[0].strip()
                rlt['ben_' + str(cp)]=raw_outputs
        
        else:
            for cp in cp_list:
                with torch.inference_mode():
                    if model_class_name == "LlavaLlamaForCausalLM_PDrop":
                        model.model.image_token_ratio_list = [cp]
                        model.model.image_token_ratio_list.insert(0, 1.0)
                    adv_output_ids = model.generate(
                        inputs_embeds=adv_input_embeds,
                        do_sample=True if args.temperature > 0 else False,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True
                        # image_sizes=image_sizes,
                        # temperature=args.temperature,
                        # top_p=args.top_p,
                    )
                adv_outputs = tokenizer.batch_decode(adv_output_ids, skip_special_tokens=True)[0].strip()
                rlt[str(cp)]=adv_outputs

        if 'vqa' in args.res_file:
            rlt=counter.add_rlt(gt=gt, rlt=rlt)
        else:
            counter.add_rlt(gt=gt, rlt=rlt)
        rlt['gt']=gt
        
        logger.print_log(idx, ':', cur_prompt.replace('\n', ' '))
        logger.print_log(counter.total, ':', rlt)
        if counter.total%19==0 and counter.total>0:
            metric=counter.get_metric(logger=logger)
        
        del adv_input_embeds, input_ids, raw_input_embeds
        torch.cuda.empty_cache()
    
    counter.get_metric(logger=logger)
    
    print_args(args, logger)
    del logger

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
