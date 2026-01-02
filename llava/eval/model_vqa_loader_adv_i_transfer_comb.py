import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import random
from llava.eval.find_adv import opt_adv_pro, judge_adv, opt_adv_image_token, opt_adv_image_text_pro
from llava.eval.find_adv_patch import opt_adv_image_patch
from llava.eval.find_adv_trf_comb import opt_adv_image_patch_transf_comb
from llava.eval.utils import Logger, extract_name_from_path, save_clip_tensor_as_image, print_args, process_input
import numpy as np
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
from llava.eval.model_vqa_loader_adv_i import get_chunk, create_data_loader, Counter, Counter_acc

def eval_model(args, tokenizer, model, image_processor):
    most_or_least=getattr(args, 'most_or_least', True)
    dataset_name=extract_name_from_path(args.adv_path)
    dir_name=dataset_name
    logger=Logger(dir_name=dir_name)
    logger.print_log('from transfer')
    print_args(args, logger)

    # Model
    disable_torch_init()
    if args.layer_list is not None:
        pdrop_infer = True  # whether to use pdrop infer
    
    model.model.logger=logger
    conv_mode = conv_templates[args.conv_mode].copy()

    model_class_name = type(model).__name__

    # random.seed(124)
    
    adv_path_list=eval(args.adv_path)
    adv_patch_dict, adv_least_dict, raw_data_dict={}, {}, {}
    for adv_path in adv_path_list:
        logger.print_log(adv_path)
        tmp_adv_data=torch.load(adv_path)
        for idx in tmp_adv_data:
            tmp_adv_patch=tmp_adv_data[idx]['adv_patch']
            if idx in adv_patch_dict:
                adv_patch_dict[idx].append(tmp_adv_patch)
            else:
                adv_patch_dict[idx]=[tmp_adv_patch]
            
            tmp_adv_least=tmp_adv_data[idx]['adv_least']
            if idx in adv_least_dict:
                adv_least_dict[idx].append(tmp_adv_least)
            else:
                adv_least_dict[idx]=[tmp_adv_least]
            
            if idx not in raw_data_dict:
                raw_data_dict[idx]={
                    'raw_image_tensor':tmp_adv_data[idx]['raw_image_tensor'],
                    'gt':tmp_adv_data[idx]['gt'],
                    'text':tmp_adv_data[idx]['text'],
                }
    
    layer_list=eval(args.layer_list)
    cp_list=eval(args.image_token_ratio_list)
    if 'vqa' in adv_path_list[0]:
        evaluator = TextVQAAccuracyEvaluator()
        counter=Counter_acc(cp_list, evaluator=evaluator)
    else:
        counter=Counter(cp_list)

    count=0
    for idx in raw_data_dict:
        count+=1
        if count<19:
            continue
        gt = raw_data_dict[idx]['gt']
        if isinstance(gt, tuple):
            gt=gt[0]
        text = raw_data_dict[idx]['text']
        image_tensor = raw_data_dict[idx]['raw_image_tensor'].unsqueeze(0)

        
        input_ids, _, image_sizes=process_input(text, image_tensor, tokenizer, image_processor, model.config, conv_mode)
        save_clip_tensor_as_image(image_tensor.cpu(),'1.png')

        adv_image=opt_adv_image_patch_transf_comb(
            image_tensor,
            adv_least_dict[idx],
            adv_patch_dict[idx],
            epsilon=args.epsilon,
        )
        
        rlt={'patch_inter_ratio':0, 'least_inter_ratio':0}
        for cp in cp_list:
            with torch.inference_mode():
                if model_class_name == "LlavaLlamaForCausalLM_PDrop":
                    model.model.layer_list = layer_list
                    model.model.image_token_ratio_list = [cp]
                    model.model.image_token_ratio_list.insert(0, 1.0)
                    model.model.most_or_least=most_or_least
                adv_output_ids = model.generate(
                    input_ids,
                    images=adv_image.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    do_sample=True if args.temperature > 0 else False,
                    # image_sizes=image_sizes,
                    # temperature=args.temperature,
                    # top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True
                )
            adv_outputs = tokenizer.batch_decode(adv_output_ids, skip_special_tokens=True)[0].strip()
            rlt[str(cp)]=adv_outputs

        torch.cuda.empty_cache()
        if 'textvqa' in args.question_file:
            rlt=counter.add_rlt(gt=gt, rlt=rlt)
        else:
            counter.add_rlt(gt=gt, rlt=rlt)
        rlt['gt']=gt
        logger.print_log(idx, ':', text)
        logger.print_log(counter.total,':', rlt)
        if counter.total%19==0 and counter.total>0:
            counter.get_metric(logger=logger)
        # if counter.total==76:
        #     break
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
