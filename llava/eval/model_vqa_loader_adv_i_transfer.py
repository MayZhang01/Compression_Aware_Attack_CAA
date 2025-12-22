import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import random
from llava.eval.find_adv import opt_adv_pro, judge_adv, opt_adv_image_token, opt_adv_image_text_pro
from llava.eval.find_adv_patch import opt_adv_image_patch
from llava.eval.utils import Logger, extract_name_from_path, save_clip_tensor_as_image, print_args, process_input
import numpy as np
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
from llava.eval.model_vqa_loader_adv_i import get_chunk, create_data_loader, Counter, Counter_acc

def eval_model(args, tokenizer, model, image_processor):
    most_or_least=getattr(args, 'most_or_least', True)
    logger=Logger()
    logger.print_log('from transfer')
    print_args(args, logger)

    # Model
    disable_torch_init()
    if args.layer_list is not None:
        pdrop_infer = True  # whether to use pdrop infer
    
    model.model.logger=logger
    conv_mode = conv_templates[args.conv_mode].copy()

    model_class_name = type(model).__name__

    random.seed(124)
    rlt_path=args.rlt_path
    logger.print_log(rlt_path)
    dataset = torch.load(rlt_path)
    
    layer_list=eval(args.layer_list)
    cp_list=eval(args.image_token_ratio_list)
    if 'vqa' in rlt_path:
        evaluator = TextVQAAccuracyEvaluator()
        counter=Counter_acc(cp_list, evaluator=evaluator)
    else:
        counter=Counter(cp_list)

    for idx in dataset:
        gt = dataset[idx]['gt']
        if isinstance(gt, tuple):
            gt=gt[0]
        text = dataset[idx]['text']
        image_tensor = dataset[idx]['adv_image_tensor']
        
        input_ids, image_tensor, image_sizes=process_input(text, image_tensor, tokenizer, image_processor, model.config, conv_mode)
        save_clip_tensor_as_image(image_tensor.cpu(),'1.png')
        
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
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
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
        if counter.total%20==0 and counter.total>0:
            counter.get_metric(logger=logger)
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
