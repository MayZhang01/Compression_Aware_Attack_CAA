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
from llava.eval.find_adv import opt_adv_pro, judge_adv, opt_adv_image_token, opt_adv_image_text_pro
from llava.eval.find_adv_patch import opt_adv_image_patch
from llava.eval.utils import Logger, extract_name_from_path, save_clip_tensor_as_image
import numpy as np
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
from llava.eval.model_vqa_loader_adv_i import get_chunk, create_data_loader, Counter, Counter_acc

def process_input(text, image_tensor, tokenizer, image_processor, model_config, conv_mode):
    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + text

    conv = conv_mode.copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if len(image_tensor.shape)==3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = process_images(image_tensor, image_processor, model_config)
    image_sizes=image_tensor.size
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).to(device='cuda')
    return input_ids, image_tensor, image_sizes

def eval_model(args, tokenizer, model, image_processor):
    logger=Logger()
    logger.print_log('from transfer')
    for key, value in args.__dict__.items():
        if not key.startswith('__'):
            if key=='epsilon':
                logger.print_log('epsilon:', int(args.epsilon*255))
            else:
                logger.print_log(f"{key}: {value}")

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
    cp_list=[1, 0.5, 0.2, 0.15, 0.1]#, 0.5, 0.3
    if 'vqa' in rlt_path:
        evaluator = TextVQAAccuracyEvaluator()
        counter=Counter_acc(cp_list, evaluator=evaluator)
    else:
        counter=Counter(cp_list)

    # adv_data_dict={}
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
                adv_output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    # temperature=args.temperature,
                    # top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True
                )
            adv_outputs = tokenizer.batch_decode(adv_output_ids, skip_special_tokens=True)[0].strip()
            rlt[str(cp)]=adv_outputs

        # with torch.inference_mode():
        #     if model_class_name == "LlavaLlamaForCausalLM_PDrop":
        #         model.model.layer_list = []
        #         model.model.image_token_ratio_list = []
        #     output_ids = model.generate(
        #         input_ids,
        #         images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
        #         image_sizes=image_sizes,
        #         do_sample=True if args.temperature > 0 else False,
        #         # temperature=args.temperature,
        #         # top_p=args.top_p,
        #         num_beams=args.num_beams,
        #         max_new_tokens=args.max_new_tokens,
        #         use_cache=True
        #     )
        # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # rlt['ori']=outputs

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
    for key, value in args.__dict__.items():
        if not key.startswith('__'):
            if key=='epsilon':
                logger.print_log('epsilon:', int(args.epsilon*255))
            else:
                logger.print_log(f"{key}: {value}")
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
