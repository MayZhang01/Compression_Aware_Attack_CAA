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
from llava.eval.utils import Logger, extract_name_from_path
import numpy as np
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
from llava.eval.model_vqa_loader_adv_i import get_chunk, create_data_loader, Counter, Counter_acc


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
    dataset_name=extract_name_from_path(args.question_file)
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

    random.seed(124)
    rlt_path='saved_data/llava-v1.6-vicuna-7b_llava_mme_filter2_adv_image_50_50_0.12549019607843137_310.pt'
    logger.print_log(rlt_path)
    dataset = torch.load(rlt_path)
    
    layer_list=eval(args.layer_list)
    cp_list=[1, 0.5, 0.2, 0.15, 0.1]#, 0.5, 0.3
    if 'textvqa' in args.question_file:
        evaluator = TextVQAAccuracyEvaluator()
        counter=Counter_acc(cp_list, evaluator=evaluator)
    else:
        counter=Counter(cp_list)

    adv_data_dict={}
    for idx in dataset:
        gt = dataset[idx]['gt']
        if isinstance(gt, tuple):
            gt=gt[0]
        input_ids = dataset[idx]['input_ids'].to(device='cuda', non_blocking=True)
        # attention_mask = dataset[idx]['attention_mask']
        # new_input_embeds = dataset[idx]['new_input_embeds']
        image_sizes = dataset[idx]['image_sizes']
        image_tensor = dataset[idx]['image_tensor']
        least_inter_ratio = dataset[idx]['least_inter_ratio']
        patch_inter_ratio = dataset[idx]['patch_inter_ratio']
        adv_num = dataset[idx]['adv_num']
        ori_prompt_len = dataset[idx]['ori_prompt_len']
        adv_input_embeds = dataset[idx]['adv_input_embeds']
        
        with torch.no_grad():
            (
                _,
                _,
                _,
                _,
                _,
                _,  
            ) = model.prepare_inputs_labels_for_multimodal_pdrop(       
                input_ids, position_ids=None, attention_mask=None, past_key_values=None, labels=None,
                images=torch.tensor([0]),#
                # images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True), 
                image_sizes=image_sizes,
                image_feature_size=(1,576,4096)
            )
            ori_prompt_len=model.model.prompt_len[0]

        rlt={'patch_inter_ratio':patch_inter_ratio, 'least_inter_ratio':least_inter_ratio}
        for cp in cp_list:
            with torch.inference_mode():
                if model_class_name == "LlavaLlamaForCausalLM_PDrop":
                    model.model.layer_list = layer_list
                    model.model.image_token_ratio_list = [cp]
                    model.model.image_token_ratio_list.insert(0, 1.0)
                model.model.prompt_len[0]=ori_prompt_len+adv_num
                adv_output_ids = model.generate(
                    inputs_embeds=adv_input_embeds,
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

        with torch.inference_mode():
            if model_class_name == "LlavaLlamaForCausalLM_PDrop":
                model.model.layer_list = []
                model.model.image_token_ratio_list = []
            model.model.prompt_len[0]=ori_prompt_len
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
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        rlt['ori']=outputs
        if 'textvqa' in args.question_file:
            rlt=counter.add_rlt(gt=gt, rlt=rlt)
        else:
            counter.add_rlt(gt=gt, rlt=rlt)
        rlt['gt']=gt
        logger.print_log(rlt)
        if counter.total%20==0 and counter.total>0:
            counter.get_metric(logger=logger)
        # adv_data_dict[idx]={
        #     'input_ids': input_ids, 
        #     'image_tensor': image_tensor, 
        #     'image_sizes': image_sizes, 
        #     'gt': gt,
        #     'attention_mask': attention_mask,
        #     'new_input_embeds': new_input_embeds,
        # }
    counter.get_metric(logger=logger)
    # torch.save(
    #     adv_data_dict, 
    #     'saved_data/'+'_'.join([
    #         model_name, 
    #         dataset_name,
    #         'adv_image',
    #         str(len(questions))
    #     ])+'.pt'
    # )

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
