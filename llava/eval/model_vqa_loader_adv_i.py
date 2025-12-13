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
from llava.eval.find_adv import opt_adv_pro, opt_adv_image_token, opt_adv_image_text_pro
from llava.eval.find_adv_patch import opt_adv_image_patch
from llava.eval.utils import Logger, extract_name_from_path, print_args
from llava.eval.find_adv_util import judge_adv
import numpy as np
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
from llava.eval.model_vqa_loader_util import split_list, get_chunk, CustomDataset, Counter_acc, Counter, create_data_loader, collate_fn

from llava.eval.find_adv_vnl import opt_adv_image_vanilla

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
    dataset_name=extract_name_from_path(args.question_file)
    max_new_tokens = args.max_new_tokens
    most_or_least=getattr(args, 'most_or_least', True)
    patch_or_noise=getattr(args, 'patch_or_noise', True)
    dir_name=dataset_name

    logger=Logger(dir_name=dir_name)
    print_args(args, logger)
    logger.print_log('model_name:', model_name)
    logger.print_log('dataset_name:', dataset_name)
    
    model.model.logger=logger

    model_class_name = type(model).__name__

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    random.seed(160)
    random.shuffle(questions)
    q_start=0
    questions = questions[q_start:157]
    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, conv_mode=args.conv_mode)
    
    try:
        w_bpr_l, w_bpr_p, w_dir_l, w_dir_p, w_bpr_lp = args.w_bpr_l, args.w_bpr_p, args.w_dir_l, args.w_dir_p, args.w_bpr_lp
        w_ori_l, w_ori_p = args.w_ori_l, args.w_ori_p
        most_flag=args.most_flag
        stop_teps=args.stop_teps
        stop_ratio=args.stop_ratio
    except:
        w_bpr_l, w_bpr_p, w_dir_l, w_dir_p, w_bpr_lp = 20, 20, 10, 10, 10
        w_ori_l, w_ori_p = -2, -2
        most_flag=False if args.atk=='ca' else True
        stop_teps, stop_ratio=80, 0.8

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    
    layer_list=eval(args.layer_list)
    cp_list=eval(args.image_token_ratio_list)
    if layer_list==[2,6,15]:
        sparse_flag=True
    else:
        sparse_flag=False


    if 'textvqa' in args.question_file:
        evaluator = TextVQAAccuracyEvaluator()
        counter=Counter_acc(cp_list, evaluator=evaluator)
    else:
        counter=Counter(cp_list)

    adv_data_dict={}
    for (input_ids, image_tensor, image_sizes, gt), line in tqdm(zip(data_loader, questions), total=len(questions)):
        if isinstance(gt, tuple):
            gt=gt[0]
        
        idx = line["question_id"]
        cur_prompt = line["text"]
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.no_grad():
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                new_input_embeds,
                _,  
            ) = model.prepare_inputs_labels_for_multimodal_pdrop(       
                input_ids, position_ids=None, attention_mask=None, past_key_values=None, labels=None,
                # images=torch.tensor([0]),#
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True), 
                image_sizes=image_sizes,
                # image_feature_size=(1,576,4096)
            )
            ori_prompt_len=model.model.prompt_len[0]
        if args.atk=="ca":
            adv_images, adv_input_embeds, adv_num, patch_inter_ratio, least_inter_ratio=opt_adv_image_patch(
                model, new_input_embeds, attention_mask,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True), 
                image_shape=576,        
                # token_length_list=[new_input_embeds.shape[1]],  
                pre_prompt_length_list=[torch.where(input_ids==-200)[1][0].item()], 
                target_layer_idx=args.target_layer_idx, 
                adv_steps=args.adv_steps, 
                adv_lr=args.adv_lr,
                epsilon=args.epsilon,
                n_patch_num=args.n_patch_num,
                l_patch_num=args.l_patch_num,
                bpr_ths=args.bpr_ths,
                image_size=args.image_size,
                patch_size=args.patch_size,
                logger=logger,
                patch_or_noise=patch_or_noise,
                most_flag=most_flag,
                w_bpr_l=w_bpr_l, w_bpr_p=w_bpr_p, w_dir_l=w_dir_l, w_dir_p=w_dir_p, w_bpr_lp=w_bpr_lp,
                w_ori_l=w_ori_l, w_ori_p=w_ori_p,
                stop_teps=stop_teps, stop_ratio=stop_ratio,
                sparse_flag=sparse_flag
            )
        elif args.atk=="vnl":
            target_label=get_target_label_from_gt(['Sorry', ' Sorry', 'sorry', ' sorry'], tokenizer, target_label=[])
            untarget_label=None
            if isinstance(gt, list):
                untarget_label=get_target_label_from_gt(gt, tokenizer, target_label=[11125])
            elif 'yes' in gt.lower():
                target_label=get_target_label_from_gt(['No', ' No', 'no', ' no'], tokenizer, target_label=[])
                untarget_label=get_target_label_from_gt(['Yes', ' Yes', 'yes', ' yes'], tokenizer, target_label=[])
            elif 'no' in gt.lower():
                target_label=get_target_label_from_gt(['Yes', ' Yes', 'yes', ' yes'], tokenizer, target_label=[])
                untarget_label=get_target_label_from_gt(['No', ' No', 'no', ' no'], tokenizer, target_label=[])
            
            adv_images, adv_input_embeds, adv_num, patch_inter_ratio, least_inter_ratio=opt_adv_image_vanilla(
                model, new_input_embeds, attention_mask,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True), 
                image_shape=576,        
                # token_length_list=[new_input_embeds.shape[1]],  
                pre_prompt_length_list=[torch.where(input_ids==-200)[1][0].item()], 
                target_layer_idx=args.target_layer_idx, 
                adv_steps=args.adv_steps, 
                adv_lr=args.adv_lr,
                epsilon=args.epsilon,
                n_patch_num=args.n_patch_num,
                l_patch_num=args.l_patch_num,
                bpr_ths=args.bpr_ths,
                image_size=args.image_size,
                patch_size=args.patch_size,
                logger=logger,
                patch_or_noise=patch_or_noise,
                target_label=target_label,
                untarget_label=untarget_label,
                mask_layer=args.mask_layer,
                sparse_flag=sparse_flag
            )

        rlt={'patch_inter_ratio':patch_inter_ratio, 'least_inter_ratio':least_inter_ratio}
        try:
            for cp in cp_list:
                with torch.inference_mode():
                    if model_class_name == "LlavaLlamaForCausalLM_PDrop":
                        model.model.layer_list = layer_list
                        model.model.image_token_ratio_list = [cp]
                        model.model.image_token_ratio_list.insert(0, 1.0)
                        model.model.most_or_least=most_or_least
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
        except Exception as e:
            logger.print_log(f"Error during generation for idx {idx}: {e}")
            torch.cuda.empty_cache()
            continue

        if 'textvqa' in args.question_file:
            rlt=counter.add_rlt(gt=gt, rlt=rlt)
        else:
            counter.add_rlt(gt=gt, rlt=rlt)
        rlt['gt']=gt
        
        logger.print_log(idx, ':', cur_prompt.replace('\n', ' '))
        logger.print_log(counter.total+q_start, ':', rlt)
        if counter.total%20==0 and counter.total>0:
            metric=counter.get_metric(logger=logger)
            # if metric['0.2']['ACC']>0.5:
            #     break
        adv_data_dict[idx]={
            'text':cur_prompt,
            # 'adv_image_tensor':adv_images.detach().cpu(),
            # 'raw_image_tensor': image_tensor.detach().cpu(), 
            'image_sizes': image_sizes, 
            'gt': gt,
            'rlt':rlt,
            'input_ids': input_ids.detach().cpu(),
            'adv_input_embeds': adv_input_embeds.detach().cpu(),
            'raw_input_embeds': new_input_embeds.detach().cpu(),
        }
        del adv_images, adv_input_embeds, image_tensor, new_input_embeds, input_ids
        torch.cuda.empty_cache()
    counter.get_metric(logger=logger)
    data_path='saved_data/'+'_'.join([
        model_name, 
        dataset_name,
        'adv_image',
        str(args.n_patch_num),
        str(args.l_patch_num),
        str(int(args.epsilon*255)),
        str(args.adv_steps),
        str(args.adv_lr),
        str(args.target_layer_idx),
        str(patch_or_noise),
        str(most_or_least),
        str(args.cut_num),
        str(args.atk),
        str(args.re_scale),
        str(len(questions)),
        str(most_flag),
        str(w_bpr_l), str(w_bpr_p),
        str(w_dir_l), str(w_dir_p),
        str(w_bpr_lp),
        str(w_ori_l), str(w_ori_p),
        str(stop_teps), str(stop_ratio),
    ])+'.pt'
    if args.adv_steps>0:
        torch.save(adv_data_dict, data_path)
    logger.print_log('saved_data_path:', data_path)
    
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
