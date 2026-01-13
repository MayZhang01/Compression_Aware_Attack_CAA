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
from llava.eval.find_adv_patch import opt_adv_image_patch
from llava.eval.find_adv_trf import opt_adv_image_patch_transf
from llava.eval.find_adv_most import opt_adv_image_most
from llava.eval.utils import Logger, extract_name_from_path, print_args, get_arg_or_default
from llava.eval.find_adv_util import judge_adv
import numpy as np
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
from llava.eval.model_vqa_loader_util import split_list, get_chunk, CustomDataset, Counter_acc, Counter, create_data_loader, collate_fn

from llava.eval.find_adv_vnl import opt_adv_image_vanilla


from llava.eval.utils import loss_norm, intersection_ratio, add_square_patch, save_clip_tensor_as_image, add_border
from llava.eval.utils import crop_and_resize_image
from llava.eval.find_adv_util import cat_adv_embed, trace_transformer_layers_grad, get_attn_mean, BPR_loss, weighted_sign_mse_loss, mse_loss
from llava.eval.find_adv_util import get_patch_mask, effective_weight, get_max_min_Q

import torch
import torch.nn.functional as F

def highlight_tokens_soft(
    image: torch.Tensor,              # (3,H,W)
    pixel_least_mask: torch.Tensor,    # (H,W)
    pixel_most_mask: torch.Tensor,     # (H,W)
    alpha_gray: float = 0.25,
    alpha_color: float = 0.7,
):
    """
    使用浅灰 / 浅蓝 / 浅红的半透明覆盖
    """

    device = image.device
    dtype = image.dtype
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(3, 1, 1)
    image = image*std+mean
    image = image.clamp(0,1)
    assert image.shape[0] == 3
    _, H, W = image.shape


    img = image.clone()

    least = torch.as_tensor(pixel_least_mask, device=device).bool()
    most  = torch.as_tensor(pixel_most_mask,  device=device).bool()
    other = ~(least | most)

    # 判断值域
    img_max = float(img.max())
    scale = 255.0 if img_max > 1.5 else 1.0

    # 颜色定义（浅色）
    gray = torch.tensor([0.85, 0.85, 0.85], device=device, dtype=dtype) * scale
    blue = torch.tensor([0.70, 0.82, 0.95], device=device, dtype=dtype) * scale
    red  = torch.tensor([0.95, 0.70, 0.70], device=device, dtype=dtype) * scale

    # 其余 token：浅灰
    if other.any():
        img[:, other] = (
            (1 - alpha_gray) * img[:, other]
            + alpha_gray * gray[:, None]
        )

    # least token：浅蓝
    if least.any():
        img[:, least] = (
            (1 - alpha_color) * img[:, least]
            + alpha_color * blue[:, None]
        )

    # most token：浅红
    if most.any():
        img[:, most] = (
            (1 - alpha_color) * img[:, most]
            + alpha_color * red[:, None]
        )
    img=(img-mean)/std
    return img



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
    random.seed(124)
    random.shuffle(questions)
    q_start=0
    questions = questions[q_start:50]
    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, conv_mode=args.conv_mode)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    
    layer_list=eval(args.layer_list)
    if layer_list==[2,6,15]:
        sparse_flag=True
    else:
        sparse_flag=False

    target_layer_idx=args.target_layer_idx
    image_shape=576
    patch_size=args.patch_size
    image_size=args.image_size
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
                input_embeds,
                _,  
            ) = model.prepare_inputs_labels_for_multimodal_pdrop(       
                input_ids, position_ids=None, attention_mask=None, past_key_values=None, labels=None,
                # images=torch.tensor([0]),#
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True), 
                image_sizes=image_sizes,
                # image_feature_size=(1,576,4096)
            )
            ori_prompt_len=model.model.prompt_len[0]
        pre_prompt_length_list=[torch.where(input_ids==-200)[1][0].item()]
        device = input_embeds.device
        dtype=input_embeds.dtype
        _, input_num, embed_dim = input_embeds.shape
        v_token_num = image_shape

        v_token_start = pre_prompt_length_list[0] if len(pre_prompt_length_list) != 0 else 0 
        t_token_start = v_token_start+v_token_num
        t_token_num = input_num - t_token_start
        
        torch.manual_seed(123)                   
        torch.cuda.manual_seed(123)
        fixed_input_embeds = input_embeds.detach().clone()

        most_num=int(v_token_num*0.1)
        second_num=int(v_token_num*0.8)
        least_num=v_token_num-most_num-second_num
        n_patch_num=int(args.n_patch_num*v_token_num)
        l_patch_num=int(args.l_patch_num*v_token_num)
        if isinstance(input_num, int)==False:
            input_num = int(input_num.item())
       
        ben_image=crop_and_resize_image(image_tensor[0], cut_loc=4, cut_size=0.65).unsqueeze(0).to(device)
        
        with torch.no_grad():  
            ben_image_embeds=model.encode_images(ben_image)
            ben_input_embeds=cat_adv_embed(
                fixed_input_embeds, ben_image_embeds, 
                start_idx=v_token_start, end_idx=t_token_start
            )

            if sparse_flag:
                img_embs=ben_input_embeds[:, v_token_start:t_token_start]
                txt_embs=ben_input_embeds[:, t_token_start:]
                img_txt_inter = img_embs @ txt_embs.transpose(1, 2) # [1, 576, 53]
                img_txt_inter = img_txt_inter.softmax(2).mean(1)#
                _, t_token_idx = torch.where(img_txt_inter > img_txt_inter.mean())

            ben_hidden_states_list_nc, ben_attn_score_list_nc, _= trace_transformer_layers_grad(
                model,  
                ben_input_embeds,
                attention_mask,
                start_layer=0,
                end_layer=target_layer_idx,  
                return_all=True,
                v_token_start=v_token_start,
                v_token_num=v_token_num,
            )

            if sparse_flag:
                img_embs=ben_input_embeds[:, v_token_start:t_token_start]
                txt_embs=ben_input_embeds[:, t_token_start:]
                img_txt_inter = img_embs @ txt_embs.transpose(1, 2) # [1, 576, 53]
                img_txt_inter = img_txt_inter.softmax(2).mean(1)#
                _, t_token_idx = torch.where(img_txt_inter > img_txt_inter.mean())
                t_idx_num=t_token_idx.size()[0]
            else:
                t_token_idx = torch.tensor([-1], device=device)
                t_idx_num=1
            
            all_indices = torch.arange(ben_attn_score_list_nc[target_layer_idx].size(3), device=device)

            txt_mask = torch.zeros(t_token_num, dtype=torch.bool)
            txt_mask[t_token_idx] = True
            un_txt_mask = torch.ones(t_token_num, dtype=torch.bool)
            un_txt_mask[t_token_idx] = False
            
            all_txt_mask   = torch.zeros_like(all_indices, dtype=torch.bool)
            t_token_idx = t_token_idx + t_token_start
            all_txt_mask[t_token_idx] = True
            
            ben_attn_v_tar_mean = ben_attn_score_list_nc[target_layer_idx].mean(1)[0, all_txt_mask ,v_token_start:t_token_start].mean(0)
            topk_v_score, topk_v_tar  = torch.topk(ben_attn_v_tar_mean, k=v_token_num)
            topk_v_tar  = topk_v_tar+v_token_start

            pixel_least_idx = topk_v_tar[-least_num:]-v_token_start
            pixel_least_mask= get_patch_mask(
                pixel_least_idx, 
                image_size=image_size, 
                patch_size=patch_size
            ).to(device)
            
            pixel_most_idx = topk_v_tar[:most_num]-v_token_start
            pixel_most_mask=get_patch_mask(
                pixel_most_idx, 
                image_size=image_size, 
                patch_size=patch_size
            ).to(device)
            h_image=highlight_tokens_soft(
                ben_image[0],
                pixel_least_mask,
                pixel_most_mask,
                alpha_gray=0.25,
                alpha_color=0.7,
            )
            save_clip_tensor_as_image(h_image, 'saved_img/'+str(idx)+'_'+model_name+'.png')