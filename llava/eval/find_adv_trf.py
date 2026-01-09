import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import math
import numpy as np
from torch.optim.lr_scheduler import StepLR
from llava.eval.utils import loss_norm, intersection_ratio, add_square_patch, save_clip_tensor_as_image, add_border
from llava.eval.utils import crop_and_resize_image
from llava.eval.find_adv_util import cat_adv_embed, trace_transformer_layers_grad, get_attn_mean, BPR_loss, weighted_sign_mse_loss, mse_loss
# from qwen2_5_vl.modeling_qwen2_5_vl import (
#     apply_multimodal_rotary_pos_emb, repeat_kv
# )
from llava.eval.find_adv_util import get_patch_mask, effective_weight, get_max_min_Q

def opt_adv_image_patch_transf(
    model, input_embeds,  attention_mask,
    image_shape,       
    pre_prompt_length_list, 
    images=None,   
    target_layer_idx=1, 
    adv_steps=20, 
    adv_lr=1e-1,
    epsilon=8/255,
    n_patch_num=50,
    l_patch_num=0,
    bpr_ths=0.1,
    patch_size=14,
    image_size=336,
    logger=None,
    patch_or_noise=False,
    cut_num=1,
    most_flag=True,
    w_bpr_l=20, w_bpr_p=20, w_dir_l=2, w_dir_p=2, w_bpr_lp=10,
    w_ori_l=-2, w_ori_p=-2,
    ori_bound=2, sparse_flag=False, 
    stop_teps=80, stop_ratio=0.5,
    atk_name='trf',b_w=2,
):

    device = input_embeds.device
    dtype=input_embeds.dtype
    _, input_num, embed_dim = input_embeds.shape
    v_token_num = image_shape
    v_token_start = pre_prompt_length_list[0] if len(pre_prompt_length_list) != 0 else 0 
    t_token_start = v_token_start+v_token_num
    t_token_num = input_num - t_token_start

    cut_figs=[(0,1), (1,0.9), (2,0.9), (3,0.9)]
    cut_figs=cut_figs[:cut_num]
    image_max=images.max()

    if 'least' in atk_name:
        b_w=0
    border_l, border_r,border_t,border_b=b_w,b_w,b_w,b_w
    (grid_t, channel, H, W)=images.shape
    H1=H-patch_size*(border_l+border_r)
    W1=W-patch_size*(border_t+border_b)
    grid_h=H//patch_size
    grid_w=W//patch_size
    left_border, right_border, top_border, bottom_border = border_l*patch_size, border_r*patch_size, border_t*patch_size, border_b*patch_size
    
    resized = F.interpolate(images, size=(H1, W1), mode='bilinear', align_corners=False)
    
    image_with_border, mask_padded=add_border(
        resized[0], 
        padding = (left_border, right_border, top_border, bottom_border), 
        value=0
    )
    half_mask_padded=mask_padded.clone()
    half_mask_padded[:mask_padded.shape[1]//2,:]=0

    torch.manual_seed(123)                   
    torch.cuda.manual_seed(123)
    fixed_input_embeds = input_embeds.detach().clone()

    if 'full' in atk_name:
        adv_patch = torch.zeros((channel, grid_h*patch_size, grid_w*patch_size), dtype=dtype, device=device)
        adv_least = torch.zeros((channel, grid_h*patch_size, grid_w*patch_size), dtype=dtype, device=device)
    else:
        adv_patch=torch.randn(channel*patch_size*patch_size, dtype=dtype, device=device)
        # adv_patch = adv_patch*images.std()
        # adv_patch.data=torch.clamp(adv_patch.data, -1.1*image_max, 1.1*image_max)
        adv_patch=torch.zeros_like(adv_patch)

        adv_least=torch.randn(channel*patch_size*patch_size, dtype=dtype, device=device)
        # adv_least = adv_least*images.std()
        # adv_least.data=torch.clamp(adv_least.data, -1.1*image_max, 1.1*image_max)
        adv_least=torch.zeros_like(adv_least)

    patch_inter_ratio, least_inter_ratio=0,0
    if adv_steps>0: 
        adv_patch.requires_grad_(True)
        adv_least.requires_grad_(True)
        optimizer = torch.optim.SGD([adv_patch, adv_least], lr=adv_lr)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    # b_grid_h=grid_h+border_t+border_b
    # b_grid_w=grid_w+border_l+border_r
    # n_image_grid_thw=torch.tensor([[grid_t, b_grid_h, b_grid_w]])
    # border_size=(b_grid_h*b_grid_w-(b_grid_h-border_t-border_b)*(b_grid_w-border_l-border_r))//4
    # n_input_num = input_num + border_size
    # new_t_token_start=t_token_start + border_size
    # n_v_token_num = v_token_num+border_size

    most_num=int(v_token_num*0.3)
    second_num=int(v_token_num*0.4)
    least_num=v_token_num-most_num-second_num
    n_patch_num=int(n_patch_num*v_token_num)
    l_patch_num=int(l_patch_num*v_token_num)
    if isinstance(input_num, int)==False:
        input_num = int(input_num.item())

    aug_masks = [
        crop_and_resize_image(mask_padded, cut_loc=cut_loc, cut_size=cut_size)[0].to(torch.uint8).unsqueeze(0)
        for (cut_loc, cut_size) in cut_figs
    ]
    aug_masks=torch.cat(aug_masks)

    with torch.no_grad():   
        ben_image = image_with_border.clone() + torch.randn_like(image_with_border)*image_with_border.std()*mask_padded*0.3
        save_clip_tensor_as_image(ben_image.cpu(),'1.png')
        ben_images = [
            crop_and_resize_image(ben_image, cut_loc=cut_loc, cut_size=cut_size)
            for (cut_loc, cut_size) in cut_figs
        ]
        ben_image_embeds = [
            model.encode_images(ben_image.unsqueeze(0))
            for ben_image in ben_images
        ]
        ben_input_embeds = [
            cat_adv_embed(
                fixed_input_embeds, ben_image_embed, 
                start_idx=v_token_start, end_idx=t_token_start
            )
            for ben_image_embed in ben_image_embeds
        ]
        ben_input_embeds=torch.cat(ben_input_embeds)

        
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
            # v_token_start=v_token_start,
            # v_token_num=v_token_num,
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
        all_indices = all_indices.repeat(aug_masks.shape[0],1)    
        
        txt_mask = torch.zeros(t_token_num, dtype=torch.bool)
        txt_mask[t_token_idx] = True
        un_txt_mask = torch.ones(t_token_num, dtype=torch.bool)
        un_txt_mask[t_token_idx] = False
        
        all_txt_mask   = torch.zeros_like(all_indices[0], dtype=torch.bool)
        t_token_idx = t_token_idx + t_token_start
        all_txt_mask[t_token_idx] = True
        
        ben_attn_v_tar_mean = ben_attn_score_list_nc[target_layer_idx].mean(1)[:, all_txt_mask ,v_token_start:t_token_start].mean(1)
        _, topk_v_tar  = torch.topk(ben_attn_v_tar_mean, k=v_token_num, dim=1)
        topk_v_tar  = topk_v_tar+v_token_start

        most_mask   = torch.zeros_like(all_indices, dtype=torch.bool)
        second_mask   = torch.zeros_like(all_indices, dtype=torch.bool)
        least_mask   = torch.zeros_like(all_indices, dtype=torch.bool)
        least_adv_mask   = torch.zeros_like(all_indices, dtype=torch.bool)
        patch_adv_mask   = torch.zeros_like(all_indices, dtype=torch.bool)
        for i in range(all_indices.shape[0]):
            most_mask[i,topk_v_tar[i,:most_num-n_patch_num]] = True
            second_mask[i,topk_v_tar[i,most_num:most_num+second_num]] = True
            least_mask[i,topk_v_tar[i,-least_num:-l_patch_num]] = True
            least_adv_mask[i,topk_v_tar[i,-l_patch_num:]] = True
            patch_adv_mask[i,topk_v_tar[i,most_num-n_patch_num:most_num]] = True

        pixel_least_idx = [topk_v_tar[i, -l_patch_num:]-v_token_start for i in range(cut_num)]
        pixel_least_mask= [
            get_patch_mask(
                pixel_least_idx[i], 
                image_size=image_size, 
                patch_size=patch_size
            ).to(device)
            for i in range(cut_num)
        ]
        pixel_patch_idx = [topk_v_tar[i, :n_patch_num]-v_token_start for i in range(cut_num)]
        pixel_patch_mask=[
            get_patch_mask(
                pixel_patch_idx[i], 
                image_size=image_size, 
                patch_size=patch_size
            ).to(device)
            for i in range(cut_num)
        ]
        pixel_most_idx = [topk_v_tar[i, :most_num]-v_token_start for i in range(cut_num)]
        pixel_most_mask=[
            get_patch_mask(
                pixel_most_idx[i], 
                image_size=image_size, 
                patch_size=patch_size
            ).to(device)
            for i in range(cut_num)
        ]
        pixel_second_idx = [topk_v_tar[i, most_num:most_num+second_num]-v_token_start for i in range(cut_num)]
        pixel_second_mask=[
            get_patch_mask(
                pixel_second_idx[i], 
                image_size=image_size, 
                patch_size=patch_size
            ).to(device)
            for i in range(cut_num)
        ]
        pixel_all_l_idx = [topk_v_tar[i, most_num+second_num:]-v_token_start for i in range(cut_num)]
        pixel_all_l_mask=[
            get_patch_mask(
                pixel_all_l_idx[i], 
                image_size=image_size, 
                patch_size=patch_size
            ).to(device)
            for i in range(cut_num)
        ]
        pixel_border_mask=mask_padded.unsqueeze(0).repeat((3,1,1)).to(device)
        
        # Select the first token embedding, x is (hidden_dim,)
        if target_layer_idx>0:
            x = ben_hidden_states_list_nc[target_layer_idx-1][0, all_txt_mask]
        else:
            x = fixed_input_embeds[0, all_txt_mask]
        
        # Get first transformer block
        first_block = model.model.layers[target_layer_idx]
        
        v_max, v_min=get_max_min_Q(x, first_block.self_attn, t_idx_num, dtype=dtype)
        maxQ_least=[
            v_max.unsqueeze(0).repeat(least_adv_mask[i].sum(), 1) 
            for i in range(cut_num)
        ]
        maxQ_patch=[
            v_min.unsqueeze(0).repeat(patch_adv_mask[i].sum(), 1) 
            for i in range(cut_num)
        ]
        
        
    if adv_steps==0:
        adv_steps+=1
        update_flag=False
    else:
        update_flag=True
    last_adv_patch=torch.zeros_like(adv_patch)
    last_adv_least=torch.zeros_like(adv_least)
    tmp_r_adv_image = ben_image.clone() 
    tmp_r_adv_image = torch.clamp(tmp_r_adv_image, -image_max*1.1, image_max*1.1)
    for step in range(adv_steps):
        if 'full' in atk_name:
            tmp_adv_patch = adv_patch
            tmp_adv_least = adv_least
        else:
            tmp_adv_patch = adv_patch.reshape(channel, 1, patch_size, 1, patch_size)
            tmp_adv_patch = tmp_adv_patch.repeat(1,grid_h,1,grid_w,1)
            tmp_adv_patch = tmp_adv_patch.reshape(channel, grid_h*patch_size, grid_w*patch_size)
            tmp_adv_least = adv_least.reshape(channel, 1, patch_size, 1, patch_size)
            tmp_adv_least = tmp_adv_least.repeat(1,grid_h,1,grid_w,1)
            tmp_adv_least = tmp_adv_least.reshape(channel, grid_h*patch_size, grid_w*patch_size)
        adv_image = tmp_r_adv_image.clone().detach() 
        # adv_image = adv_image + torch.randn_like(image_with_border)*image_with_border.std()*0.1#*(1-mask_padded)
        adv_images = [
            crop_and_resize_image(adv_image, cut_loc=cut_loc, cut_size=cut_size)
            for (cut_loc, cut_size) in cut_figs
        ]
        if 'full' in atk_name:
            adv_images=[
                adv_image 
                +pixel_all_l_mask[i]*tmp_adv_least*epsilon 
                +pixel_most_mask[i]*tmp_adv_patch*epsilon
                +pixel_second_mask[i]*tmp_adv_patch*epsilon
                for adv_image in adv_images
            ]
        else:
            adv_images=[
                adv_image 
                +pixel_least_mask[i]*tmp_adv_least*epsilon 
                +pixel_patch_mask[i]*tmp_adv_patch*epsilon
                for adv_image in adv_images
            ]
        adv_images=[
            torch.clamp(adv_image, -image_max*1.1, image_max*1.1)
            for adv_image in adv_images
        ]
        save_clip_tensor_as_image(adv_images[0].cpu(),'2.png')
        adv_image_embeds = [
            model.encode_images(adv_image.unsqueeze(0))
            for adv_image in adv_images
        ]
        tmp_input_embeds = fixed_input_embeds #*0.8+torch.randn_like(fixed_input_embeds)*fixed_input_embeds.std()*0.2
        adv_input_embeds = [
            cat_adv_embed(
                tmp_input_embeds, adv_image_embed, 
                start_idx=v_token_start, end_idx=t_token_start
            )
            for adv_image_embed in adv_image_embeds
        ]
        adv_input_embeds=torch.cat(adv_input_embeds)

        loss_list, loss_dict=[], {}
        if sparse_flag:
            img_adv_embs=adv_input_embeds[:, v_token_start:t_token_start]
            txt_adv_embs=adv_input_embeds[:, t_token_start:]
            img_txt_inter = img_adv_embs @ txt_adv_embs.transpose(1, 2) # [1, 576, 53]
            img_txt_inter = img_txt_inter.mean(1)#.softmax(2)
            loss_txt=img_txt_inter[:,un_txt_mask].max()-img_txt_inter[:,txt_mask].min()
            loss_list.append(torch.relu(loss_txt.mean()+0.5)*10)
            loss_dict['loss_txt']=round(loss_txt.item(),2)
        
        hidden_states_list_nc, attn_scores_list_nc, attn_logits_list_nc= trace_transformer_layers_grad(
            model,  
            adv_input_embeds,
            attention_mask,
            end_layer=target_layer_idx,  
            return_all=True,
        )
        
        attn_logits_nc=torch.vstack(attn_logits_list_nc[:target_layer_idx+1]).mean(0).unsqueeze(0)
        # attn_logits_nc = attn_logits_list_nc[target_layer_idx]
        attn_logits_nc_mean = attn_logits_nc.mean(1)
        attn_logits_nc_last = attn_logits_nc_mean[:,all_txt_mask,:].mean(1)

        attn_scores_nc = attn_scores_list_nc[target_layer_idx]
        attn_scores_nc_mean = attn_scores_nc.mean(1)
        attn_scores_nc_last = attn_scores_nc_mean[:,all_txt_mask,:].mean(1)
        
        lbp=[]
        loss_bpr_p=[
            BPR_loss(
                attn_logits_nc_last[i], 
                p_list=[
                    second_mask[i],
                    least_mask[i],
                    patch_adv_mask[i], 
                ],
                # t=5 
            )
            for i in range(cut_num)
        ]
        for loss_bpr_p_i in loss_bpr_p:
            lbp.append(torch.relu(loss_bpr_p_i-bpr_ths))
        loss_list.append(torch.stack(lbp).mean()*w_bpr_p)
        loss_dict['bpr_p']=[round(loss_bpr_p_i.item(),2) for loss_bpr_p_i in loss_bpr_p]
        
        lbm=[]
        loss_bpr_m=[
            BPR_loss(
                attn_logits_nc_last[i], 
                p_list=[
                    least_adv_mask[i],
                    most_mask[i],
                    second_mask[i],
                ],
                # t=5
            )
            for i in range(cut_num)
        ]
        for loss_bpr_m_i in loss_bpr_m:
            lbm.append(torch.relu(loss_bpr_m_i-bpr_ths))
        loss_list.append(torch.stack(lbm).mean()*w_bpr_l)
        loss_dict['bpr_m']=[round(loss_bpr_m_i.item(),2) for loss_bpr_m_i in loss_bpr_m]
        
        lbpm=[]
        loss_bpr_pm=[
            BPR_loss(
                attn_logits_nc_last[i], 
                p_list=[
                    least_adv_mask[i],
                    patch_adv_mask[i], 
                ],
            )
            for i in range(len(cut_figs))
        ]
        for loss_bpr_pm_i in loss_bpr_pm:
            lbpm.append(torch.relu(loss_bpr_pm_i-bpr_ths))
        loss_list.append(torch.stack(lbpm).mean()*w_bpr_lp)
        loss_dict['bpr_pm']=[round(loss_bpr_pm_i.item(),2) for loss_bpr_pm_i in loss_bpr_pm]
        
        ldp=[]
        loss_dir_p=[
            mse_loss(
                hidden_states_list_nc[target_layer_idx][i, patch_adv_mask[i]], 
                maxQ_patch[i], 
                k=1
            )
            for i in range(cut_num)
        ]
        for loss_dir_p_i in loss_dir_p:
            ldp.append((loss_dir_p_i))
        loss_list.append(torch.stack(ldp).mean()*w_dir_p)
        loss_dict['dir_p']=[round(loss_dir_p_i.item(),2) for loss_dir_p_i in loss_dir_p]

        ldl=[]
        loss_dir_l=[
            mse_loss(
                hidden_states_list_nc[target_layer_idx][i, least_adv_mask[i]], 
                maxQ_least[i], 
                k=1
            )
            for i in range(cut_num)
        ]
        for loss_dir_l_i in loss_dir_l:
            ldl.append((loss_dir_l_i))
        loss_list.append(torch.stack(ldl).mean()*w_dir_l)
        loss_dict['dir_l']=[round(loss_dir_l_i.item(),2) for loss_dir_l_i in loss_dir_l]

        lop=[]
        loss_ori_p=[
            [
                mse_loss(
                    hidden_states_list_nc[j][i, patch_adv_mask[i]], 
                    ben_hidden_states_list_nc[j][i, patch_adv_mask[i]], 
                    k=1
                )
                for j in range(target_layer_idx, target_layer_idx+1)
            ]
            for i in range(cut_num)
        ]
        for loss_ori_p_i in loss_ori_p:
            # lop.append((loss_ori_p_i))
            lop.append(torch.relu(ori_bound-torch.stack(loss_ori_p_i).mean()))
        loss_list.append(torch.stack(lop).mean()*w_ori_p)
        loss_dict['oir_p']=[round(torch.stack(loss_ori_p_i).mean().item(),2) for loss_ori_p_i in loss_ori_p]
        
        lol=[]
        loss_ori_l=[
            [
                mse_loss(
                    hidden_states_list_nc[j][i, least_adv_mask[i]], 
                    ben_hidden_states_list_nc[j][i, least_adv_mask[i]], 
                    k=1
                )
                for j in range(target_layer_idx, target_layer_idx+1)
            ]
            for i in range(cut_num)
        ]
        for loss_ori_l_i in loss_ori_l:
            # lol.append((loss_ori_l_i))
            lol.append(torch.relu(ori_bound-torch.stack(loss_ori_l_i).mean()))
        loss_list.append(torch.stack(lol).mean()*w_ori_l)
        loss_dict['oir_l']=[round(torch.stack(loss_ori_l_i).mean().item(),2) for loss_ori_l_i in loss_ori_l]

        loss = torch.stack(loss_list).sum()
        
        if torch.isnan(loss):
            logger.print_log('Loss is NaN, stop optimization.')
            adv_patch=last_adv_patch
            adv_least=last_adv_least
            break
        
        last_adv_patch=adv_patch.detach().clone()
        last_adv_least=adv_least.detach().clone()

        if update_flag:
            # optimizer.zero_grad()
            loss.backward()
            # optimizer.step()
            # scheduler.step()
            adv_patch.data=adv_patch.data-torch.sign(adv_patch.grad.data)*epsilon*0.3
            adv_least.data=adv_least.data-torch.sign(adv_least.grad.data)*epsilon*0.3
            adv_patch.grad.zero_()
            adv_least.grad.zero_()
            adv_patch.data=torch.clamp(adv_patch.data, -image_max*1.1, image_max*1.1)
            adv_least.data=torch.clamp(adv_least.data, -image_max*1.1, image_max*1.1)
        
        attn_sum = attn_scores_nc_last.sum(dim=1, keepdim=True).mean()
        
        sys_attn_ratio  =attn_scores_nc_last[:,0:v_token_start].sum(dim=1, keepdim=True)/attn_sum
        image_attn_ratio=attn_scores_nc_last[:,v_token_start:t_token_start].sum(dim=1, keepdim=True)/attn_sum
        text_attn_ratio =attn_scores_nc_last[:,t_token_start:].sum(dim=1, keepdim=True)/attn_sum

        tar_most = [
            attn_scores_nc_last[i, most_mask[i]]
            for i in range(cut_num)
        ]
        tar_second = [
            attn_scores_nc_last[i, second_mask[i]]
            for i in range(cut_num)
        ]
        tar_least = [
            attn_scores_nc_last[i, least_mask[i]]
            for i in range(cut_num)
        ]
        tar_all = [
            attn_scores_nc_last[i, v_token_start:t_token_start]
            for i in range(cut_num)
        ]
        n_patch = [
            attn_scores_nc_last[i, patch_adv_mask[i]]
            for i in range(cut_num)
        ]
        n_least = [
            attn_scores_nc_last[i, least_adv_mask[i]]
            for i in range(cut_num)
        ]

        tar_most_ratio = [
            (tar_most[i].sum()/attn_sum).item() 
            for i in range(cut_num)
        ]
        tar_second_ratio = [
            (tar_second[i].sum()/attn_sum).item() 
            for i in range(cut_num)
        ]
        tar_least_ratio = [
            (tar_least[i].sum()/attn_sum).item() 
            for i in range(cut_num)
        ]
        tar_all_ratio = [
            (tar_all[i].sum()/attn_sum).item() 
            for i in range(cut_num)
        ]
        n_patch_ratio = [
            (n_patch[i].sum()/attn_sum).item() 
            for i in range(cut_num)
        ]
        n_least_ratio = [
            (n_least[i].sum()/attn_sum).item() 
            for i in range(cut_num)
        ]
        
        _, topk_image = torch.topk(attn_scores_nc_last[:, v_token_start:t_token_start], k=v_token_num, dim=1)
        topk_image=topk_image+v_token_start

        patch_inter_ratio=[
            (
                intersection_ratio(topk_image[i,-most_num:], topk_v_tar[i,:most_num]),
                intersection_ratio(topk_image[i,:most_num], topk_v_tar[i,-most_num:]),
                intersection_ratio(topk_image[i,:most_num], topk_v_tar[i,:most_num]),
                intersection_ratio(topk_image[i,-most_num:], topk_v_tar[i,-most_num:])
            )
            for i in range(cut_num)
        ]

        logger.print_log(
            step, 
            'loss:', round(loss.item(),1), loss_dict, 
            'inter_ratio:', patch_inter_ratio,
            'all:', round(np.mean(tar_all_ratio),3), 
            'n_patch:', round(np.mean(n_patch_ratio),3), round(np.mean(n_least_ratio),3),
            'each:', round(np.mean(tar_most_ratio),3), round(np.mean(tar_second_ratio),3), round(np.mean(tar_least_ratio),3), 
            'attn_ratio:', round(sys_attn_ratio.mean().item(),3), round(image_attn_ratio.mean().item(),3), round(text_attn_ratio.mean().item(),3),
        )
        # if np.min(patch_inter_ratio) >= 0.2 and step>60:
        #     logger.print_log('Patch intersection ratio satisfied, stop optimization.')
        #     break

        # if (loss_bpr1.item()<=bpr_ths):# or (loss_bpr2.item()<=0.9): #  and tar_up_v_map.min()>tar_down_v_map.max()
        #     break
        
        # del loss, loss_cur, attn_logits_nc, attn_map_mean, down_v_logits, up_v_logits
        del adv_image, adv_images, adv_image_embeds, adv_input_embeds
        del hidden_states_list_nc, attn_scores_list_nc, attn_logits_list_nc
        del attn_logits_nc, attn_logits_nc_mean, attn_logits_nc_last
        del attn_scores_nc, attn_scores_nc_mean, attn_scores_nc_last
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        
        if 'full' in atk_name:
            tmp_adv_patch = adv_patch
            tmp_adv_least = adv_least
        else:
            tmp_adv_patch = adv_patch.reshape(channel, 1, patch_size, 1, patch_size)
            tmp_adv_patch = tmp_adv_patch.repeat(1,grid_h,1,grid_w,1)
            tmp_adv_patch = tmp_adv_patch.reshape(channel, grid_h*patch_size, grid_w*patch_size)
            
            tmp_adv_least = adv_least.reshape(channel, 1, patch_size, 1, patch_size)
            tmp_adv_least = tmp_adv_least.repeat(1,grid_h,1,grid_w,1)
            tmp_adv_least = tmp_adv_least.reshape(channel, grid_h*patch_size, grid_w*patch_size)
        
        if ('least' in atk_name) or ('full' in atk_name) :
            adv_image = image_with_border.clone()
            adv_image = adv_image + pixel_all_l_mask[0]*tmp_adv_least*epsilon
            adv_image = adv_image + pixel_most_mask[0]*tmp_adv_patch*epsilon  
            adv_image = adv_image + pixel_second_mask[0]*tmp_adv_patch*epsilon 
        else:
            adv_image = image_with_border.clone() + tmp_adv_least*mask_padded#*epsilon
            # adv_image = adv_image + torch.randn_like(image_with_border)*image_with_border.std()*mask_padded*0.1#*epsilon
            # adv_image = adv_image + tmp_adv_patch*(1-mask_padded)*0.2
            # adv_image = torch.clamp(adv_image, -image_max*1.1, image_max*1.1)
            # adv_image_tensor = image2tensor(adv_image, grid_h, grid_w, patch_size=patch_size) 
            if most_flag:
                adv_image = adv_image + (pixel_all_l_mask[0]*(1-pixel_border_mask))*tmp_adv_patch*epsilon 
                adv_image = adv_image + (pixel_most_mask[0]*(1-pixel_border_mask))*tmp_adv_patch*epsilon  
                adv_image = adv_image + (pixel_second_mask[0]*(1-pixel_border_mask))*tmp_adv_patch*epsilon   
        
        # adv_image_tensor = adv_image_tensor + (pixel_all_l_mask[0]*pixel_border_mask)*image2tensor(tmp_adv_least, grid_h, grid_w, patch_size=patch_size)*(1-epsilon)
        # adv_image_tensor = adv_image_tensor + (pixel_most_mask[0]*pixel_border_mask)*image2tensor(tmp_adv_patch, grid_h, grid_w, patch_size=patch_size)*(1-epsilon)
        # adv_image_tensor = adv_image_tensor + (pixel_second_mask[0]*pixel_border_mask)*image2tensor(tmp_adv_patch, grid_h, grid_w, patch_size=patch_size)*(1-epsilon)
        # adv_image_tensor = adv_image_tensor + pixel_all_l_mask[0]*image2tensor(tmp_adv_least, grid_h, grid_w, patch_size=patch_size)*epsilon 
        # adv_image_tensor = adv_image_tensor + pixel_most_mask[0]*image2tensor(tmp_adv_patch, grid_h, grid_w, patch_size=patch_size)*epsilon
        # adv_image_tensor = adv_image_tensor + pixel_second_mask[0]*image2tensor(tmp_adv_patch, grid_h, grid_w, patch_size=patch_size)*epsilon
        adv_image = torch.clamp(adv_image, -image_max*1.1, image_max*1.1)
        adv_image_embed = model.encode_images(adv_image.unsqueeze(0))
        adv_input_embed = cat_adv_embed(
            fixed_input_embeds, adv_image_embed, 
            start_idx=v_token_start, end_idx=t_token_start
        )
    # adv_image = reshape_image(adv_image_tensor, grid_t, grid_h, grid_w, channel, temporal_patch_size, patch_size)    
    save_clip_tensor_as_image(adv_image.cpu(),'3.png')
    
    return adv_image, adv_input_embed, adv_input_embed.shape[1]-input_embeds.shape[1], patch_inter_ratio, least_inter_ratio, adv_patch, adv_least