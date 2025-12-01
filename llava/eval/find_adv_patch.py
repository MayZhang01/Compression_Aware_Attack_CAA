import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import math
from torch.optim.lr_scheduler import StepLR
from llava.eval.utils import loss_norm, intersection_ratio, add_square_patch, save_clip_tensor_as_image
from llava.eval.find_adv_util import cat_adv_embed, trace_transformer_layers_grad, get_attn_mean, BPR_loss, weighted_sign_mse_loss, mse_loss

from llava.eval.find_adv_util import get_patch_mask


def opt_adv_image_patch(
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
    image_size=336,
    patch_size=14,
    logger=None,
    patch_or_noise=False,
    most_flag=False,
    w_bpr_l=20, w_bpr_p=20, w_dir_l=10, w_dir_p=10, w_bpr_lp=10,
    w_ori_l=-2, w_ori_p=-2,
    stop_teps=80, stop_ratio=0.8,
):

    device = input_embeds.device
    dtype=input_embeds.dtype
    _, input_num, embed_dim = input_embeds.shape
    v_token_num = image_shape
    v_token_start = pre_prompt_length_list[0] if len(pre_prompt_length_list) != 0 else 0 
    t_token_start = v_token_start+v_token_num
    most_num=int(v_token_num*0.3)
    second_num=int(v_token_num*0.4)
    least_num=v_token_num-most_num-second_num
    n_patch_num=int(n_patch_num*v_token_num)
    l_patch_num=int(l_patch_num*v_token_num)
    save_clip_tensor_as_image(images.cpu(),'1.png')
    image_max=images.max()

    fixed_input_embeds = input_embeds.detach().clone()
    opt_params=[]
    
    adv_perturbs = torch.randn_like(images, dtype=dtype, device=device)
    adv_perturbs = adv_perturbs*images.std()
    adv_perturbs = torch.clamp(adv_perturbs, -epsilon*image_max, epsilon*image_max)
    opt_params.append(adv_perturbs)

    adv_patch = torch.randn_like(images, dtype=dtype, device=device)
    adv_patch = adv_patch*images.std()
    if patch_or_noise==False:
        adv_patch = torch.clamp(adv_patch, -epsilon*image_max, epsilon*image_max)
    opt_params.append(adv_patch)

    torch.manual_seed(123)                   
    torch.cuda.manual_seed(123)
    
    if most_flag:
        adv_mosts = torch.randn_like(images, dtype=dtype, device=device)
        adv_mosts = adv_mosts*images.std()
        adv_mosts = torch.clamp(adv_mosts, -epsilon*image_max, epsilon*image_max)
        opt_params.append(adv_mosts)
    
    patch_inter_ratio, least_inter_ratio=0,0
    if adv_steps>0: 
        adv_perturbs.requires_grad_(True)
        adv_patch.requires_grad_(True)
        optimizer = torch.optim.SGD(opt_params, lr=adv_lr)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    
    with torch.no_grad():   
        ben_image_embeds=model.encode_images(images)
        ben_input_embeds=cat_adv_embed(
            fixed_input_embeds, ben_image_embeds, 
            start_idx=v_token_start, end_idx=t_token_start
        )
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
        
        ben_attn_v_tar_mean =ben_attn_score_list_nc[target_layer_idx].mean(1)[0, input_num-1 ,v_token_start:t_token_start]
        _, topk_v_tar  = torch.topk(ben_attn_v_tar_mean, k=v_token_num)
        topk_v_tar  = topk_v_tar+v_token_start
        
        all_indices = torch.arange(ben_attn_score_list_nc[target_layer_idx].size(3), device=device)

        most_mask   = torch.zeros_like(all_indices, dtype=torch.bool)
        second_mask = torch.zeros_like(all_indices, dtype=torch.bool)
        least_mask  = torch.zeros_like(all_indices, dtype=torch.bool)
        noise_patch_mask  = torch.zeros_like(all_indices, dtype=torch.bool)
        least_patch_mask  = torch.zeros_like(all_indices, dtype=torch.bool)
        most_mask[topk_v_tar[:most_num]] = True
        second_mask[topk_v_tar[most_num:most_num+second_num]] = True
        least_mask[topk_v_tar[most_num+second_num:]] = True

        if l_patch_num>0:
            least_patch_mask[topk_v_tar[-l_patch_num:-n_patch_num]] = True
            least_mask[topk_v_tar[-l_patch_num:-n_patch_num]] = False
            most_mask[topk_v_tar[-l_patch_num:-n_patch_num]]  = False
            second_mask[topk_v_tar[-l_patch_num:-n_patch_num]]= False

            pixel_least_idx = topk_v_tar[-l_patch_num:-n_patch_num]-v_token_start
            pixel_least_mask=get_patch_mask(
                pixel_least_idx, 
                image_size=image_size, 
                patch_size=patch_size
            )
            pixel_least_mask = pixel_least_mask.to(device)
        else:
            pixel_least_mask=torch.zeros((image_size, image_size), dtype=dtype, device=device)

        if n_patch_num>0:
            noise_patch_mask[topk_v_tar[-n_patch_num:]] = True
            most_mask[topk_v_tar[-n_patch_num:]]  = False
            second_mask[topk_v_tar[-n_patch_num:]]= False
            least_mask[topk_v_tar[-n_patch_num:]] = False

            pixel_patch_idx = topk_v_tar[-n_patch_num:]-v_token_start
            pixel_patch_mask= get_patch_mask(pixel_patch_idx, image_size=image_size, patch_size=patch_size)
            pixel_patch_mask= pixel_patch_mask.to(device)
        else:
            pixel_patch_mask=torch.zeros((image_size, image_size), device=device, dtype=dtype)

        if most_flag:
            pixel_most_idx = topk_v_tar[:-l_patch_num]-v_token_start
            pixel_most_mask=get_patch_mask(
                pixel_most_idx, 
                image_size=image_size, 
                patch_size=patch_size
            )
            pixel_most_mask = pixel_most_mask.to(device)

        
        # Get first transformer block
        first_block = model.model.layers[target_layer_idx]
        # The Q and K projection layers are Linear modules
        q_proj = first_block.self_attn.q_proj
        k_proj = first_block.self_attn.k_proj

        # Extract the projection weight matrices
        # Each has shape: (hidden_dim, hidden_dim)
        W_Q = q_proj.weight.detach()
        W_K = k_proj.weight.detach()

        # print("W_Q shape:", W_Q.shape)
        # print("W_K shape:", W_K.shape)

        # Select the first token embedding (for example)
        # Here x is (hidden_dim,)
        if target_layer_idx>0:
            x = ben_hidden_states_list_nc[target_layer_idx-1][0, -1]
        else:
            x = fixed_input_embeds[0, -1]

        # Compute q0 = x @ W_Q
        q0 = torch.matmul(x, W_Q)
        # print("q0 vector norm:", q0.norm().item())

        # Compute v = W_K.T @ q0
        v_dir = torch.matmul(W_K.T, q0)
        v = v_dir / (v_dir.norm() + 1e-12)       # 方向对齐；若想更大logit，可再乘一个标量α（非Pre-Norm时）
        maxQ_patch=v.unsqueeze(0).repeat(n_patch_num, 1)
        if l_patch_num>0:
            maxQ_least=v.unsqueeze(0).repeat(l_patch_num-n_patch_num, 1)
        # maxQ_least=v.unsqueeze(0).repeat(l_patch_num, 1)
        # noise_patch=torch.randn_like(ben_input_embeds[:,noise_patch_mask], dtype=dtype, device=device)
        # noise_least=torch.randn_like(ben_input_embeds[:,least_patch_mask], dtype=dtype, device=device)

    
    if adv_steps==0:
        adv_steps+=1
        update_flag=False
    else:
        update_flag=True
    last_adv_perturbs=torch.zeros_like(adv_perturbs)
    last_adv_patch=torch.zeros_like(adv_patch)
    if most_flag:
        last_adv_mosts=torch.zeros_like(adv_mosts)
    
    for step in range(adv_steps):
        adv_images=images.clone()
        if patch_or_noise:
            adv_images=adv_images*(1-pixel_patch_mask)+adv_patch*pixel_patch_mask
        else:
            adv_images=adv_images+adv_patch*pixel_patch_mask
        adv_images=adv_images+adv_perturbs*pixel_least_mask
        if most_flag:
            adv_images=adv_images+adv_mosts*pixel_most_mask
        # adv_images = torch.clamp(adv_images, -image_max*1.1, image_max*1.1)
        save_clip_tensor_as_image(adv_images.cpu(),'2.png')
        adv_image_embeds = model.encode_images(adv_images)
        adv_input_embeds = cat_adv_embed(
            fixed_input_embeds, adv_image_embeds, 
            start_idx=v_token_start, end_idx=t_token_start
        )
        
        hidden_states_list_nc, attn_scores_list_nc, attn_logits_list_nc= trace_transformer_layers_grad(
            model, adv_input_embeds, attention_mask,
            end_layer=target_layer_idx,  
            return_all=True,
        )
        
        attn_logits_nc = attn_logits_list_nc[target_layer_idx]
        attn_logits_nc_mean = attn_logits_nc.mean(1)
        attn_logits_nc_last = attn_logits_nc_mean[:,input_num-1,:]

        attn_scores_nc = attn_scores_list_nc[target_layer_idx]
        attn_scores_nc_mean = attn_scores_nc.mean(1)
        attn_scores_nc_last = attn_scores_nc_mean[:,input_num-1,:]
           
        loss_list, loss_dict=[], {}
        if l_patch_num>0:
            loss_bpr_l=BPR_loss(
                attn_logits_nc_last[0], 
                p_list=[ 
                    least_patch_mask, 
                    most_mask,
                ] 
            )
            loss_list.append(torch.relu(loss_bpr_l-bpr_ths)*w_bpr_l)
            loss_dict['bpr_l']=round(loss_bpr_l.item(),2)
            
            loss_dir_l=mse_loss(
                hidden_states_list_nc[target_layer_idx][:,least_patch_mask], 
                maxQ_least, 
                k=1
            )
            loss_list.append((loss_dir_l)*w_dir_l)
            loss_dict['dir_l']=round(loss_dir_l.item(),2)
            
            loss_ori_l=mse_loss(
                hidden_states_list_nc[target_layer_idx][:,least_patch_mask], 
                ben_hidden_states_list_nc[target_layer_idx][:,least_patch_mask], 
                k=1
            )
            loss_list.append(torch.relu(2-loss_ori_l)*w_ori_l)
            loss_dict['ori_l']=round(loss_ori_l.item(),2)

        if n_patch_num>0:
            loss_bpr_p=BPR_loss(
                attn_logits_nc_last[0], 
                p_list=[
                    noise_patch_mask, 
                    most_mask,
                ] 
            )
            loss_list.append(torch.relu(loss_bpr_p-bpr_ths)*w_bpr_p)
            loss_dict['bpr_p']=round(loss_bpr_p.item(),2)
            
            loss_dir_p=mse_loss(
                hidden_states_list_nc[target_layer_idx][:,noise_patch_mask], 
                maxQ_patch, 
                k=1
            )
            loss_list.append((loss_dir_p)*w_dir_p)
            loss_dict['dir_p']=round(loss_dir_p.item(),2)
            
            loss_ori_p=mse_loss(
                hidden_states_list_nc[target_layer_idx][:,noise_patch_mask], 
                ben_hidden_states_list_nc[target_layer_idx][:,noise_patch_mask], 
                k=1
            )
            loss_list.append(torch.relu(2-loss_ori_p)*w_ori_p)
            loss_dict['ori_p']=round(loss_ori_p.item(),2)
            
        if l_patch_num>0 and n_patch_num>0:
            loss_bpr_lp=BPR_loss(
                attn_logits_nc_last[0], 
                p_list=[
                    noise_patch_mask, 
                    least_patch_mask,
                ] 
            )
            loss_list.append(torch.relu(loss_bpr_lp-bpr_ths)*w_bpr_lp)
            loss_dict['bpr_lp']=round(loss_bpr_lp.item(),2)
        
        loss_bpr=BPR_loss(
            attn_logits_nc_last[0], 
            p_list=[
                most_mask,
                second_mask,
                least_mask, 
            ] 
        )
        # loss_list.append(torch.relu(loss_bpr-bpr_ths)*20)
        loss_dict['bpr_w']=round(loss_bpr.item(),2)
        
        loss = torch.stack(loss_list).sum()

        if torch.isnan(loss):
            logger.print_log('Loss is NaN, stop optimization.')
            adv_perturbs=last_adv_perturbs
            adv_patch=last_adv_patch
            if most_flag:
                adv_mosts=last_adv_mosts
            break
        
        last_adv_perturbs=adv_perturbs.detach().clone()
        last_adv_patch=adv_patch.detach().clone()
        if most_flag:
            last_adv_mosts=adv_mosts.detach().clone()

        if update_flag:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            adv_perturbs.data=torch.clamp(adv_perturbs.data, -epsilon*image_max, epsilon*image_max)
            if patch_or_noise==False:
                adv_patch.data=torch.clamp(adv_patch.data, -epsilon*image_max, epsilon*image_max)
            if most_flag:
                adv_mosts.data=torch.clamp(adv_mosts.data, -epsilon*image_max, epsilon*image_max)
        
        attn_sum = attn_scores_nc_last.sum()
        
        sys_attn_ratio  =attn_scores_nc_last[:,0:v_token_start].sum()/attn_sum
        image_attn_ratio=attn_scores_nc_last[:,v_token_start:t_token_start].sum()/attn_sum
        text_attn_ratio =attn_scores_nc_last[:,t_token_start:input_num+1].sum()/attn_sum
        
        tar_most   = attn_scores_nc_last[:, most_mask]
        tar_second = attn_scores_nc_last[:, second_mask]
        tar_least  = attn_scores_nc_last[:, least_mask]
        n_patch  = attn_scores_nc_last[:, noise_patch_mask]
        l_patch  = attn_scores_nc_last[:, least_patch_mask]

        tar_most_ratio  =(tar_most.sum()/attn_sum).item()
        tar_second_ratio =(tar_second.sum()/attn_sum).item()
        tar_least_ratio =(tar_least.sum()/attn_sum).item()
        n_patch_ratio =(n_patch.sum()/attn_sum).item()
        l_patch_ratio =(l_patch.sum()/attn_sum).item()

        tar_most_ratio  =(tar_most.sum()/attn_sum).item()
        tar_second_ratio=(tar_second.sum()/attn_sum).item()
        tar_least_ratio  =(tar_least.sum()/attn_sum).item()
        n_patch_ratio =(n_patch.sum()/attn_sum).item()
        l_patch_ratio =(l_patch.sum()/attn_sum).item()
        
        _, topk_image  = torch.topk(attn_scores_nc_last[:, v_token_start:t_token_start], k=v_token_num)
        topk_image=topk_image+v_token_start

        patch_inter_ratio=intersection_ratio(topk_image[0,:n_patch_num], topk_v_tar[-n_patch_num:])
        least_inter_ratio=intersection_ratio(topk_image[0,:l_patch_num], topk_v_tar[-l_patch_num:])

        logger.print_log(
            step, 
            'loss: ', round(loss.item(),1), loss_dict, 
            'inter_ratio', patch_inter_ratio, least_inter_ratio, 
            'attn_ratio:', round(sys_attn_ratio.item(),3), round(image_attn_ratio.item(),3), round(text_attn_ratio.item(),3),
            'n_patch: ', round(n_patch_ratio,3),
            'l_patch: ', round(l_patch_ratio,3),
            'most: ', round(tar_most_ratio,3), 
            'second: ', round(tar_second_ratio,3), 
            'least: ', round(tar_least_ratio,3), 
        )

        if n_patch_num==0:
            patch_inter_ratio=stop_ratio+0.1
        if l_patch_num==0:
            least_inter_ratio=stop_ratio+0.1
        if patch_inter_ratio >= stop_ratio and least_inter_ratio >= stop_ratio and step>stop_teps:
            logger.print_log('Patch intersection ratio satisfied, stop optimization.')
            break
        
        # del loss, loss_cur, attn_logits_nc, attn_map_mean, down_v_logits, up_v_logits
        # del hidden_states_list_nc, attn_scores_list_nc, attn_logits_list_nc
        del attn_logits_list_nc, attn_scores_list_nc
        del hidden_states_list_nc
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        adv_images=images.clone()
        if patch_or_noise:
            adv_images=adv_images*(1-pixel_patch_mask)+adv_patch*pixel_patch_mask
        else:
            adv_images=adv_images+adv_patch*pixel_patch_mask
        adv_images=adv_images+adv_perturbs*pixel_least_mask
        if most_flag:
            adv_images=adv_images+adv_mosts*pixel_most_mask
        # adv_images = torch.clamp(adv_images, -image_max*1.1, image_max*1.1)
        adv_image_embeds = model.encode_images(adv_images)
        adv_input_embeds=cat_adv_embed(
            fixed_input_embeds, adv_image_embeds, 
            start_idx=v_token_start, end_idx=t_token_start
        )
    save_clip_tensor_as_image(adv_images.cpu(),'3.png')
    return adv_images, adv_input_embeds, adv_input_embeds.shape[1]-input_embeds.shape[1], patch_inter_ratio, least_inter_ratio