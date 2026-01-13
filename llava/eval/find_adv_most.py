import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import math
from torch.optim.lr_scheduler import StepLR
from llava.eval.utils import loss_norm, intersection_ratio, add_square_patch, save_clip_tensor_as_image
from llava.eval.find_adv_util import cat_adv_embed, trace_transformer_layers_grad, get_attn_mean, BPR_loss, weighted_sign_mse_loss, mse_loss
from llava.eval.find_adv_util import get_patch_mask, effective_weight


def opt_adv_image_most(
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
    w_bpr=20, w_dir=10, w_ori=10, ori_bound=2,
    stop_teps=80, stop_ratio=0.8,
    sparse_flag=False, 
):

    device = input_embeds.device
    dtype=input_embeds.dtype
    _, input_num, embed_dim = input_embeds.shape
    v_token_num = image_shape
    v_token_start = pre_prompt_length_list[0] if len(pre_prompt_length_list) != 0 else 0 
    t_token_start = v_token_start+v_token_num
    t_token_num = input_num - t_token_start
    most_num=int(v_token_num*0.3)
    second_num=int(v_token_num*0.4)
    least_num=v_token_num-most_num-second_num
    n_patch_num=int(n_patch_num*v_token_num)
    l_patch_num=int(l_patch_num*v_token_num)
    save_clip_tensor_as_image(images.cpu(),'1.png')
    image_max=images.max()

    fixed_input_embeds = input_embeds.detach().clone()
    opt_params=[]

    adv_patch = torch.zeros_like(images, dtype=dtype, device=device)
    adv_patch = adv_patch*images.std()
    if patch_or_noise==False:
        adv_patch = torch.clamp(adv_patch, -epsilon*image_max, epsilon*image_max)
    opt_params.append(adv_patch)

    torch.manual_seed(123)                   
    torch.cuda.manual_seed(123)
    
    patch_inter_ratio, least_inter_ratio=0,0
    if adv_steps>0: 
        adv_patch.requires_grad_(True)
        optimizer = torch.optim.SGD(opt_params, lr=adv_lr)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    
    with torch.no_grad():   
        ben_image_embeds=model.encode_images(images)
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
        _, topk_v_tar  = torch.topk(ben_attn_v_tar_mean, k=v_token_num)
        topk_v_tar  = topk_v_tar+v_token_start

        
        most_mask   = torch.zeros_like(all_indices, dtype=torch.bool)
        other_mask = torch.zeros_like(all_indices, dtype=torch.bool)


        if n_patch_num>0:
            most_mask[topk_v_tar[:n_patch_num]] = True
            other_mask[topk_v_tar[n_patch_num:]] = True

            pixel_patch_idx = topk_v_tar[:n_patch_num]-v_token_start
            pixel_patch_mask= get_patch_mask(pixel_patch_idx, image_size=image_size, patch_size=patch_size)
            pixel_patch_mask= pixel_patch_mask.to(device)
        else:
            pixel_patch_mask=torch.zeros((image_size, image_size), device=device, dtype=dtype)

        
        # Get first transformer block
        first_block = model.model.layers[target_layer_idx]
        # The Q and K projection layers are Linear modules
        q_proj = first_block.self_attn.q_proj
        k_proj = first_block.self_attn.k_proj

        # Extract the projection weight matrices
        # Each has shape: (hidden_dim, hidden_dim)
        
        W_Q = effective_weight(q_proj)    # (out_q, in)
        W_K = effective_weight(k_proj)    # (out_k, in)

        # print("W_Q shape:", W_Q.shape)
        # print("W_K shape:", W_K.shape)

        # Select the first token embedding (for example)
        # Here x is (hidden_dim,)
        if target_layer_idx>0:
            x = ben_hidden_states_list_nc[target_layer_idx-1][0, all_txt_mask]
        else:
            x = fixed_input_embeds[0, all_txt_mask]

        # Compute q0 = x @ W_Q
        q0 = torch.matmul(x, W_Q)
        # print("q0 vector norm:", q0.norm().item())

        # Compute v = W_K.T @ q0
        v_dir = torch.matmul(W_K.T, q0.T).mean(1)   
        v = v_dir / (v_dir.norm() + 1e-12)       
        maxQ_patch=v.unsqueeze(0).repeat(n_patch_num, 1)

    
    if adv_steps==0:
        adv_steps+=1
        update_flag=False
    else:
        update_flag=True
    last_adv_patch=torch.zeros_like(adv_patch)
    
    for step in range(adv_steps):
        adv_images=images.clone()
        if patch_or_noise:
            adv_images=adv_images*(1-pixel_patch_mask)+adv_patch*pixel_patch_mask
        else:
            adv_images=adv_images+adv_patch*pixel_patch_mask
            
        save_clip_tensor_as_image(adv_images.cpu(),'2.png')
        adv_image_embeds = model.encode_images(adv_images)
        adv_input_embeds = cat_adv_embed(
            fixed_input_embeds, adv_image_embeds, 
            start_idx=v_token_start, end_idx=t_token_start
        )
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
            model, adv_input_embeds, attention_mask,
            end_layer=target_layer_idx,  
            return_all=True,
        )
        
        attn_logits_nc = attn_logits_list_nc[target_layer_idx]
        attn_logits_nc_mean = attn_logits_nc.mean(1)
        attn_logits_nc_last = attn_logits_nc_mean[:,all_txt_mask,:].mean(1)

        attn_scores_nc = attn_scores_list_nc[target_layer_idx]
        attn_scores_nc_mean = attn_scores_nc.mean(1)
        attn_scores_nc_last = attn_scores_nc_mean[:,all_txt_mask,:].mean(1)
           

        if n_patch_num>0:
            loss_bpr=BPR_loss(
                attn_logits_nc_last[0], 
                p_list=[
                    most_mask,
                    other_mask, 
                ] 
            )
            loss_list.append(torch.relu(loss_bpr-bpr_ths)*w_bpr)
            loss_dict['bpr']=round(loss_bpr.item(),2)
            
            loss_dir=mse_loss(
                hidden_states_list_nc[target_layer_idx][:,most_mask], 
                maxQ_patch, 
                k=1
            )
            loss_list.append((loss_dir)*w_dir)
            loss_dict['dir']=round(loss_dir.item(),2)
            
            loss_ori=mse_loss(
                hidden_states_list_nc[target_layer_idx][:,most_mask], 
                ben_hidden_states_list_nc[target_layer_idx][:,most_mask], 
                k=1
            )
            loss_list.append(torch.relu(ori_bound-loss_ori)*w_ori)
            loss_dict['ori']=round(loss_ori.item(),2)
    
        
        loss = torch.stack(loss_list).sum()

        if torch.isnan(loss):
            logger.print_log('Loss is NaN, stop optimization.')
            adv_patch=last_adv_patch
            break
        
        last_adv_patch=adv_patch.detach().clone()

        if update_flag:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if patch_or_noise==False:
                adv_patch.data=torch.clamp(adv_patch.data, -epsilon*image_max, epsilon*image_max)
        
        _, topk_image  = torch.topk(attn_scores_nc_last[:, v_token_start:t_token_start], k=v_token_num)
        topk_image=topk_image+v_token_start

        patch_inter_ratio=intersection_ratio(topk_image[0,:n_patch_num], topk_v_tar[:n_patch_num])

        logger.print_log(
            step, 
            'loss: ', round(loss.item(),1), loss_dict, 
            'inter_ratio', patch_inter_ratio, 
        )

        del attn_logits_list_nc, attn_scores_list_nc
        del hidden_states_list_nc
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        adv_images=images.clone()
        if patch_or_noise:
            adv_images=adv_images*(1-pixel_patch_mask)+adv_patch*pixel_patch_mask
        else:
            adv_images=adv_images+adv_patch*pixel_patch_mask
            
        adv_image_embeds = model.encode_images(adv_images)
        adv_input_embeds=cat_adv_embed(
            fixed_input_embeds, adv_image_embeds, 
            start_idx=v_token_start, end_idx=t_token_start
        )
    save_clip_tensor_as_image(adv_images.cpu(),'3.png')
    return adv_images, adv_input_embeds, adv_input_embeds.shape[1]-input_embeds.shape[1], patch_inter_ratio, least_inter_ratio