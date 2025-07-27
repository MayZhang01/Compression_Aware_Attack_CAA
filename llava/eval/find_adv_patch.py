import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import math
from torch.optim.lr_scheduler import StepLR
from llava.eval.utils import loss_norm, intersection_ratio, add_square_patch, get_patch_mask, save_clip_tensor_as_image
from llava.eval.find_adv_util import cat_adv_embed, trace_transformer_layers_grad, get_attn_mean, BPR_loss, weighted_sign_mse_loss



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
    ):

    device = input_embeds.device
    dtype=input_embeds.dtype
    _, input_num, embed_dim = input_embeds.shape
    v_token_num = image_shape
    v_token_start = pre_prompt_length_list[0] if len(pre_prompt_length_list) != 0 else 0 
    t_token_start = v_token_start+v_token_num

    
    image_max=images.max()
    
    if epsilon>0:
        adv_perturbs = torch.randn_like(images, dtype=dtype, device=device)
        adv_perturbs = adv_perturbs*images.std()
        # adv_perturbs = adv_perturbs*epsilon
        adv_perturbs = torch.clamp(adv_perturbs, -epsilon*image_max, epsilon*image_max)
        # adv_perturbs = torch.zeros_like(images, dtype=dtype, device=device)
    else:
        adv_perturbs = torch.zeros_like(images, dtype=dtype, device=device)
    
    torch.manual_seed(123)                   
    torch.cuda.manual_seed(123)
    fixed_input_embeds = input_embeds.detach()

    adv_patch=torch.randn_like(images, dtype=dtype, device=device)*images.std()
    # adv_patch=torch.zeros_like(images, dtype=dtype, device=device)
    

    flag=True
    patch_inter_ratio, least_inter_ratio=0,0
    if adv_steps>0: 
        adv_perturbs.requires_grad_(True)
        adv_patch.requires_grad_(True)
        optimizer = torch.optim.SGD([adv_perturbs, adv_patch], lr=adv_lr)
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
        # ben_attn_logits_mean=get_attn_mean(ben_attn_map_list_nc[target_layer_idx].mean(1))
        ben_attn_v_tar_mean =ben_attn_score_list_nc[target_layer_idx].mean(1)[0, -1 ,v_token_start:t_token_start]
        _, topk_v_tar  = torch.topk(ben_attn_v_tar_mean, k=v_token_num)
        topk_v_tar  = topk_v_tar+v_token_start
        
        all_indices = torch.arange(ben_attn_score_list_nc[target_layer_idx].size(3), device=device)

        most_mask   = torch.zeros_like(all_indices, dtype=torch.bool)
        second_mask = torch.zeros_like(all_indices, dtype=torch.bool)
        least_mask   = torch.zeros_like(all_indices, dtype=torch.bool)
        noise_patch_mask  = torch.zeros_like(all_indices, dtype=torch.bool)
        least_patch_mask   = torch.zeros_like(all_indices, dtype=torch.bool)
        most_mask[topk_v_tar[:192]] = True
        second_mask[topk_v_tar[193:384]] = True
        least_mask[topk_v_tar[385:]] = True
        

        if l_patch_num>0:
            least_patch_mask[topk_v_tar[-l_patch_num:]] = True
            least_mask[topk_v_tar[-l_patch_num:]] = False

            pixel_least_idx = topk_v_tar[-l_patch_num:]-v_token_start
            pixel_least_mask=get_patch_mask(pixel_least_idx, image_size=image_size, patch_size=patch_size)
            pixel_least_mask = pixel_least_mask.to(device)
        else:
            pixel_least_mask=torch.zeros((image_size, image_size), dtype=dtype, device=device)

        if n_patch_num>0:
            noise_patch_mask[t_token_start-n_patch_num:t_token_start]  = True
            most_mask[t_token_start-n_patch_num:t_token_start]  = False
            second_mask[t_token_start-n_patch_num:t_token_start]  = False
            least_mask[t_token_start-n_patch_num:t_token_start]  = False

            pixel_patch_idx = all_indices[t_token_start-n_patch_num:t_token_start]-v_token_start
            pixel_patch_mask=get_patch_mask(pixel_patch_idx, image_size=image_size, patch_size=patch_size)
            pixel_patch_mask = pixel_patch_mask.to(device)
        else:
            pixel_patch_mask=torch.zeros((image_size, image_size), device=device, dtype=dtype)
        
        
        sys_mask   = torch.zeros_like(all_indices, dtype=torch.bool)
        image_mask   = torch.zeros_like(all_indices, dtype=torch.bool)
        text_mask   = torch.zeros_like(all_indices, dtype=torch.bool)
        sys_mask[0:v_token_start]=True
        image_mask[v_token_start:t_token_start]=True
        text_mask[t_token_start:]=True

        
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
        print("q0 vector norm:", q0.norm().item())

        # Compute v = W_K.T @ q0
        v = torch.matmul(W_K.T, q0)
        maxQ_patch=v.unsqueeze(0).repeat(n_patch_num, 1)
        maxQ_least=v.unsqueeze(0).repeat(l_patch_num, 1)
        # noise_patch=torch.randn_like(ben_input_embeds[:,noise_patch_mask], dtype=dtype, device=device)
        # noise_least=torch.randn_like(ben_input_embeds[:,least_patch_mask], dtype=dtype, device=device)

    
    for step in range(adv_steps):
        adv_images=images.clone()
        adv_images=adv_images*(1-pixel_patch_mask)+adv_patch*pixel_patch_mask
        adv_images=adv_images+adv_perturbs*pixel_least_mask
        # save_clip_tensor_as_image(adv_images.cpu(),'1.png')
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
        attn_logits_nc_mean = get_attn_mean(
            attn_logits_nc_mean
        )
           
        loss_list, loss_list_o=[], []
        if l_patch_num>0:
            loss_bpr_l=BPR_loss(
                attn_logits_nc_mean[0], 
                p_list=[ 
                    least_patch_mask, 
                    most_mask,
                ] 
            )
            loss_list.append(torch.relu(loss_bpr_l-bpr_ths)*10)
            # loss_dir_l=torch.norm(torch.sign(noise_least)-adv_input_embeds[:,least_patch_mask])
            # loss_dir_l=torch.norm(torch.sign(noise_least)-hidden_states_list_nc[target_layer_idx][:,least_patch_mask])
            # loss_list.append((loss_dir_l-400))
            # loss_dir_l=F.mse_loss(hidden_states_list_nc[target_layer_idx][:,least_patch_mask], torch.sign(noise_least))
            loss_dir_l=weighted_sign_mse_loss(
                hidden_states_list_nc[target_layer_idx][:,least_patch_mask], 
                maxQ_least, 
                k=5
            )
            loss_list.append((loss_dir_l)*10)
            loss_list_o.append(loss_bpr_l)
            loss_list_o.append(loss_dir_l)
        if n_patch_num>0:
            loss_bpr_p=BPR_loss(
                attn_logits_nc_mean[0], 
                p_list=[
                    noise_patch_mask, 
                    most_mask,
                ] 
            )
            loss_list.append(torch.relu(loss_bpr_p-bpr_ths)*10)
            # loss_dir_p=torch.norm(torch.sign(noise_patch)-adv_input_embeds[:,noise_patch_mask])
            # loss_dir_p=torch.norm(torch.sign(noise_patch)-hidden_states_list_nc[target_layer_idx][:,noise_patch_mask])
            # loss_list.append((loss_dir_p-400))
            loss_dir_p=weighted_sign_mse_loss(
                hidden_states_list_nc[target_layer_idx][:,noise_patch_mask], 
                maxQ_patch, 
                k=5
            )
            loss_dir_p2=F.mse_loss(
                hidden_states_list_nc[target_layer_idx][:,noise_patch_mask], 
                # ben_hidden_states_list_nc[target_layer_idx][:,noise_patch_mask]+5*torch.sign(noise_patch),
                maxQ_patch,
                # dim=-1
            )
            loss_list.append((loss_dir_p)*10)
            loss_list_o.append(loss_bpr_p)
            loss_list_o.append(loss_dir_p)
            loss_list_o.append(loss_dir_p2)
        loss_bpr=BPR_loss(
            attn_logits_nc_mean[0], 
            p_list=[
                most_mask,
                second_mask,
                least_mask, 
            ] 
        )
        loss_list.append(torch.relu(loss_bpr-bpr_ths)*10)
        loss_list_o.append(loss_bpr)
        
        loss = torch.stack(loss_list).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        tar_attn_scores_nc = attn_scores_list_nc[target_layer_idx]
        tar_attn_scores_mean =tar_attn_scores_nc.mean(1)
        tar_attn_scores_mean = get_attn_mean(tar_attn_scores_mean)
        attn_sum   = tar_attn_scores_mean.sum()
        
        sys_attn_ratio  =tar_attn_scores_mean[:,0:v_token_start].sum()/attn_sum
        image_attn_ratio=tar_attn_scores_mean[:,v_token_start:t_token_start].sum()/attn_sum
        text_attn_ratio =tar_attn_scores_mean[:,t_token_start:input_num+1].sum()/attn_sum
        
        adv_perturbs.data=torch.clamp(adv_perturbs.data, -epsilon*image_max, epsilon*image_max)

        tar_most   = tar_attn_scores_mean[:, most_mask]
        tar_second = tar_attn_scores_mean[:, second_mask]
        tar_least   = tar_attn_scores_mean[:, least_mask]
        n_patch  = tar_attn_scores_mean[:, noise_patch_mask]
        l_patch  = tar_attn_scores_mean[:, least_patch_mask]

        tar_most_ratio  =(tar_most.sum()/attn_sum).item()
        tar_second_ratio=(tar_second.sum()/attn_sum).item()
        tar_least_ratio  =(tar_least.sum()/attn_sum).item()
        n_patch_ratio =(n_patch.sum()/attn_sum).item()
        l_patch_ratio =(l_patch.sum()/attn_sum).item()
        
        _, topk_image  = torch.topk(tar_attn_scores_mean[:, v_token_start:t_token_start], k=n_patch_num+l_patch_num)
        topk_image=topk_image+v_token_start

        patch_inter_ratio=intersection_ratio(topk_image[0,:n_patch_num], all_indices[t_token_start-n_patch_num:t_token_start])
        least_inter_ratio=intersection_ratio(topk_image[0,l_patch_num:], topk_v_tar[-l_patch_num:])

        if patch_inter_ratio<0.2:
            print()

        logger.print_log(
            step, 
            'loss: ', round(loss.item(),1), [round(tl.item(),3) for tl in loss_list_o], 
            'inter_ratio', patch_inter_ratio, least_inter_ratio, 
            'attn_ratio:', round(sys_attn_ratio.item(),3), round(image_attn_ratio.item(),3), round(text_attn_ratio.item(),3),
            'n_patch: ', round(n_patch_ratio,3),
            'l_patch: ', round(l_patch_ratio,3),
            'most: ', round(tar_most_ratio,3), 
            'second: ', round(tar_second_ratio,3), 
            'least: ', round(tar_least_ratio,3), 
        )

        # if (loss_bpr1.item()<=bpr_ths):# or (loss_bpr2.item()<=0.9): #  and tar_up_v_map.min()>tar_down_v_map.max()
        #     break
        
        # del loss, loss_cur, attn_logits_nc, attn_map_mean, down_v_logits, up_v_logits
        # del hidden_states_list_nc, attn_scores_list_nc, attn_logits_list_nc
        del attn_logits_list_nc, attn_scores_list_nc
        del hidden_states_list_nc
    
    with torch.no_grad():
        adv_images=images
        adv_images=adv_images*(1-pixel_patch_mask)+adv_patch*pixel_patch_mask
        adv_images=adv_images+adv_perturbs*pixel_least_mask
        # adv_images=adv_images*pixel_patch_mask+adv_patch*(1-pixel_patch_mask)
        # adv_images = torch.zeros_like(images)
        save_clip_tensor_as_image(adv_images.cpu(),'1.png')
        adv_image_embeds = model.encode_images(adv_images)
        new_input_embeds=cat_adv_embed(
            fixed_input_embeds, adv_image_embeds, 
            start_idx=v_token_start, end_idx=t_token_start
        )
        # new_input_embeds[:,noise_patch_mask]=torch.sign(noise_dir)*0.1#+=torch.sign(noise_dir)*0.5
        # new_hidden_states_list_nc, new_attn_map_list_nc, _= trace_transformer_layers_grad(
        #     model,  
        #     new_input_embeds,
        #     attention_mask,
        #     start_layer=0,
        #     end_layer=target_layer_idx,  
        #     return_all=True,
        #     v_token_start=v_token_start,
        #     v_token_num=v_token_num,
        # )
        # new_attn_v_tar_mean =new_attn_map_list_nc[target_layer_idx].mean(1)[0, -1 ,v_token_start:t_token_start]
        # _, topk_v_new  = torch.topk(new_attn_v_tar_mean, k=v_token_num)
        # topk_v_new = topk_v_new+v_token_start
        # noise=torch.randn_like(new_input_embeds[:,most_mask])
        # new_input_embeds[:,most_mask]=new_input_embeds[:,most_mask]+torch.sign(noise)*new_input_embeds.max()*16/255#
    return new_input_embeds, new_input_embeds.shape[1]-input_embeds.shape[1], patch_inter_ratio, least_inter_ratio