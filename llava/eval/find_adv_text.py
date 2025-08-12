import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import math
from torch.optim.lr_scheduler import StepLR
from llava.eval.utils import loss_norm, intersection_ratio, add_square_patch, get_patch_mask, save_clip_tensor_as_image
from llava.eval.find_adv_util import cat_adv_embed, trace_transformer_layers_grad, get_attn_mean, BPR_loss

def opt_adv_text(
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
    t_token_end = -5

    
    # image_max=images.max()
    
    # if epsilon>0:
    #     adv_perturbs = torch.randn_like(images, dtype=dtype, device=device)
    #     adv_perturbs = adv_perturbs*images.std()
    #     # adv_perturbs = adv_perturbs*epsilon
    #     adv_perturbs = torch.clamp(adv_perturbs, -epsilon*image_max, epsilon*image_max)
    #     # adv_perturbs = torch.zeros_like(images, dtype=dtype, device=device)
    # else:
    #     adv_perturbs = torch.zeros_like(images, dtype=dtype, device=device)
    
    torch.manual_seed(123)                   
    torch.cuda.manual_seed(123)
    fixed_input_embeds = input_embeds.detach()


    flag=True
    adv_num=10
    text_last_id=-2
    patch_inter_ratio, least_inter_ratio=0,0
    if adv_steps>0 and adv_num>0: 
        vocab_embed = model.model.embed_tokens.weight  # [V, D]
        vocab_size, hidden_size = vocab_embed.shape
        adv_logits_list = [torch.randn(vocab_size, requires_grad=True, device="cuda", dtype=dtype) for _ in range(adv_num)]
        # adv_logits_list = [torch.zeros(vocab_size, requires_grad=True, device="cuda", dtype=dtype) for _ in range(adv_num)]
        optimizer = torch.optim.SGD(adv_logits_list, lr=adv_lr)
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
        
        all_indices = torch.arange(ben_attn_score_list_nc[target_layer_idx].size(3)+text_last_id, device=device)

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

            # pixel_least_idx = topk_v_tar[-l_patch_num:]-v_token_start
            # pixel_least_mask=get_patch_mask(pixel_least_idx, image_size=image_size, patch_size=patch_size)
            # pixel_least_mask = pixel_least_mask.to(device)
        # else:
        #     pixel_least_mask=torch.zeros((image_size, image_size), dtype=dtype, device=device)

        if n_patch_num>0:
            noise_patch_mask[t_token_start-n_patch_num:t_token_start]  = True
            most_mask[t_token_start-n_patch_num:t_token_start]  = False
            second_mask[t_token_start-n_patch_num:t_token_start]  = False
            least_mask[t_token_start-n_patch_num:t_token_start]  = False

            # pixel_patch_idx = all_indices[t_token_start-n_patch_num:t_token_start]-v_token_start
            # pixel_patch_mask=get_patch_mask(pixel_patch_idx, image_size=image_size, patch_size=patch_size)
            # pixel_patch_mask = pixel_patch_mask.to(device)
        # else:
        #     pixel_patch_mask=torch.zeros((image_size, image_size), device=device, dtype=dtype)
        
        
        sys_mask   = torch.zeros_like(all_indices, dtype=torch.bool)
        image_mask   = torch.zeros_like(all_indices, dtype=torch.bool)
        text_mask   = torch.zeros_like(all_indices, dtype=torch.bool)
        sys_mask[0:v_token_start]=True
        image_mask[v_token_start:t_token_start]=True
        text_mask[t_token_start:]=True

    tau=0.5
    for i in range(1):
        for step in range(adv_steps):  # 每个位置优化50步

            # 计算当前adv_embed
            adv_dist = torch.softmax(adv_logits_list[i]/tau, dim=-1)  # [V]
            # adv_dist = F.gumbel_softmax(adv_logits_list[i], tau=tau, hard=True)
            adv_embed = adv_dist @ vocab_embed  # [D]

            # 拼接：原始+已经优化好的伪token+当前token
            past_adv_embeds = [
                # (torch.softmax(l/tau, dim=-1) @ vocab_embed).unsqueeze(0) 
                (F.gumbel_softmax(l, tau=tau, hard=True) @ vocab_embed).unsqueeze(0) 
                for l in adv_logits_list#[:i]
            ]
            # adv_input_embeds = torch.cat([ben_input_embeds[0,:text_last_id,:]] + past_adv_embeds + [adv_embed.unsqueeze(0), ben_input_embeds[0, text_last_id:, :] ], dim=0).unsqueeze(0)  # [1, L+n+1, D]
            adv_input_embeds = torch.cat([ben_input_embeds[0,:t_token_end,:]] + past_adv_embeds + [ben_input_embeds[0, t_token_end:, :] ], dim=0).unsqueeze(0)  # [1, L+n+1, D]

            hidden_states_list_nc, attn_scores_list_nc, attn_logits_list_nc= trace_transformer_layers_grad(
                model, adv_input_embeds, attention_mask,
                end_layer=target_layer_idx,  
                return_all=True,
            )

            text_end_id=text_last_id-(adv_input_embeds.shape[1]-ben_input_embeds.shape[1])
            attn_logits_nc = attn_logits_list_nc[target_layer_idx]
            attn_logits_nc_mean = attn_logits_nc.mean(1)
            attn_logits_nc_mean = get_attn_mean(
                attn_logits_nc_mean,
                start_id=0,
                end_id=text_end_id
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
                loss_list_o.append(loss_bpr_l)
                # loss_dir_l=torch.norm(torch.sign(noise_least)-adv_input_embeds[:,least_patch_mask])
                # loss_dir_l=torch.norm(torch.sign(noise_least)-hidden_states_list_nc[target_layer_idx][:,least_patch_mask])
                # loss_list.append((loss_dir_l-400))
                loss_dir_l=F.mse_loss(hidden_states_list_nc[target_layer_idx][:,-1], ben_hidden_states_list_nc[target_layer_idx][:,-1])
                # loss_dir_l=weighted_sign_mse_loss(
                #     hidden_states_list_nc[target_layer_idx][:,least_patch_mask], 
                #     maxQ_least, 
                #     k=5
                # )
                # loss_list.append((loss_dir_l)*10)
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
                loss_list_o.append(loss_bpr_p)
                loss_dir_p=F.mse_loss(hidden_states_list_nc[target_layer_idx][:,-1], ben_hidden_states_list_nc[target_layer_idx][:,-1])
                # loss_dir_p=torch.norm(torch.sign(noise_patch)-adv_input_embeds[:,noise_patch_mask])
                # loss_dir_p=torch.norm(torch.sign(noise_patch)-hidden_states_list_nc[target_layer_idx][:,noise_patch_mask])
                # loss_list.append((loss_dir_p-400))
                # loss_dir_p=weighted_sign_mse_loss(
                #     hidden_states_list_nc[target_layer_idx][:,noise_patch_mask], 
                #     maxQ_patch, 
                #     k=5
                # )
                # loss_dir_p2=F.mse_loss(
                #     hidden_states_list_nc[target_layer_idx][:,noise_patch_mask], 
                #     # ben_hidden_states_list_nc[target_layer_idx][:,noise_patch_mask]+5*torch.sign(noise_patch),
                #     maxQ_patch,
                #     # dim=-1
                # )
                # loss_list.append((1-loss_dir_p)*10)
                loss_list_o.append(loss_dir_p)
                # loss_list_o.append(loss_dir_p2)
            loss_bpr=BPR_loss(
                attn_logits_nc_mean[0], 
                p_list=[
                    most_mask,
                    second_mask,
                    least_mask, 
                ] 
            )
            # loss_list.append(torch.relu(loss_bpr-bpr_ths)*10)
            loss_list_o.append(loss_bpr)
            
            loss = torch.stack(loss_list).sum()
            
            tar_attn_scores_nc = attn_scores_list_nc[target_layer_idx]
            tar_attn_scores_mean =tar_attn_scores_nc.mean(1)
            tar_attn_scores_mean = get_attn_mean(tar_attn_scores_mean, start_id=0, end_id=text_end_id)
            attn_sum   = tar_attn_scores_mean.sum()
            
            sys_attn_ratio  =tar_attn_scores_mean[:,0:v_token_start].sum()/attn_sum
            image_attn_ratio=tar_attn_scores_mean[:,v_token_start:t_token_start].sum()/attn_sum
            text_attn_ratio =tar_attn_scores_mean[:,t_token_start:input_num+1].sum()/attn_sum
            
            optimizer.zero_grad()
            loss.backward()
            # adv_logits_list[i].data=adv_logits_list[i].data+torch.sign(adv_logits_list[i].grad)*0.1
            optimizer.step()
            # scheduler.step()

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

            
            # _, topk_image  = torch.topk(tar_attn_scores_mean[:, v_token_start:t_token_start], k=patch_num)
            _, topk_image  = torch.topk(tar_attn_scores_mean[:, v_token_start:t_token_start], k=n_patch_num+l_patch_num)
            topk_image=topk_image+v_token_start

            patch_inter_ratio=intersection_ratio(topk_image[0,:n_patch_num], all_indices[t_token_start-n_patch_num:t_token_start])
            least_inter_ratio=intersection_ratio(topk_image[0,n_patch_num:], topk_v_tar[-l_patch_num:])

            logger.print_log(
                i, step, 
                'loss: ', round(loss.item(),1), [round(tl.item(),3) for tl in loss_list_o], 
                # round(loss_cos.item(),3), 
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
            # del last_attn_map_mean, last_attn_map_list_nc
            del attn_logits_list_nc
            del hidden_states_list_nc
    
    all_adv_embeds = [
        # (torch.softmax(l/tau, dim=-1) @ vocab_embed).unsqueeze(0) 
        (F.gumbel_softmax(l, tau=tau, hard=True) @ vocab_embed).unsqueeze(0) 
        for l in adv_logits_list
    ]
    new_input_embeds = torch.cat([ben_input_embeds[0,:t_token_start,:]] + all_adv_embeds + [ben_input_embeds[0, t_token_start:, :] ], dim=0).unsqueeze(0)  # [1, L+n+1, D]

    return new_input_embeds, new_input_embeds.shape[1]-input_embeds.shape[1], patch_inter_ratio, least_inter_ratio



def opt_adv_text_gcg(
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
        tokenizer=None,
    ):

    device = input_embeds.device
    dtype=input_embeds.dtype
    _, input_num, embed_dim = input_embeds.shape
    v_token_num = image_shape
    v_token_start = pre_prompt_length_list[0] if len(pre_prompt_length_list) != 0 else 0 
    t_token_start = v_token_start+v_token_num
    t_token_end=-5
    
    torch.manual_seed(123)                   
    torch.cuda.manual_seed(123)
    fixed_input_embeds = input_embeds.detach()

    flag=True
    # adv_num=10
    n = 20  # number of adversarial tokens
    k = 256  # top-k tokens by gradient
    m = 128   # number of candidates per row
    # adv_steps = 30 # Optimization loop
    vocab_embed = model.model.embed_tokens.weight  # [V, D]
    vocab_size, hidden_size = vocab_embed.shape
    adv_token_ids = torch.randint(0, vocab_size, (n,), device=device)
    adv_tokens = F.one_hot(adv_token_ids, num_classes=vocab_size).to(torch.float16)

    patch_inter_ratio, least_inter_ratio=0,0
    if adv_steps>0 and n>0: 
        adv_tokens.requires_grad_()
        # optimizer = torch.optim.SGD(adv_logits_list, lr=adv_lr)
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    
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
        
        all_indices = torch.arange(t_token_start, device=device)

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

        if n_patch_num>0:
            noise_patch_mask[t_token_start-n_patch_num:t_token_start]  = True
            most_mask[t_token_start-n_patch_num:t_token_start]  = False
            second_mask[t_token_start-n_patch_num:t_token_start]  = False
            least_mask[t_token_start-n_patch_num:t_token_start]  = False
        
        sys_mask   = torch.zeros_like(all_indices, dtype=torch.bool)
        image_mask   = torch.zeros_like(all_indices, dtype=torch.bool)
        text_mask   = torch.zeros_like(all_indices, dtype=torch.bool)
        sys_mask[0:v_token_start]=True
        image_mask[v_token_start:t_token_start]=True
        text_mask[t_token_start:]=True

    for step in range(adv_steps):
        # Step 4: adv_embed = adv_tokens @ vocab_embed
        adv_embed = adv_tokens @ vocab_embed  # [n, D]
        # Step 5: Concatenate with benign embedding
        adv_input_embed = torch.cat([
            ben_input_embeds[0,:t_token_end,:], 
            adv_embed, 
            ben_input_embeds[0, t_token_end:, :]
        ], dim=0).unsqueeze(0)  # [1, L+n, D]

        hidden_states_list_nc, attn_scores_list_nc, attn_logits_list_nc= trace_transformer_layers_grad(
            model, adv_input_embed, attention_mask,
            end_layer=target_layer_idx,  
            return_all=True,
        )

        attn_logits_nc = attn_logits_list_nc[target_layer_idx]
        attn_logits_nc_mean = attn_logits_nc.mean(1)
        attn_logits_nc_mean = get_attn_mean(
            attn_logits_nc_mean,
            start_id=0,
            end_id=t_token_start
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
            loss_list_o.append(loss_bpr_l)
            # loss_dir_l=F.mse_loss(hidden_states_list_nc[target_layer_idx][:,-1], ben_hidden_states_list_nc[target_layer_idx][:,-1])
            # loss_list_o.append(loss_dir_l)
        if n_patch_num>0:
            loss_bpr_p=BPR_loss(
                attn_logits_nc_mean[0], 
                p_list=[
                    noise_patch_mask, 
                    most_mask,
                ] 
            )
            loss_list.append(torch.relu(loss_bpr_p-bpr_ths)*10)
            loss_list_o.append(loss_bpr_p)
            loss_dir_p=F.mse_loss(hidden_states_list_nc[target_layer_idx][:,-1], ben_hidden_states_list_nc[target_layer_idx][:,-1])
            # loss_list.append((1-loss_dir_p)*10)
            loss_list_o.append(loss_dir_p)
        loss_bpr=BPR_loss(
            attn_logits_nc_mean[0], 
            p_list=[
                most_mask,
                second_mask,
                least_mask, 
            ] 
        )
        # loss_list.append(torch.relu(loss_bpr-bpr_ths)*10)
        loss_list_o.append(loss_bpr)
        loss = torch.stack(loss_list).sum()
        # Backward to get gradients w.r.t. adv_tokens
        model.zero_grad()
        loss.backward()
        adv_grads = adv_tokens.grad.detach()  # [n, V]

        # Step 7: For each row, select top-k tokens by gradient, sample m candidates
        candi_adv_token_ids = []
        # for i in range(n):
        #     topk = torch.topk(adv_grads[i], k).indices  # [k]
        #     sampled = topk[torch.randint(0, k, (m,))]    # [m]
        #     candi_adv_token_ids.append(sampled)
        # candi_adv_token_ids = torch.stack(candi_adv_token_ids, dim=0).T  # [m, n]
        for _ in range(m):
            # Clone current adversarial token ids
            candidate = adv_token_ids.clone()

            for idx in range(1):
                # Randomly choose one position to modify
                pos = torch.randint(0, n, (1,)).item()

                # Get top-k token ids at this position by gradient
                topk_ids = torch.topk(adv_grads[pos], k).indices

                # Sample one token id from top-k
                new_token_id = topk_ids[torch.randint(0, k, (1,)).item()]

                # Replace the selected position with new token id
                candidate[pos] = new_token_id

            candi_adv_token_ids.append(candidate)
        candi_adv_token_ids = torch.stack(candi_adv_token_ids, dim=0)  # [m, n]

        # Step 8: Convert candi_adv_token_ids to one-hot tensor of shape (m, n, V)
        candi_adv_tokens = F.one_hot(candi_adv_token_ids, num_classes=vocab_size).to(torch.float16)  # [m, n, V]

        # Step 9: Evaluate each candidate and compute loss
        best_loss = float("inf")
        best_tokens = None
        
        for j in range(m):
            this_adv_embed = candi_adv_tokens[j] @ vocab_embed  # [n, D]

            this_input_embed = torch.cat([
                ben_input_embeds[0,:t_token_end,:], 
                this_adv_embed, 
                ben_input_embeds[0, t_token_end:, :]
            ], dim=0).unsqueeze(0)

            
            this_hidden_states_list_nc, this_attn_scores_list_nc, this_attn_logits_list_nc= trace_transformer_layers_grad(
                model, this_input_embed, attention_mask,
                end_layer=target_layer_idx,  
                return_all=True,
            )

            this_attn_logits_nc = this_attn_logits_list_nc[target_layer_idx]
            this_attn_logits_nc_mean = this_attn_logits_nc.mean(1)
            this_attn_logits_nc_mean = get_attn_mean(
                this_attn_logits_nc_mean,
                start_id=0,
                end_id=t_token_start
            )
                
            if l_patch_num>0:
                this_loss_bpr=BPR_loss(
                    this_attn_logits_nc_mean[0], 
                    p_list=[ 
                        least_patch_mask, 
                        most_mask,
                    ] 
                )
            if n_patch_num>0:
                this_loss_bpr=BPR_loss(
                    this_attn_logits_nc_mean[0], 
                    p_list=[
                        noise_patch_mask, 
                        most_mask,
                    ] 
                )
                # this_loss_bpr=1-F.mse_loss(this_hidden_states_list_nc[target_layer_idx][:,-1], ben_hidden_states_list_nc[target_layer_idx][:,-1])
                
            if this_loss_bpr.item() < best_loss:
                best_loss = this_loss_bpr.item()
                best_tokens = candi_adv_tokens[j]
                best_token_ids = candi_adv_token_ids[j]
                best_text = tokenizer.decode(best_token_ids, skip_special_tokens=True)
                
                best_attn_scores_nc = this_attn_scores_list_nc[target_layer_idx]

        # Step 10: Update adv_tokens with best found
        adv_token_ids = best_token_ids.detach().clone()
        adv_tokens = best_tokens.detach().clone()
        adv_tokens.requires_grad_()

        best_attn_scores_mean = best_attn_scores_nc.mean(1)
        best_attn_scores_mean = get_attn_mean(best_attn_scores_mean, start_id=0, end_id=t_token_start)
        # attn_sum   = best_attn_scores_mean.sum() 
            
        # _, topk_image  = torch.topk(tar_attn_scores_mean[:, v_token_start:t_token_start], k=patch_num)
        _, topk_image  = torch.topk(best_attn_scores_mean[:, v_token_start:t_token_start], k=n_patch_num+l_patch_num)
        topk_image=topk_image+v_token_start

        patch_inter_ratio=intersection_ratio(topk_image[0,:n_patch_num], all_indices[t_token_start-n_patch_num:t_token_start])
        least_inter_ratio=intersection_ratio(topk_image[0,l_patch_num:], topk_v_tar[-l_patch_num:])
        
        logger.print_log(
            step, 
            'loss: ', round(loss.item(),1), [round(tl.item(),3) for tl in loss_list_o], 
            # round(loss_cos.item(),3), 
            'best_loss: ', round(best_loss,3),
            'inter_ratio', patch_inter_ratio, least_inter_ratio, 
            'best_text:',best_text.replace('\n',''),
            # 'attn_ratio:', round(sys_attn_ratio.item(),3), round(image_attn_ratio.item(),3), round(text_attn_ratio.item(),3),
            # 'n_patch: ', round(n_patch_ratio,3),
            # 'l_patch: ', round(l_patch_ratio,3),
            # 'most: ', round(tar_most_ratio,3), 
            # 'second: ', round(tar_second_ratio,3), 
            # 'least: ', round(tar_least_ratio,3), 
        )

        del attn_logits_list_nc
        del hidden_states_list_nc
    
    
    # Final optimized adversarial token one-hot matrix
    adv_tokens_final = adv_tokens
    adv_token_ids_final = adv_tokens.argmax(dim=-1)
    adv_embed_final = adv_tokens_final @ vocab_embed  # [n, D]
    # Step 5: Concatenate with benign embedding
    new_input_embeds = torch.cat([
        ben_input_embeds[0,:t_token_end,:], 
        adv_embed_final, 
        ben_input_embeds[0, t_token_end:, :]
    ], dim=0).unsqueeze(0)

    return new_input_embeds, new_input_embeds.shape[1]-input_embeds.shape[1], patch_inter_ratio, least_inter_ratio