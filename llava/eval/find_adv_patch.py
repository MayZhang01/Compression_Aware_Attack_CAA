import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import math
from torch.optim.lr_scheduler import StepLR
from llava.eval.utils import loss_norm, intersection_ratio, add_square_patch, get_patch_mask, save_clip_tensor_as_image


def judge_adv(output_c, output_nc):
    if output_c==output_nc:
        return True
    elif output_c.find(output_nc)>-1:# and len(output_c)<len(output_nc)*2:
        return True
    return False

def trace_transformer_layers_grad(
    model,
    inputs_embeds,
    attention_mask,
    start_layer=0,
    end_layer=None,
    return_all=True,
    tar_layer=None,
    # include_mlp=True,
    v_token_start = None,
    v_token_num = None,
    # t_token_idx = torch.tensor([[0],[27]]),
):
    """
    Differentiable forward pass through selected transformer layers,
    optionally recording hidden states and attention maps at each layer.

    Args:
        model: Transformer model with a .base_model.layers attribute.
        inputs_embeds: Tensor of shape (B, T, D), input embeddings (may include adversarial tokens).
        attention_mask: Tensor of shape (B, T), binary mask for valid tokens (1 = valid, 0 = pad).
        max_layer: (Optional) If specified, forward only up to this layer index (exclusive).
        include_mlp: Whether to include the MLP block and residual after self-attention.
        return_all: If True, return hidden states and attention maps from all layers.

    Returns:
        final_hidden_state: Tensor of shape (B, T, D), output from the last layer.
        hidden_states_list: List of intermediate hidden states (optional).
        attention_maps: List of attention maps (optional), each of shape (B, n_heads, T, T).
    """
    device=inputs_embeds.device
    hidden_states = inputs_embeds
    layers = model.base_model.layers 
    end_layer = len(layers) if end_layer is None else end_layer
    (B, T, D)=inputs_embeds.shape

    hidden_states_list = []
    attn_map_list = []
    attn_logits_list = []
    # value_list = []

    for i in range(start_layer, end_layer+1, 1):
        if i==len(layers):
            break
        layer = layers[i]
        position_ids=torch.arange(T).unsqueeze(0).to(device)

        residual = hidden_states

        hidden_states = layer.input_layernorm(hidden_states)

        if hasattr(layer, "flash_attn"):
            attn=layer.flash_attn
        else:
            attn=layer.self_attn
        output_attentions = False
        past_key_value=None
        output_attentions=True
        # use_cache=False
        # return_logits = True

        bsz, q_len, _ = hidden_states.size()

        query_states = attn.q_proj(hidden_states)
        key_states = attn.k_proj(hidden_states)
        value_states = attn.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        # value_list.append(value_states)

        kv_seq_len = key_states.shape[-2]
        if len(position_ids[0]) == 1:
            position_ids = torch.tensor([[past_key_value.get_usable_length(kv_seq_len, attn.layer_idx)]], dtype=torch.int64).cuda()
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, attn.layer_idx)
        cos, sin = attn.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None :
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, attn.layer_idx, cache_kwargs)

        L, S = query_states.size(-2), key_states.size(-2)
        scale_factor = 1 / math.sqrt(query_states.size(-1))
        attn_bias = torch.zeros(L, S, dtype=query_states.dtype)
        if attn.is_causal:
            assert attention_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias=attn_bias.to(query_states.dtype)

        attn_logits  = query_states @ key_states.transpose(2,3) * scale_factor
        attn_logits += attn_bias.to(query_states.device)
        attn_map = torch.softmax(attn_logits, dim=-1)
        
        if hasattr(layer, "flash_attn"):
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            dropout_rate = attn.attention_dropout if attn.training else 0.0

            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                if torch.is_autocast_enabled():
                    target_dtype = torch.get_autocast_gpu_dtype()
                # Handle the case where the model is quantized
                elif hasattr(attn.config, "_pre_quantization_dtype"):
                    target_dtype = attn.config._pre_quantization_dtype
                else:
                    target_dtype = attn.q_proj.weight.dtype

                query_states = query_states.to(target_dtype)
                key_states = key_states.to(target_dtype)
                value_states = value_states.to(target_dtype)
            if not output_attentions:
                attn_weights = None    
            attn_output = attn._flash_attention_forward(
                query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
            )
        
        else:
            attn_tmp = F.dropout(attn_map, p= 0.0)
            attn_output = torch.matmul(attn_tmp, value_states)

            attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, attn.hidden_size).contiguous()
        hidden_states = attn.o_proj(attn_output)

        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if return_all:
            hidden_states_list.append(hidden_states)
            attn_map_list.append(attn_map)
            attn_logits_list.append(attn_logits)
        if tar_layer is not None and i==tar_layer:
            hidden_states=torch.cat([
                hidden_states[:,:v_token_start,:],
                hidden_states[:,v_token_start:v_token_start+24*3,:],
                hidden_states[:,v_token_start+v_token_num:,:]
            ],dim=1).to(device)
            (B, T, D)=hidden_states.shape

    if return_all:
        return hidden_states_list, attn_map_list, attn_logits_list#, value_list
    else:
        return hidden_states, attn_map, attn_logits

def get_attn_mean(attn_map_mean, start_id=None, end_id=None, range=None):
    # attn_map_mean=attn_scores.mean(1)
    if range is not None:
        attn_tar=attn_map_mean[:,-1:, range]
    elif start_id is not None:
        attn_tar=attn_map_mean[:,-1:, int(start_id):int(end_id)]
    else:
        attn_tar=attn_map_mean[:,-1:, :]
    attn_tar_mean=attn_tar.mean(1)
    return attn_tar_mean

def exclusive_topk_indices(t1, t2, k):
    # Ensure both tensors have the same shape
    assert t1.shape == t2.shape, "t1 and t2 must have the same shape"
    assert t1.shape[0] == 1, "Only supports tensors with shape (1, N)"
    
    # Get the top-k indices for each tensor along dimension 1
    t1_topk_indices = torch.topk(t1, k=k, dim=1).indices[0]  # shape: (k,)
    t2_topk_indices = torch.topk(t2, k=k, dim=1).indices[0]  # shape: (k,)

    # Find indices that are in t1's top-k but not in t2's top-k
    exclusive_indices = [idx.item() for idx in t1_topk_indices if idx not in t2_topk_indices]

    return exclusive_indices

def get_adv_embed(benign_embed, adv_embed, start_idx=None, end_idx=None):
    adv_input_embeds=torch.cat([benign_embed, adv_embed], dim=1)#
    # adv_input_embeds=torch.cat([benign_embed[:, 0:start_idx], adv_embed, benign_embed[:, end_idx:]], dim=1)#
    return adv_input_embeds


def cat_adv_embed(benign_embed, adv_image_embed, adv_text_embed=None, start_idx=None, end_idx=None, adv_patch_num=None):
    if adv_text_embed is not None:
        if adv_patch_num is not None and adv_patch_num>0:
            cat_embedding=[
                benign_embed[:, 0:start_idx], adv_image_embed[:,:-adv_patch_num], adv_text_embed, benign_embed[:, end_idx:],
            ]
        elif adv_patch_num==0:
            cat_embedding=[
                benign_embed[:, 0:start_idx], adv_image_embed[:,:], benign_embed[:, end_idx:],
            ]
        
        else:
            cat_embedding=[
                benign_embed[:, 0:start_idx], adv_image_embed[:,:], adv_text_embed, benign_embed[:, end_idx:],
            ]
    else:
        cat_embedding=[
            benign_embed[:, 0:start_idx], adv_image_embed, benign_embed[:, end_idx:], 
        ]
    adv_input_embeds=torch.cat(cat_embedding, dim=1)
    return adv_input_embeds


def BPR_loss(y_tensor, p_list=None):#p1=None, p2=None, p3=None, p4=None, 
    y_list=[]
    for idx in range(len(p_list)):
        y_list.append(y_tensor[p_list[idx]])
    
    loss_l=[]
    for idx in range(1, len(p_list), 1):
        tmp_diff=y_list[idx-1][:, None] - y_list[idx][None, :]
        tmp_diff = torch.clamp(tmp_diff, min=-20, max=20)
        tmp_loss=-torch.mean(torch.log(torch.sigmoid(tmp_diff)))
        if tmp_loss==torch.inf:
            tmp_loss=-torch.mean(-F.softplus(-tmp_diff)) 
        loss_l.append(tmp_loss.unsqueeze(0))
    return torch.sum(torch.cat(loss_l))

def cos_loss(tensor_a, tensor_b):
    """
    Compute mean cosine distance loss between two tensors.

    Args:
        tensor_a (torch.Tensor): shape (batch_size, feature_dim)
        tensor_b (torch.Tensor): shape (batch_size, feature_dim)

    Returns:
        torch.Tensor: scalar loss value
    """
    # Normalize each vector to unit length
    tensor_a_norm = F.normalize(tensor_a, p=2, dim=-1)
    tensor_b_norm = F.normalize(tensor_b, p=2, dim=-1)

    # Compute cosine similarity: (batch_size,)
    cosine_sim = (tensor_a_norm * tensor_b_norm).sum(dim=-1)

    # Cosine distance = 1 - cosine similarity
    cosine_distance = 1 - cosine_sim

    # Return mean distance as loss
    return cosine_distance.mean()

def weighted_sign_mse_loss(x, y, k=1):
    diff = x - y
    sign_diff = torch.tanh(x*5) - torch.tanh(y*5)
    loss = (diff ** 2) * (sign_diff ** 2)#
    return loss.mean()

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
        
        tar_attn_scores_nc = attn_scores_list_nc[target_layer_idx]
        tar_attn_scores_mean =tar_attn_scores_nc.mean(1)
        tar_attn_scores_mean = get_attn_mean(tar_attn_scores_mean)
        attn_sum   = tar_attn_scores_mean.sum()
        
        sys_attn_ratio  =tar_attn_scores_mean[:,0:v_token_start].sum()/attn_sum
        image_attn_ratio=tar_attn_scores_mean[:,v_token_start:t_token_start].sum()/attn_sum
        text_attn_ratio =tar_attn_scores_mean[:,t_token_start:input_num+1].sum()/attn_sum
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        scheduler.step()

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

        
        # _, topk_image  = torch.topk(tar_attn_scores_mean[:, v_token_start:t_token_start], k=patch_num)
        _, topk_image  = torch.topk(tar_attn_scores_mean[:, v_token_start:t_token_start], k=n_patch_num+l_patch_num)
        topk_image=topk_image+v_token_start

        patch_inter_ratio=intersection_ratio(topk_image[0,:n_patch_num], all_indices[t_token_start-n_patch_num:t_token_start])
        least_inter_ratio=intersection_ratio(topk_image[0,n_patch_num:], topk_v_tar[-l_patch_num:])

        logger.print_log(
            step, 
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
    
    
    # print("v vector norm:", v.norm().item())

    # Optionally normalize v to desired norm
    # norm_target = 5.0
    # y_star = v / v.norm() * norm_target
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