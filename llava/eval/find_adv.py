import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import math
from torch.optim.lr_scheduler import StepLR
from llava.eval.utils import loss_norm


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

def intersection_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute the intersection ratio between two integer tensors after sorting.
    Intersection is defined as the number of common elements (set-wise), 
    divided by the total number of unique elements (from either tensor).

    Args:
        a (torch.Tensor): First tensor, must be 1D or flattenable
        b (torch.Tensor): Second tensor, same shape as a

    Returns:
        float: Ratio of intersection count to total number of elements in a
    """
    # Flatten both tensors to 1D
    # a = a.flatten()
    # b = b.flatten()
    
    # # Sort both tensors
    # a_sorted, _ = torch.sort(a)
    # b_sorted, _ = torch.sort(b)

    # Compute set-wise intersection
    intersect = torch.isin(a, b)
    num_match = intersect.sum().item()
    total = min(a.numel(),b.numel())

    # Compute ratio
    ratio = num_match / total
    return ratio

def get_adv_embed(benign_embed, adv_embed, start_idx=None, end_idx=None):
    adv_input_embeds=torch.cat([benign_embed, adv_embed], dim=1)#
    # adv_input_embeds=torch.cat([benign_embed[:, 0:start_idx], adv_embed, benign_embed[:, end_idx:]], dim=1)#
    return adv_input_embeds

def opt_adv_pro(
        model, input_embeds, position_ids, attention_mask,
        image_shape,        
        token_length_list,  
        pre_prompt_length_list, 
        image=None,  
        target_layer_idx=1, 
        adv_steps=20, 
        adv_lr=1e-1,
        stop_ths=[],
        alpha=1,
    ):

    device = input_embeds.device
    embed_dim = input_embeds.shape[-1]
    v_token_num = image_shape
    v_token_start = pre_prompt_length_list[0] if len(pre_prompt_length_list) != 0 else 0 
    text_token_start = v_token_start + image_shape 

    ben_layer=2
    adv_num=1
    
    adv_token = input_embeds[:,-adv_num:].detach().clone()
    ori_adv_token = adv_token.detach().clone()
    torch.manual_seed(123)                   
    torch.cuda.manual_seed(123)

    fixed_input_embeds = input_embeds.detach()#[:,:-1]
    if adv_steps>0: 
        adv_token.requires_grad_(True)
        optimizer = torch.optim.SGD([adv_token], lr=adv_lr) #, img_token1
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    with torch.no_grad():
        
        _, ben_attn_map_list_nc, _= trace_transformer_layers_grad(
            model, 
            get_adv_embed(
                fixed_input_embeds, adv_token, 
                start_idx=v_token_start, end_idx=text_token_start
            ),
            attention_mask,
            start_layer=0,
            end_layer=None,  
            return_all=True,
            v_token_start=v_token_start,
            v_token_num=v_token_num,
        )
        ben_attn_v_mean=ben_attn_map_list_nc[target_layer_idx].mean(1)[0,-1,v_token_start:v_token_start+v_token_num]
        _, topk_v_idx = torch.topk(ben_attn_v_mean, k=v_token_num)
        ben_attn_v_last_mean=ben_attn_map_list_nc[-1].mean(1)[0,-1,v_token_start:v_token_start+v_token_num]
        _, topk_v_last = torch.topk(ben_attn_v_last_mean, k=v_token_num)
        i_r =intersection_ratio(topk_v_last[:5], topk_v_idx[:50], )
        i_r1=intersection_ratio(topk_v_last[:10],topk_v_idx[:50], )
        i_r2=intersection_ratio(topk_v_last[:50],topk_v_idx[:50], )
        i_r3=intersection_ratio(topk_v_last[:50],topk_v_idx[:100],)
        print('ratio', i_r, i_r1, i_r2, i_r3)
        topk_v_last = topk_v_last+v_token_start
        all_indices = torch.arange(ben_attn_map_list_nc[target_layer_idx].size(3))

        up_mask_v   = torch.zeros_like(all_indices, dtype=torch.bool)
        down_mask_v = torch.zeros_like(all_indices, dtype=torch.bool)
        up_mask_v[topk_v_last[50:100]]  = True
        down_mask_v[topk_v_last[20:50]] = True

        last_attn_map_nc = ben_attn_map_list_nc[-1]
        last_attn_map_mean=last_attn_map_nc.mean(1)
        last_down_v_map = get_attn_mean(
            last_attn_map_mean, 
            range=down_mask_v
        )
        last_up_v_map = get_attn_mean(
            last_attn_map_mean, 
            range=up_mask_v
        )
        print(' ',
            round(last_up_v_map.max().item(),5), round(last_up_v_map.min().item(),5),
            round(last_down_v_map.max().item(),5), round(last_down_v_map.min().item(),5)
        )
    
    flag=False
    for step in range(adv_steps):
        adv_input_embeds=get_adv_embed(
            fixed_input_embeds, adv_token, 
            start_idx=v_token_start, end_idx=text_token_start
        )
        
        hidden_states_list_nc, attn_map_list_nc, attn_logits_list_nc= trace_transformer_layers_grad(
            model, adv_input_embeds, attention_mask,
            end_layer=target_layer_idx,  
            return_all=True,
        )
        
        attn_logits_nc = attn_logits_list_nc[target_layer_idx]
        attn_map_mean=attn_logits_nc.mean(1)
        down_v_logits = get_attn_mean(
            attn_map_mean, 
            range=down_mask_v
        )
        up_v_logits1 = get_attn_mean(
            attn_map_mean, 
            range=up_mask_v
        )
        
        loss_cur=(
            -up_v_logits1.mean()*1.3
            +down_v_logits.mean()
        )
        
        loss = alpha*loss_cur
        
        optimizer.zero_grad()
        loss.backward()
        
        print(
            step, round(loss_cur.item(),3), 
        )
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            _, last_attn_map_list_nc, _= trace_transformer_layers_grad(
                model, hidden_states_list_nc[target_layer_idx], attention_mask,
                start_layer=target_layer_idx+1,  
                return_all=True,
            )
            last_attn_map_nc = last_attn_map_list_nc[-1]
            last_attn_map_mean=last_attn_map_nc.mean(1)
            last_down_v_map = get_attn_mean(
                last_attn_map_mean, 
                range=down_mask_v
            )
            last_up_v_map = get_attn_mean(
                last_attn_map_mean, 
                range=up_mask_v
            )
            print(' ',
                round(last_up_v_map.mean().item(),5), round(last_down_v_map.mean().item(),5),
                round(last_up_v_map.max().item(),5), round(last_up_v_map.min().item(),5),
                round(last_down_v_map.max().item(),5), round(last_down_v_map.min().item(),5)
            )
        
        if (
            last_up_v_map.max().item()>last_down_v_map.max().item()*stop_ths[0]
        ) and (
            last_up_v_map.min().item()>last_down_v_map.min().item()*stop_ths[1]
        ) and (
            last_up_v_map.max().item()>stop_ths[2]
        ):
            flag=True
            break
        
        del loss, loss_cur, attn_logits_nc,attn_map_mean,down_v_logits
        del last_attn_map_mean
        del attn_logits_list_nc
        del last_attn_map_list_nc, hidden_states_list_nc

    new_input_embeds=get_adv_embed(
        fixed_input_embeds, adv_token, 
        start_idx=v_token_start, end_idx=text_token_start
    )
    return new_input_embeds, new_input_embeds.shape[1]-input_embeds.shape[1], flag



def cat_adv_embed(benign_embed, adv_image_embed, adv_text_embed=None, start_idx=None, end_idx=None):
    # cat_embedding=[
    #     benign_embed[:, 0:start_idx], adv_image_embed, benign_embed[:, end_idx:],#benign_embed[:, -1:]
    # ]
    if adv_text_embed is not None:
        cat_embedding=[
            benign_embed[:, 0:start_idx], adv_image_embed[:,:-1], adv_text_embed, benign_embed[:, end_idx:],#benign_embed[:, -1:]
        ]
    else:
        cat_embedding=[
            benign_embed[:, 0:start_idx], adv_image_embed, benign_embed[:, end_idx:], #benign_embed[:, -1:]
        ]
    adv_input_embeds=torch.cat(cat_embedding, dim=1)
    return adv_input_embeds

def intersect_topk_range(t1, t2, k1, k2):
    """
    Compute the intersection of top-k1 to top-k2 indices between two tensors.

    Args:
        t1 (Tensor): A tensor of shape (1, 576).
        t2 (Tensor): A tensor of shape (1, 576).
        k1 (int): Start of the top-k range (inclusive).
        k2 (int): End of the top-k range (inclusive).

    Returns:
        Tensor: A 1D tensor containing the indices that are common in the top-k1 to top-k2 
                range of both t1 and t2.
    """
    # Validate shapes
    assert t1.shape == t2.shape == (1, 576), "Both tensors must have shape (1, 576)"
    assert 0 <= k1 <= k2 <= t1.shape[1], "k1 and k2 must define a valid range"

    # Get top-k2 indices for each tensor (returns sorted in descending order of values)
    t1_topk = torch.topk(t1, k=k2, dim=1).indices[0]
    t2_topk = torch.topk(t2, k=k2, dim=1).indices[0]

    # Slice from k1 to k2 to get the desired rank range
    t1_range = t1_topk[k1:k2]
    t2_range = t2_topk[k1:k2]

    # Compute the intersection of the index sets and convert to a tensor
    intersection = torch.tensor(
        list(set(t1_range.tolist()) & set(t2_range.tolist())),
        dtype=torch.long
    )

    return intersection

def opt_adv_image_token(
        model, input_embeds, position_ids, attention_mask,
        image_shape,        
        token_length_list,  
        pre_prompt_length_list, 
        input_ids=None,
        images=None,  
        tokenizer=None,
        t_token=None,
        target_layer_idx=1, 
        adv_steps=20, 
        adv_lr=1e-1,
        stop_ths=[],
        alpha=1,
        logger=None,
        epsilon=8/255,
    ):

    device = input_embeds.device
    dtype=input_embeds.dtype
    _, input_num, embed_dim = input_embeds.shape
    v_token_num = image_shape
    v_token_start = pre_prompt_length_list[0] if len(pre_prompt_length_list) != 0 else 0 
    t_token_start=v_token_start+v_token_num
    text_token_start = v_token_start + image_shape 

    ben_layer=2
    adv_num=1
    
    image_max=images.max()

    logger.print_log('epsilon', epsilon)
    
    adv_perturbs = torch.randn_like(images, dtype=dtype, device=device)
    adv_perturbs = adv_perturbs*images.std()
    adv_perturbs = adv_perturbs*epsilon
    
    # adv_perturbs = torch.zeros_like(images, dtype=dtype, device=device)
    torch.manual_seed(123)                   
    torch.cuda.manual_seed(123)
    # adv_token.data = adv_token.data+torch.randn(
    #     adv_token.shape, dtype=adv_token.dtype
    # ).to(adv_token.device)*0.01
    fixed_input_embeds = input_embeds.detach()#[:,:-1]
    # adv_perturbs[:,-1:]=fixed_input_embeds[:,-1:]

    all_token_embedding = model.get_input_embeddings().weight
    vocab_size, emb_dim=all_token_embedding.shape
    # adv_token_logit = torch.randn((adv_num, vocab_size), device=device, dtype=dtype)
    # adv_token_logit = (adv_token_logit-0.5)*2
    # adv_token_logit[:, 29901]=adv_token_logit.max()+10
    # ini_token_embeds=fixed_input_embeds[:, t_token_start:t_token_start+adv_num].detach().clone()
    ini_token_embeds=fixed_input_embeds[:, -adv_num:].detach().clone()
    adv_token_embeds=ini_token_embeds.detach().clone()
    adv_token_embeds=torch.randn_like(ini_token_embeds, device=device)*ini_token_embeds.std()

    flag=True
    if adv_steps>0: 
        adv_perturbs.requires_grad_(True)
        # adv_token_logit.requires_grad_(True)
        adv_token_embeds.requires_grad_(True)
        optimizer = torch.optim.SGD([adv_perturbs, adv_token_embeds], lr=adv_lr)#
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
        # flag=False
    
    with torch.no_grad():    
        ben_image_embeds=model.encode_images(images)#+adv_perturbs
        ben_input_embeds=cat_adv_embed(
            fixed_input_embeds, ben_image_embeds, 
            # ini_token_embeds,
            start_idx=v_token_start, end_idx=text_token_start
        )
        ben_hidden_states_list_nc, ben_attn_map_list_nc, _= trace_transformer_layers_grad(
            model,  
            ben_input_embeds,
            attention_mask,
            start_layer=0,
            end_layer=None,  
            return_all=True,
            v_token_start=v_token_start,
            v_token_num=v_token_num,
        )
        ben_attn_v_tar_mean =ben_attn_map_list_nc[target_layer_idx].mean(1)[0, -1 ,v_token_start:t_token_start]#
        ben_attn_v_last_mean=ben_attn_map_list_nc[-1].mean(1)[0,-1,v_token_start:t_token_start]#
        _, topk_v_last = torch.topk(ben_attn_v_last_mean, k=v_token_num)
        _, topk_v_tar  = torch.topk(ben_attn_v_tar_mean, k=v_token_num)
        
        ben_last_logits = model.lm_head(ben_hidden_states_list_nc[-1])
        ben_last_logits = ben_last_logits[:,-1,:]
        ben_tar_logits = model.lm_head(ben_hidden_states_list_nc[target_layer_idx])
        ben_tar_logits = ben_tar_logits[:,-1,:]
        # _, topk_l_last = torch.topk(ben_last_logits, k=10)
        # logger.print_log(tokenizer.decode(topk_l_last[0]))
        # i_r =intersection_ratio(topk_v_last[:5],  topk_v_tar[:50], )
        # i_r1=intersection_ratio(topk_v_last[:10], topk_v_tar[:50], )
        # i_r2=intersection_ratio(topk_v_last[:50], topk_v_tar[:50], )
        # i_r3=intersection_ratio(topk_v_last[:50], topk_v_tar[:100],)
        # logger.print_log('ratio', i_r, i_r1, i_r2, i_r3)

        topk_v_last = topk_v_last+v_token_start
        topk_v_tar  = topk_v_tar+v_token_start
        inter_down=torch.tensor(
            list(set(topk_v_last[0:100].tolist()) & set(topk_v_tar[0:100].tolist())),
            dtype=torch.long
        )
        inter_up=torch.tensor(
            list(set(topk_v_last[200:500].tolist()) & set(topk_v_tar[200:500].tolist())),
            dtype=torch.long
        )
        
        all_indices = torch.arange(ben_attn_map_list_nc[target_layer_idx].size(3))

        up_mask_v   = torch.zeros_like(all_indices, dtype=torch.bool)
        down_mask_v = torch.zeros_like(all_indices, dtype=torch.bool)
        # down_mask_l = torch.zeros(vocab_size, dtype=torch.bool)
        # up_mask_l   = torch.zeros(vocab_size, dtype=torch.bool)
        up_mask_v[topk_v_tar[-100:]]  = True
        down_mask_v[topk_v_tar[:100]] = True
        # up_mask_v[topk_v_tar[50:100]]  = True
        # down_mask_v[topk_v_tar[20:50]] = True
        # up_mask_v[inter_up]  = True
        # down_mask_v[inter_down] = True
        # up_mask_v[t_token_start:t_token_start+adv_num]  = True
        # up_mask_v[topk_v_last[50:100]]  = True
        # down_mask_v[topk_v_last[20:50]] = True
        # down_mask_v[-1] = True
        # down_mask_l[topk_l_last[:,0]] = True
        # down_mask_l[3869]=True
        # down_mask_l[1939]=True
        # up_mask_l[topk_l_last[:,1:]] = True
        # up_mask_l[2] = True
        # up_mask_v[inter_down] = True
        # down_mask_v[inter_up] = True        

        last_attn_map_nc = ben_attn_map_list_nc[-1]
        last_attn_map_mean=last_attn_map_nc.mean(1)
        last_down_v_map = get_attn_mean(
            last_attn_map_mean, 
            range=down_mask_v
        )
        last_up_v_map = get_attn_mean(
            last_attn_map_mean, 
            range=up_mask_v
        )
        logger.print_log('ben: ',
            round(last_up_v_map.max().item(),5), round(last_up_v_map.min().item(),5),
            round(last_down_v_map.max().item(),5), round(last_down_v_map.min().item(),5)
        )
    n_ben_hi=ben_hidden_states_list_nc[target_layer_idx][:,up_mask_v]
    # n_ben_hi=n_ben_hi+torch.randn_like(n_ben_hi)*n_ben_hi.std()*64/255
    for step in range(adv_steps):
        # adv_token_distribution = F.softmax(adv_token_logit, dim=1)
        # adv_token_embeds = torch.matmul(adv_token_distribution, all_token_embedding).unsqueeze(0)#.unsqueeze(0)
        adv_images=images+adv_perturbs#
        adv_image_embeds = model.encode_images(adv_images)
        adv_input_embeds = cat_adv_embed(
            fixed_input_embeds, adv_image_embeds, 
            # adv_token_embeds, 
            start_idx=v_token_start, end_idx=text_token_start
        )
        
        hidden_states_list_nc, attn_map_list_nc, attn_logits_list_nc= trace_transformer_layers_grad(
            model, adv_input_embeds, attention_mask,
            end_layer=target_layer_idx,  
            return_all=True,
        )
        logits = model.lm_head(hidden_states_list_nc[-1])
        logits = logits[:,-1,:]
        hi_loss=loss_norm(
            ground_logits=n_ben_hi, 
            pred_logits=hidden_states_list_nc[target_layer_idx][:,up_mask_v]
        )
        # cf_s=torch.softmax(logits, dim=-1)
        attn_logits_nc = attn_logits_list_nc[target_layer_idx]
        attn_map_mean=attn_logits_nc.mean(1)
        down_v_logits = get_attn_mean(
            attn_map_mean, 
            range=down_mask_v
        )
        up_v_logits = get_attn_mean(
            attn_map_mean, 
            range=up_mask_v
        )
        all_image_logits = get_attn_mean(
            attn_map_mean, 
            start_id=v_token_start,#input_num,#
            end_id=t_token_start,#t_token_start,#input_num,#+adv_num
        )
        up_mean=up_v_logits.mean()
        down_mean=down_v_logits.mean()
        # logits_up_mean=logits[:,up_mask_l].mean()
        # logits_down_mean=logits[:,down_mask_l].mean()
        loss_cur=(
            -up_mean*1.3
            -all_image_logits.mean()*10
            +down_mean
            # +torch.norm(adv_image_embeds[:,-1:]-ini_token_embeds)
            # -logits_up_mean*1
            # +hi_loss*0.1
            # +logits_down_mean*2
        )
        
        loss = alpha*loss_cur
        
        optimizer.zero_grad()
        loss.backward()
        
        # adv_perturbs.grad=torch.clamp(adv_perturbs.grad, -epsilon*image_max*0.5, epsilon*image_max*0.5)
        optimizer.step()
        scheduler.step()
        
        adv_perturbs.data=torch.clamp(adv_perturbs.data, -epsilon*image_max, epsilon*image_max)

        with torch.no_grad():
            last_hidden_states_list_nc, last_attn_map_list_nc, _= trace_transformer_layers_grad(
                model, hidden_states_list_nc[target_layer_idx], attention_mask,
                start_layer=target_layer_idx+1,  
                return_all=True,
            )
            last_attn_map_nc = last_attn_map_list_nc[-1]
            last_attn_map_mean=last_attn_map_nc.mean(1)
            last_down_v_map = get_attn_mean(
                last_attn_map_mean, 
                range=down_mask_v
            )
            last_up_v_map = get_attn_mean(
                last_attn_map_mean, 
                range=up_mask_v
            )
            
            if target_layer_idx>=2:
                tar_attn_map_nc = attn_map_list_nc[target_layer_idx]
            else:
                tar_attn_map_nc = last_attn_map_list_nc[1-target_layer_idx]
            tar_attn_map_mean=tar_attn_map_nc.mean(1)
            tar_down_v_map = get_attn_mean(
                tar_attn_map_mean, 
                range=down_mask_v
            )
            tar_up_v_map = get_attn_mean(
                tar_attn_map_mean, 
                range=up_mask_v
            )
            attn_sum=get_attn_mean(tar_attn_map_mean).sum()
            sys_attn_ratio=get_attn_mean(
                tar_attn_map_mean, 
                start_id=0,
                end_id=v_token_start
            ).sum()/attn_sum

            image_attn_ratio=get_attn_mean(
                tar_attn_map_mean, 
                start_id=v_token_start,
                end_id=t_token_start
            ).sum()/attn_sum

            text_attn_ratio=get_attn_mean(
                tar_attn_map_mean, 
                start_id=t_token_start,
                end_id=input_num+1
            ).sum()/attn_sum
            
            # h_logits = model.lm_head(hidden_states_list_nc[-1])
            # h_logits = h_logits[:,-1,:]
            # _, h_top_l_last  = torch.topk(h_logits, k=15)
            # logger.print_log(tokenizer.decode(h_top_l_last[0]))
            # t_logits = model.lm_head(last_hidden_states_list_nc[-1])
            # t_logits = t_logits[:,-1,:]
            # _, top_l_last  = torch.topk(t_logits, k=15)
            # logger.print_log(tokenizer.decode(top_l_last[0]))
        tar_up_ratio=(tar_up_v_map.sum()/attn_sum).item()
        tar_down_ratio=(tar_down_v_map.sum()/attn_sum).item()
        logger.print_log(
            step, 
            'loss: ',
            round(loss_cur.item(),3),  round(up_mean.item(),3),  round(down_mean.item(),3),  round(all_image_logits.mean().item(),3),   round(hi_loss.item(),3),
            # round(logits_up_mean.item(),5), round(logits_down_mean.item(),5), round(ce_loss.item(),5),
            'attn_ratio:', round(sys_attn_ratio.item(),3), round(image_attn_ratio.item(),3), round(text_attn_ratio.item(),3),
            'tar_up: ', round(tar_up_ratio,3),round(tar_up_v_map.max().item(),5),  
            # round(tar_up_v_map.mean().item(),5),
            'tar_down: ', round(tar_down_ratio,3), round(tar_down_v_map.max().item(),5),
            # round(tar_down_v_map.mean().item(),5),
            'last_up: ', round((last_up_v_map.sum()/attn_sum).item(),3),
            # round(last_up_v_map.max().item(),5),  round(last_up_v_map.mean().item(),5),
            'last_down: ', round((last_down_v_map.sum()/attn_sum).item(),3),
            # round(last_down_v_map.max().item(),5),round(last_down_v_map.mean().item(),5),
        )
        
        # if (
        #     last_up_v_map.max().item()>last_down_v_map.max().item()*stop_ths[0]
        # ) and (
        #     last_up_v_map.min().item()>last_down_v_map.min().item()*stop_ths[1]
        # ) and (
        #     last_up_v_map.max().item()>stop_ths[2]
        # ):
        #     flag=True
        #     break
        # if tar_up_ratio>0.07 and tar_down_ratio>0.04:
        #     break
        # if image_attn_ratio.item()>0.12:
        #     flag=True
        #     # if image_attn_ratio.item()>0.3:
        #     #     adv_perturbs=adv_perturbs*0.5
        #     break
        
        # del loss, loss_cur, attn_logits_nc, attn_map_mean, down_v_logits, up_v_logits
        del last_attn_map_mean
        del attn_logits_list_nc
        del last_attn_map_list_nc, hidden_states_list_nc

    # adv_token.data = adv_token.data+torch.randn(
    #     adv_token.shape, dtype=adv_token.dtype
    # ).to(adv_token.device)*0.001
    with torch.no_grad():
        # adv_token_distribution = F.softmax(adv_token_logit, dim=1)
        # adv_token_embeds = torch.matmul(adv_token_distribution, all_token_embedding).unsqueeze(0)#.unsqueeze(0)
        adv_images=images+adv_perturbs#+torch.randn_like(adv_perturbs)*images.std()*8/255
        adv_image_embeds = model.encode_images(adv_images)
        new_input_embeds=cat_adv_embed(
            fixed_input_embeds, adv_image_embeds, 
            # adv_token_embeds, 
            start_idx=v_token_start, end_idx=text_token_start
        )
    return new_input_embeds, new_input_embeds.shape[1]-input_embeds.shape[1], flag