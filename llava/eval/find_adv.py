import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import math
from torch.optim.lr_scheduler import StepLR

def trace_transformer_layers_grad(
    model,
    inputs_embeds,
    attention_mask,
    max_layer=None,
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
    num_layers = len(layers) if max_layer is None else max_layer
    (B, T, D)=inputs_embeds.shape

    hidden_states_list = []
    attn_map_list = []
    attn_logits_list = []
    value_list = []

    for i in range(num_layers+1):
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
        value_list.append(value_states)

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
            # attn_output = torch.nn.functional.scaled_dot_product_attention(
            #     query_states,
            #     key_states,
            #     value_states,
            #     attn_mask=attention_mask,
            #     dropout_p=attn.attention_dropout if attn.training else 0.0,
            #     # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            #     is_causal=attn.is_causal and attention_mask is None and q_len > 1,
            # )

            # scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(128)

            # # 假设 attention_mask 是 [B, 1, Q, K] 维度的 float mask，值为 0/-inf 或类似的 bias
            # if attention_mask is not None:
            #     scores = scores + attention_mask  # additive mask，加法而不是 masked_fill

            # if attn.is_causal and attention_mask is None and q_len > 1:
            #     causal_mask = torch.tril(torch.ones(scores.shape[-2], scores.shape[-1], device=scores.device))
            #     causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1,1,Q,K]
            #     scores = scores.masked_fill(causal_mask == 0, float("-inf"))

            # attn = torch.softmax(scores, dim=-1)
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
        return hidden_states_list, attn_map_list, attn_logits_list, value_list
    else:
        return hidden_states, attn_map, attn_logits

def get_attn_mean(attn_scores, start_id=None, end_id=None, range=None):
    attn_map_mean=attn_scores.mean(1)
    if range is not None:
        attn_tar=attn_map_mean[:,-1:, range]
    else:
        attn_tar=attn_map_mean[:,-1:, int(start_id):int(end_id)]
    attn_tar_mean=attn_tar.mean(1)
    return attn_tar_mean

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

def opt_adv_pro(
        model, input_embeds, position_ids, attention_mask,
        image_shape,        
        token_length_list,  
        pre_prompt_length_list,   
        target_layer_idx=1, 
        adv_steps=20, 
        adv_lr=1e-1
    ):

    device = input_embeds.device
    embed_dim = input_embeds.shape[-1]
    v_token_num = image_shape
    v_token_start = pre_prompt_length_list[0] if len(pre_prompt_length_list) != 0 else 0 
    text_token_start = v_token_start + image_shape 

    ben_layer=2
    adv_num=1
    
    adv_token = input_embeds[:,-adv_num:].detach().clone()
    torch.manual_seed(123)                   
    torch.cuda.manual_seed(123)
    # adv_token.data = adv_token.data+torch.randn(
    #     adv_token.shape, dtype=adv_token.dtype
    # ).to(adv_token.device)*0.01
    ori_adv_token = adv_token.detach().clone()
    # img_token1 = input_embeds[0,v_token_start:v_token_start+24*6].detach().clone().unsqueeze(0)
    # img_token2 = input_embeds[0,v_token_start+24*6: v_token_start+v_token_num].detach().clone().unsqueeze(0)
    fixed_input_embeds = input_embeds.detach()#[:,:-1]
    if adv_steps>0: 
        adv_token.requires_grad_(True)
        optimizer = torch.optim.SGD([adv_token], lr=adv_lr) #, img_token1
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        with torch.no_grad():
            # ben_hidden_states_list_c, ben_attn_map_list_c, ben_attn_logits_list_c, ben_value_list_c =trace_transformer_layers_grad(
            #     model, #input_embeds, 
            #     torch.cat([fixed_input_embeds, input_embeds[:,-adv_num:]], dim=1), 
            #     attention_mask,
            #     max_layer=target_layer_idx+ben_layer,  
            #     return_all=True,
            #     v_token_start=v_token_start,
            #     v_token_num=v_token_num,
            #     tar_layer=target_layer_idx,
            #     # t_token_idx=torch.tensor([[0],[token_length_list[0]-v_token_num-v_token_start]]),
            # )
            ben_hidden_states_list_nc, ben_attn_map_list_nc, ben_attn_logits_list_nc, ben_value_list_nc= trace_transformer_layers_grad(
                model, #input_embeds,
                torch.cat([fixed_input_embeds, input_embeds[:,-adv_num:]], dim=1), 
                attention_mask,
                max_layer=None,  
                return_all=True,
                v_token_start=v_token_start,
                v_token_num=v_token_num,
                # t_token_idx=torch.tensor([[0],[token_length_list[0]-v_token_num-v_token_start]]),
            )
            ben_attn_v_mean=ben_attn_map_list_nc[target_layer_idx].mean(1)[0,-1,v_token_start:v_token_start+v_token_num]
            _, topk_v_idx = torch.topk(ben_attn_v_mean, k=v_token_num)
            ben_attn_v_last_mean=ben_attn_map_list_nc[-1].mean(1)[0,-1,v_token_start:v_token_start+v_token_num]
            _, topk_v_last = torch.topk(ben_attn_v_last_mean, k=v_token_num)
            i_r=intersection_ratio(topk_v_last[:5],topk_v_idx[:50], )
            i_r1=intersection_ratio(topk_v_last[:10],topk_v_idx[:50], )
            i_r2=intersection_ratio(topk_v_last[:50],topk_v_idx[:50], )
            i_r3=intersection_ratio(topk_v_last[:50],topk_v_idx[:100], )
            print('ratio', i_r, i_r1, i_r2,i_r3)
            ben_attn_p_mean=ben_attn_map_list_nc[target_layer_idx].mean(1)[0,-1,:v_token_start]
            _, topk_p_idx = torch.topk(ben_attn_p_mean, k=v_token_start)

            # # 2. Create a mask to select the remaining (non-top-100) indices
            topk_v_idx=topk_v_idx+v_token_start
            all_indices = torch.arange(ben_attn_map_list_nc[target_layer_idx].size(3))
            
            up_mask_p   = torch.zeros_like(all_indices, dtype=torch.bool)
            down_mask_p = torch.zeros_like(all_indices, dtype=torch.bool)
            up_mask_p[topk_p_idx[:30]]=True
            down_mask_p[topk_p_idx[10:]]=True

            up_mask_v   = torch.zeros_like(all_indices, dtype=torch.bool)
            up_mask_v2   = torch.zeros_like(all_indices, dtype=torch.bool)
            down_mask_v = torch.zeros_like(all_indices, dtype=torch.bool)

            keep_mask_v = torch.zeros_like(all_indices, dtype=torch.bool)
            # up_mask_v[v_token_start:v_token_start+24] = True
            # down_mask[v_token_start+24*6:v_token_start+24*24] = True
            # up_mask_v[v_token_start+24*23:v_token_start+24*24] = True
            # down_mask[v_token_start:v_token_start+24*18] = True
            up_mask_v2[topk_v_last[50:100]] = True
            # down_mask[topk_v_idx[:50]] = True
            up_mask_v[topk_v_last[50:150]] = True
            down_mask_v[topk_v_last[30:50]] = True
            keep_mask_v[topk_v_last[0:5]] = True
            keep_mask_v[topk_v_last[200:]] = True
            # down_mask_v[topk_v_idx[:10]] = False
            # up_mask[topk_indices[20:50]] = True
            # down_mask[topk_indices[0:10]] = True

            keep_v_logits_ben = get_attn_mean(
                ben_attn_logits_list_nc[target_layer_idx],
                range=keep_mask_v
            )

    for step in range(adv_steps):
        adv_input_embeds=torch.cat([fixed_input_embeds, adv_token], dim=1)#
        # hidden_states=torch.cat([fixed_input_embeds[:,:v_token_start], img_token1,fixed_input_embeds[:,v_token_start+24*6:],adv_token], dim=1)

        # v_t = adv_input_embeds[:, v_token_start: text_token_start, :]
        # t_t = adv_input_embeds[:, text_token_start: , :]
        # m_v_t = v_t @ t_t.transpose(1, 2) # [1, 576, 53]
        # m_v_t = m_v_t.softmax(2).mean(1)
        
        # hidden_states_list_c, attn_map_list_c, attn_logits_list_c, value_list_c= trace_transformer_layers_grad(
        #     model, adv_input_embeds, attention_mask,
        #     max_layer=target_layer_idx+ben_layer,  
        #     return_all=True,
        #     v_token_start=v_token_start,
        #     v_token_num=v_token_num,
        #     tar_layer=target_layer_idx
        # )
        hidden_states_list_nc, attn_map_list_nc, attn_logits_list_nc, value_list_nc= trace_transformer_layers_grad(
            model, adv_input_embeds, attention_mask,
            # max_layer=target_layer_idx+ben_layer,  
            return_all=True,
        )
        
        attn_logits_nc      = attn_logits_list_nc[target_layer_idx]
        # up_p_logits = get_attn_mean(
        #     attn_logits_nc, 
        #     # 0, v_token_start+24
        #     range=up_mask_p
        # )
        # down_p_logits = get_attn_mean(
        #     attn_logits_nc, 
        #     # 0, v_token_start+24
        #     range=down_mask_p
        # )
        down_v_logits = get_attn_mean(
            attn_logits_nc, 
            # v_token_start+24*6, v_token_start+v_token_num,
            range=down_mask_v
        )
        up_v_logits1 = get_attn_mean(
            attn_logits_nc, 
            # v_token_start+24*6, v_token_start+v_token_num,
            range=up_mask_v
        )
        # up_v_logits2 = get_attn_mean(
        #     attn_logits_nc, 
        #     # v_token_start+24*6, v_token_start+v_token_num,
        #     range=up_mask_v2
        # )
        keep_v_logits = get_attn_mean(
            attn_logits_nc,
            range=keep_mask_v
        )
        # min_mean=torch.min([attn_logits_mean_nc.mean(), attn_logits_mean_nc_v.mean()])
        # max_mean=torch.max([attn_logits_mean_nc.mean(), attn_logits_mean_nc_v.mean()])
        # loss_cur=-min_mean*1-max_mean*3+attn_logits_mean_nc2.mean()
        loss_keep=F.mse_loss(keep_v_logits, keep_v_logits_ben)
        loss_cur=(
            -up_v_logits1.mean()*1.3
            # -up_v_logits2.mean()
            +down_v_logits.mean()
            # -loss_keep*0.03
        )#+(
            # -up_p_logits.mean()*0
            # +down_p_logits.mean()
        # )
        # loss_cur=-F.cross_entropy(
        #     F.log_softmax(attn_logits_list_nc[target_layer_idx][0, :, -1,v_token_start: v_token_start+v_token_num],dim=1),#v_token_start+24*6: v_token_start+v_token_num
        #     F.softmax(ben_attn_logits_list_nc[target_layer_idx][0, :, -1,v_token_start: v_token_start+v_token_num],dim=1),#v_token_start+24*6: v_token_start+v_token_num
        #     reduction='mean'
        # )
        ben_hidden_nc=ben_hidden_states_list_nc[target_layer_idx+ben_layer][:,-2]
        adv_hidden_nc=hidden_states_list_nc[target_layer_idx+ben_layer][:,-1]
        # ben_hidden=ben_hidden.reshape(ben_hidden.shape[0],-1)
        # adv_hidden=adv_hidden.reshape(ben_hidden.shape[0],-1)
        loss_h_nc=F.mse_loss(
            ben_hidden_nc,
            adv_hidden_nc
        )
        # ben_hidden_c=ben_hidden_states_list_c[target_layer_idx+ben_layer][:,-2]
        # adv_hidden_c=hidden_states_list_c[target_layer_idx+ben_layer][:,-1]
        # loss_h_c=-F.mse_loss(
        #     ben_hidden_c,
        #     adv_hidden_c
        # )
        
        alpha=0.9
        loss = alpha*loss_cur#+(1-alpha)*(loss_h_nc+loss_h_c*1.3)*1e1#loss_t+-0.5*loss_next_c
        # loss = loss_cur+loss_h_nc+1*loss_h_c
        
        optimizer.zero_grad()
        loss.backward()
        epsilon=5e-2
        # adv_token.data  = adv_token.data  - adv_lr * adv_token.grad.sign()
        
        # torch.nn.utils.clip_grad_norm_([adv_token], max_norm=1.0)
        # delta=adv_token.data-adv_lr * adv_token.grad.data-ori_adv_token.data
        # delta=torch.clip(adv_lr * adv_token.grad.data, min=-10, max=10)
        # delta=torch.clip(delta, min=-1e-5, max=1e-5)
        # adv_token.data = ori_adv_token + delta
        print(
            step, round(loss_cur.item(),3), 
            round(up_v_logits1.max().item(),5), round(up_v_logits1.min().item(),5),
            round(down_v_logits.max().item(),5), round(down_v_logits.min().item(),5)
        )#
        # if loss_value.item()>0.02:
        #     break
        # if loss_cur.item()<-0.02:
        #     break
        # adv_embeds=adv_token.clone().detach()
        optimizer.step()
        scheduler.step()
        # adv_token.data  = torch.max(torch.min(adv_token.data , ori_adv_token.data + epsilon), ori_adv_token.data - epsilon)
        # del loss, loss_next, loss_t, loss_cur
        # del attn_logits_list_nc, v_t, t_t, m_v_t
    
    # adv_token.data = adv_token.data+torch.randn(
    #     adv_token.shape, dtype=adv_token.dtype
    # ).to(adv_token.device)*0.001
    aa=attn_map_list_nc[2].mean(1)[0,-1,v_token_start:v_token_start+v_token_num]
    print(aa.mean().item(), aa.min().item(), aa.max().item(), )
    new_input_embeds=torch.cat([fixed_input_embeds, adv_token], dim=1)#
    # new_input_embeds=torch.cat([fixed_input_embeds[:,:v_token_start], img_token1,fixed_input_embeds[:,v_token_start+24*6:],adv_token], dim=1)
    return new_input_embeds, new_input_embeds.shape[1]-input_embeds.shape[1]
    # return fixed_input_embeds, 0