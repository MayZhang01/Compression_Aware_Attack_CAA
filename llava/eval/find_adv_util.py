import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import math
from torch.optim.lr_scheduler import StepLR
from llava.eval.utils import loss_norm, intersection_ratio, add_square_patch, get_patch_mask, save_clip_tensor_as_image
import random
import argparse


@torch.no_grad()
def effective_weight(linear):
    dev = next(linear.parameters()).device
    # 选一个合理 dtype；或直接用下一行从参数推断
    dtype = next((p.dtype for p in linear.parameters() if p.requires_grad), torch.float16)
    in_f, out_f = linear.in_features, linear.out_features
    I = torch.eye(in_f, device=dev, dtype=dtype)           # (in_f, in_f)
    Y = linear(I)                                          # (in_f, out_f) = I @ W^T + b
    b = linear.bias if linear.bias is not None else 0
    W_T = Y - b                                            # 去掉 bias 广播
    return W_T.T.contiguous()                              # (out_f, in_f)

def judge_adv(output_c:str, output_nc:str):
    output_c=output_c.lower()
    output_nc=output_nc.lower()
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
    mask_layer=None, 
    most_mask=None,
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

        
        if most_mask is not None and mask_layer is not None and mask_layer>0:
            if i+1==mask_layer:
                hidden_states = hidden_states[:, most_mask]
                # cos = position_embeddings[0][:, :, most_mask]
                # sin = position_embeddings[1][:, :, most_mask]
                # position_embeddings = (cos, sin)
                (B, T, D)=hidden_states.shape

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


def BPR_loss(y_tensor, p_list=None, clamp_min=-20, clamp_max=20, t=1):
    y_list=[]
    for idx in range(len(p_list)):
        if torch.sum(p_list[idx])>0:
            y_list.append(y_tensor[p_list[idx]])
    if len(y_list)<=1:
        return torch.tensor(1, device=y_tensor.device)
    loss_l=[]
    for idx in range(1, len(y_list), 1):
        tmp_diff=y_list[idx-1][:, None] - y_list[idx][None, :]
        tmp_diff = torch.clamp(tmp_diff, min=clamp_min, max=clamp_max)
        tmp_loss=-torch.mean(torch.log(torch.sigmoid(tmp_diff/t)))
        if tmp_loss==torch.inf:
            tmp_loss=-torch.mean(-F.softplus(-tmp_diff)) 
        loss_l.append(tmp_loss.unsqueeze(0))
    return torch.mean(torch.cat(loss_l))

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

def weighted_sign_mse_loss(x, y, k=5):
    diff = x - y
    sign_diff = torch.tanh(x*k) - torch.tanh(y*k)
    loss = (diff ** 2) * (sign_diff ** 2)#
    return loss.mean()

def mse_loss(x, y, k=1):
    loss = (x - y) ** 2
    return loss.mean()

def get_args(ArgsDefault):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=None)
    parser.add_argument('--layer_list', type=str, default=None)
    parser.add_argument('--image_token_ratio_list', type=str, default=None)
    parser.add_argument('--target_layer_idx', type=int, default=None)
    parser.add_argument('--adv_steps_list', type=str, default=None)
    parser.add_argument('--adv_lr', type=float, default=None)
    parser.add_argument('--n_patch_num', type=float, default=None)
    parser.add_argument('--l_patch_num', type=float, default=None)
    parser.add_argument('--epsilon_list', type=str, default=None)
    parser.add_argument('--bpr_ths', type=float, default=None)
    parser.add_argument('--cut_num', type=int, default=None)
    parser.add_argument('--re_scale', type=str, default=None)
    parser.add_argument('--atk', type=str, default=None)
    parser.add_argument('--patch_or_noise', type=str, default=None)
    parser.add_argument('--most_or_least', type=str, default=None)
    parser.add_argument('--rlt_path', type=str, default=None)
    parser.add_argument('--w_bpr_l', type=float, default=None)
    parser.add_argument('--w_bpr_p', type=float, default=None)
    parser.add_argument('--w_dir_l', type=float, default=None)
    parser.add_argument('--w_dir_p', type=float, default=None)
    parser.add_argument('--w_bpr_lp', type=float, default=None)
    parser.add_argument('--w_ori_l', type=float, default=None)
    parser.add_argument('--w_ori_p', type=float, default=None)
    parser.add_argument('--most_flag', type=str, default=None)
    parser.add_argument('--stop_teps', type=float, default=None)
    parser.add_argument('--stop_ratio', type=float, default=None)
    parser.add_argument('--mask_layer', type=int, default=None)
    parser.add_argument('--adv_path', type=str, default=None)
    parser.add_argument('--ori_bound', type=int, default=None)
    parser.add_argument('--res_file', type=str, default=None)
    parser.add_argument('--benign_flag', type=str, default=None)

    args_cmd = parser.parse_args()
    # 默认参数
    args = ArgsDefault()
    # 如果命令行有输入，则覆盖默认
    for k in vars(args_cmd):
        v = getattr(args_cmd, k)
        if v is not None:
            if v == "True":
                v = True
            elif v == "False":
                v = False
            setattr(args, k, v)
    return args


@torch.no_grad()
def get_max_min_Q(x, attn, t_idx_num, dtype=torch.float16):
    q_proj, k_proj = attn.q_proj, attn.k_proj

    # 1) 得到等效浮点权重（无论是否量化/LoRA，都数值等价于模块前向）
    W_Q = effective_weight(q_proj)    # (out_q, in)
    W_K = effective_weight(k_proj)    # (out_k, in)
    # b_Q = q_proj.bias if q_proj.bias is not None else 0
    # b_K = k_proj.bias if k_proj.bias is not None else 0

    # 2) 计算 q 向量并去掉 bias
    # 注意：等价于 q_proj(x) - b_Q
    q = x @ W_Q.T                     # (out_q,)

    # # 3) 若 q 与 k 维度不一致（GQA），做组均值到 kv 维度
    # num_heads = attn.num_heads
    # num_kv_heads = attn.num_key_value_heads
    # head_dim = q.shape[0] // num_heads
    # if num_heads != num_kv_heads:
    #     g = num_heads // num_kv_heads
    #     q_heads = q.view(num_heads, head_dim)                  # (H, d)
    #     q_kv = q_heads.view(num_kv_heads, g, head_dim).mean(1) # (Kv, d)
    #     q_target = q_kv.reshape(-1)                            # (out_k,)
    # else:
    #     q_target = q                                           # (out_k,)
    
    # 前置：已得到 q_target (即你的 q_target / q_kv) 以及等效 W_K
    Wf = W_K.to(torch.float32)               # (out_k, in)
    qf = q.to(torch.float32)          # (out_k,)

    v_dir_max = (Wf.T @ qf.T).mean(1)                      # (in,)
    v_max = v_dir_max / (v_dir_max.norm() + 1e-12)       # 方向对齐；若想更大logit，可再乘一个标量α（非Pre-Norm时）
    v_max = v_max.to(dtype)   

    # 让打分最小：取反方向
    v_dir_min = -(Wf.T @ qf.T).mean(1)                    # (in,)
    v_min = v_dir_min / (v_dir_min.norm() + 1e-12)
    v_min = v_min.to(dtype)
    
    # # 最小二乘
    # Wf   = W_K.to(torch.float32)                 # (out_k, in)
    # rhsB = (q_target - b_K).to(torch.float32).unsqueeze(-1)  # (out_k, 1)
    # v = torch.linalg.lstsq(Wf, rhsB).solution.squeeze(-1)    # (in,)
    # v = v.to(W_K.dtype)                      # (in,)
    return v_max, v_min