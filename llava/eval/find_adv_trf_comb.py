import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import math
import numpy as np
from torch.optim.lr_scheduler import StepLR
from llava.eval.utils import loss_norm, intersection_ratio, add_square_patch, save_clip_tensor_as_image, add_border
from llava.eval.utils import crop_and_resize_image
from llava.eval.find_adv_util import cat_adv_embed, trace_transformer_layers_grad, get_attn_mean, BPR_loss, weighted_sign_mse_loss, mse_loss
from llava.eval.find_adv_util import get_patch_mask, effective_weight, get_max_min_Q

def get_patch_index(
    x: torch.Tensor,
    patch_size: int = 14,
    threshold: float = 1.0,
    large_or_small=True
):
    """
    Args:
        x: Tensor of shape (H, W), here (336, 336)
        patch_size: patch height / width
        threshold: sum threshold

    Returns:
        indices: 1D LongTensor of patch indices (row-major)
    """
    H, W = x.shape

    gh = H // patch_size
    gw = W // patch_size

    # (gh, patch, gw, patch)
    patches = x.view(
        gh, patch_size,
        gw, patch_size
    )

    # sum over patch area -> (gh, gw)
    patch_sums = patches.sum(dim=(1, 3))

    # mask of patches satisfying condition
    if large_or_small:
        mask = patch_sums >= threshold
    else:
        mask = patch_sums < threshold

    # row-major indices
    indices = torch.nonzero(mask, as_tuple=False)
    patch_indices = indices[:, 0] * gw + indices[:, 1]

    return patch_indices

def random_split_into_k_parts(patch_indices: torch.Tensor, k: int):
    """
    Randomly shuffle patch_indices and split into k nearly equal parts.

    Args:
        patch_indices: 1D LongTensor of shape (N,)
        k: number of parts

    Returns:
        parts: list of k tensors
    """
    N = patch_indices.numel()
    assert k > 0, "k must be > 0"

    perm = torch.randperm(N, device=patch_indices.device)
    shuffled = patch_indices[perm]

    base = N // k
    remainder = N % k

    parts = []
    start = 0
    for i in range(k):
        size = base + (1 if i < remainder else 0)
        parts.append(shuffled[start:start + size])
        start += size

    return parts

def opt_adv_image_patch_transf_comb(
    image,
    adv_least_list,
    adv_patch_list,
    patch_size = 14,
    eps = 1e-2,
    epsilon=64/255,
):
    k=len(adv_least_list)
    device=image.device
    B, C, H, W =image.shape
    grid_h=int(H/patch_size)
    grid_w=int(W/patch_size)
    mask = torch.zeros((H, W), device=image.device)
    
    b_w=3 #q2l  pope
    # b_w=2 #q2l  vqa
    image[0][:, :patch_size*b_w, :] = 0          # 上
    image[0][:, -patch_size*b_w:, :] = 0         # 下
    image[0][:, :, :patch_size*b_w] = 0          # 左
    image[0][:, :, -patch_size*b_w:] = 0         # 右

    zero_pos = (image[0].abs() < eps).any(dim=0)  # (H, W)
    mask[zero_pos] = 1

    mask_patch_indices=get_patch_index(mask, patch_size=patch_size, threshold=patch_size, large_or_small=True)
    image_patch_indices=get_patch_index(mask, patch_size=patch_size, threshold=patch_size, large_or_small=False)

    mask_parts=random_split_into_k_parts(mask_patch_indices, k)
    image_parts=random_split_into_k_parts(image_patch_indices, k)

    mask_raise=[
        get_patch_mask(mp, H, patch_size=patch_size).to(device)
        for mp in mask_parts
    ]
    mask_press=[
        get_patch_mask(ip, H, patch_size=patch_size).to(device)
        for ip in image_parts
    ]

    adv_image=image.clone()
    for idx in range(k):
        adv_image=torch.clamp(adv_image*mask_press[idx], -0.2, 1)
        if epsilon>0:
            adv_patch=adv_patch_list[idx].to(device)
            tmp_adv_patch = adv_patch.reshape(C, 1, patch_size, 1, patch_size)
            tmp_adv_patch = tmp_adv_patch.repeat(1, grid_h, 1, grid_w, 1)
            tmp_adv_patch = tmp_adv_patch.reshape(C, grid_h*patch_size, grid_w*patch_size)
            adv_image=adv_image+mask_press[idx]*tmp_adv_patch*epsilon
 
        adv_least=adv_least_list[idx].to(device)
        tmp_adv_least = adv_least.reshape(C, 1, patch_size, 1, patch_size)
        tmp_adv_least = tmp_adv_least.repeat(1,grid_h,1,grid_w,1)
        tmp_adv_least = tmp_adv_least.reshape(C, grid_h*patch_size, grid_w*patch_size)
        adv_image=adv_image+mask_raise[idx]*tmp_adv_least
        
    
    save_clip_tensor_as_image(adv_image.cpu(),'2.png')
    return adv_image