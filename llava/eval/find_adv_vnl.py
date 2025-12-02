import torch
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import StepLR
from llava.eval.utils import save_clip_tensor_as_image
from llava.eval.find_adv_util import cat_adv_embed, trace_transformer_layers_grad
from llava.eval.find_adv_util import get_patch_mask


def opt_adv_image_vanilla(
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
    target_label=67597,
    untarget_label=None,
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
    if target_layer_idx>=model.config.num_hidden_layers:
        target_layer_idx=model.config.num_hidden_layers-1
    
    torch.manual_seed(123)                   
    torch.cuda.manual_seed(123)
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

        ben_logits = model.lm_head(ben_hidden_states_list_nc[target_layer_idx][:,-1])
        ben_target = ben_logits.argmax(dim=1)
        
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
            most_mask[topk_v_tar[-l_patch_num:-n_patch_num]] = False
            second_mask[topk_v_tar[-l_patch_num:-n_patch_num]] = False

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
            second_mask[topk_v_tar[-n_patch_num:]] = False
            least_mask[topk_v_tar[-n_patch_num:]] = False

            pixel_patch_idx = topk_v_tar[-n_patch_num:]-v_token_start
            pixel_patch_mask= get_patch_mask(pixel_patch_idx, image_size=image_size, patch_size=patch_size)
            pixel_patch_mask= pixel_patch_mask.to(device)
        else:
            pixel_patch_mask=torch.zeros((image_size, image_size), device=device, dtype=dtype)

    if adv_steps==0:
        adv_steps+=1
        update_flag=False
    else:
        update_flag=True
    
    last_adv_perturbs=torch.zeros_like(adv_perturbs)
    last_adv_patch=torch.zeros_like(adv_patch)
    for step in range(adv_steps):
        adv_images=images.clone()
        if n_patch_num>0:
            if patch_or_noise:
                adv_images=adv_images*(1-pixel_patch_mask)+adv_patch*pixel_patch_mask
            else:
                adv_images=adv_images+adv_patch*pixel_patch_mask
        if l_patch_num>0:
            adv_images=adv_images+adv_perturbs*pixel_least_mask
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
        logits = model.lm_head(hidden_states_list_nc[target_layer_idx][:,-1])
        loss_list, loss_dict=[], {}
        if untarget_label is not None:
            if isinstance(untarget_label, list):
                ce_list = []
                for label in untarget_label:
                    target = label * torch.ones_like(ben_target)
                    ce = F.cross_entropy(logits, target.long())
                    ce_list.append(ce)
                loss_un = torch.stack(ce_list).min()
            else:
                loss_un = F.cross_entropy(logits, untarget_label * torch.ones_like(ben_target))
        else:
            loss_un = F.cross_entropy(logits, ben_target)  # 目标类设为第0类
        if (isinstance(target_label, list) and ben_target.item() in target_label) or (isinstance(target_label, int) and ben_target.item()==target_label):
            loss_un = loss_un +21
            loss_list.append(loss_un)
        else:
            loss_list.append((loss_un)*(-3))
        loss_dict['CE_un']=round(loss_un.item(),2)
        if target_label is not None:
            if isinstance(target_label, list):
                ce_list = []
                for label in target_label:
                    target = label * torch.ones_like(ben_target)
                    ce = F.cross_entropy(logits, target.long())
                    ce_list.append(ce)
                loss_t = torch.stack(ce_list).min()
            else:
                loss_t = F.cross_entropy(logits, target_label * torch.ones_like(ben_target))
            
            loss_list.append((loss_t)*1)
            loss_dict['CE_t']=round(loss_t.item(),2)
        loss = torch.stack(loss_list).sum()
        
        if torch.isnan(loss):
            logger.print_log('Loss is NaN, stop optimization.')
            adv_perturbs=last_adv_perturbs
            adv_patch=last_adv_patch
            break
        
        last_adv_perturbs=adv_perturbs.detach().clone()
        last_adv_patch=adv_patch.detach().clone()

        if update_flag:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            adv_perturbs.data=torch.clamp(adv_perturbs.data, -epsilon*image_max, epsilon*image_max)
            if patch_or_noise==False:
                adv_patch.data=torch.clamp(adv_patch.data, -epsilon*image_max, epsilon*image_max)
        adv_label=logits.argmax(dim=1).item()
        logger.print_log(
            step, 
            'loss: ', round(loss.item(),1), loss_dict, 
            "ben_label:", ben_target.item(), 
            "adv_label:", adv_label,
            "untarget_label:", untarget_label,
            "target_label:", target_label,
        )
        if (loss_un.item()>20 and loss_t.item()<1) and (
            adv_label in (target_label if isinstance(target_label, list) else [target_label])
        ) and (
            (untarget_label is None) or (adv_label not in (untarget_label if isinstance(untarget_label, list) else [untarget_label]))
        ):
            logger.print_log('Attack success, stop optimization.')
            break
        del hidden_states_list_nc
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        adv_images=images.clone()
        if patch_or_noise:
            adv_images=adv_images*(1-pixel_patch_mask)+adv_patch*pixel_patch_mask
        else:
            adv_images=adv_images+adv_patch*pixel_patch_mask
        adv_images=adv_images+adv_perturbs*pixel_least_mask
        # adv_images = torch.clamp(adv_images, -image_max*1.1, image_max*1.1)
        adv_image_embeds = model.encode_images(adv_images)
        adv_input_embeds=cat_adv_embed(
            fixed_input_embeds, adv_image_embeds, 
            start_idx=v_token_start, end_idx=t_token_start
        )
        
    save_clip_tensor_as_image(adv_images.cpu(),'3.png')
    return adv_images, adv_input_embeds, adv_input_embeds.shape[1]-input_embeds.shape[1], patch_inter_ratio, least_inter_ratio