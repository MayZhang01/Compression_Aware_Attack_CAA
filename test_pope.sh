# python llava_eval_pope.py \
# --model_path "../../model_weight/llava-v1.5-7b" \
# --layer_list "[2, 6, 15]" \
# --target_layer_idx 2 \
# --adv_steps_list "[150]" \
# --adv_lr 1 \
# --n_patch_num 0.2 \
# --l_patch_num 0.4 \
# --epsilon_list "[32/255, 16/255, 64/255, ]" \
# --bpr_ths 0.1 \
# --cut_num 1 \
# --re_scale "False" \
# --atk "ca" \
# --patch_or_noise "False" \
# --most_or_least "True"

# python llava_eval_pope.py \
# --model_path "../../model_weight/llava-v1.5-7b" \
# --layer_list "[8, 16, 24]" \
# --target_layer_idx 8 \
# --adv_steps_list "[150]" \
# --adv_lr 1 \
# --n_patch_num 0.2 \
# --l_patch_num 0.4 \
# --epsilon_list "[32/255]" \
# --bpr_ths 0.1 \
# --cut_num 1 \
# --re_scale "False" \
# --atk "ca" \
# --patch_or_noise "False" \
# --most_or_least "True"

# python llava_eval_pope.py \
# --model_path "../../model_weight/llava-v1.5-7b" \
# --layer_list "[8, 16, 24]" \
# --target_layer_idx 8 \
# --adv_steps_list "[150]" \
# --adv_lr 1 \
# --n_patch_num 0.2 \
# --l_patch_num 0.4 \
# --epsilon_list "[32/255]" \
# --bpr_ths 0.1 \
# --cut_num 1 \
# --re_scale "False" \
# --atk "ca" \
# --patch_or_noise "False" \
# --most_or_least "True"


# python llava_eval_mme.py \
# --model_path "../../model_weight/llava-v1.6-vicuna-7b" \
# --layer_list "[2, 6, 15]" \
# --target_layer_idx 2 \
# --adv_steps_list "[120]" \
# --adv_lr 1 \
# --n_patch_num 0.3 \
# --l_patch_num 0.5 \
# --epsilon_list "[16/255, 64/255, 32/255, ]" \
# --bpr_ths 0.1 \
# --cut_num 1 \
# --re_scale "False" \
# --atk "ca" \
# --patch_or_noise "False" \
# --most_or_least "True" \
# --w_bpr_l 20 --w_bpr_p 20 --w_dir_l 5 --w_dir_p 5 --w_bpr_lp 10 --w_ori_l 10 --w_ori_p 10 \
# --stop_ratio 0.8 --stop_teps 20



# python llava_eval_vqa.py \
# --model_path "../../model_weight/llava-v1.6-vicuna-7b" \
# --layer_list "[2, 6, 15]" \
# --target_layer_idx 2 \
# --adv_steps_list "[100]" \
# --adv_lr 1 \
# --n_patch_num 0.05 \
# --l_patch_num 0.1 \
# --epsilon_list "[32/255, 16/255, 64/255]" \
# --bpr_ths 0.1 \
# --cut_num 1 \
# --re_scale "False" \
# --atk "ca" \
# --patch_or_noise "False" \
# --most_or_least "True"

# python llava_eval_mme.py \
# --model_path "../../model_weight/llava-v1.5-7b" \
# --layer_list "[2, 6, 15]" \
# --target_layer_idx 2 \
# --adv_steps_list "[150]" \
# --adv_lr 1 \
# --n_patch_num 0.2 \
# --l_patch_num 0.4 \
# --epsilon_list "[32/255, 16/255, 64/255]" \
# --bpr_ths 0.1 \
# --cut_num 1 \
# --re_scale "False" \
# --atk "ca" \
# --patch_or_noise "False" \
# --most_or_least "True"

# python llava_eval_vqa.py \
# --model_path "../../model_weight/llava-v1.5-7b" \
# --layer_list "[2, 6, 15]" \
# --target_layer_idx 2 \
# --adv_steps_list "[100]" \
# --adv_lr 1 \
# --n_patch_num 0.1 \
# --l_patch_num 0.2 \
# --epsilon_list "[32/255, 16/255, 64/255]" \
# --bpr_ths 0.1 \
# --cut_num 1 \
# --re_scale "False" \
# --atk "ca" \
# --patch_or_noise "False" \
# --most_or_least "True"

# python llava_eval_vqa.py \
# --model_path "../../model_weight/llava-v1.6-vicuna-7b" \
# --layer_list "[2]" \
# --target_layer_idx 2 \
# --adv_steps_list "[100]" \
# --adv_lr 1 \
# --n_patch_num 0.05 \
# --l_patch_num 0.1 \
# --epsilon_list "[32/255, 16/255, 64/255]" \
# --bpr_ths 0.1 \
# --cut_num 1 \
# --re_scale "False" \
# --atk "ca" \
# --patch_or_noise "False" \
# --most_or_least "True"


# python llava_eval_mme.py \
# --model_path "../../model_weight/llava-v1.6-vicuna-7b" \
# --layer_list "[2]" \
# --target_layer_idx 2 \
# --adv_steps_list "[120]" \
# --adv_lr 1 \
# --n_patch_num 0.3 \
# --l_patch_num 0.5 \
# --epsilon_list "[16/255, 64/255, 32/255, ]" \
# --bpr_ths 0.1 \
# --cut_num 1 \
# --re_scale "False" \
# --atk "ca" \
# --patch_or_noise "False" \
# --most_or_least "True" \
# --w_bpr_l 20 --w_bpr_p 20 --w_dir_l 5 --w_dir_p 5 --w_bpr_lp 10 --w_ori_l 10 --w_ori_p 10 \
# --stop_ratio 0.8 --stop_teps 20

# python llava_eval_pope.py \
# --model_path "../../model_weight/llava-v1.6-vicuna-7b" \
# --layer_list "[2, 6, 15]" \
# --target_layer_idx 2 \
# --adv_steps_list "[150]" \
# --adv_lr 1 \
# --n_patch_num 0.2 \
# --l_patch_num 0.4 \
# --epsilon_list "[32/255, 16/255, 64/255, ]" \
# --bpr_ths 0.1 \
# --cut_num 1 \
# --re_scale "False" \
# --atk "ca" \
# --patch_or_noise "False" \
# --most_or_least "True"

# python llava_eval_pope.py \
# --model_path "../../model_weight/llava-v1.6-vicuna-7b" \
# --layer_list "[2]" \
# --target_layer_idx 2 \
# --adv_steps_list "[150]" \
# --adv_lr 1 \
# --n_patch_num 0.2 \
# --l_patch_num 0.4 \
# --epsilon_list "[32/255, 16/255, 64/255, ]" \
# --bpr_ths 0.1 \
# --cut_num 1 \
# --re_scale "False" \
# --atk "ca" \
# --patch_or_noise "False" \
# --most_or_least "True"

python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[8, 16, 24]" \
--target_layer_idx 8 \
--adv_steps_list "[150]" \
--adv_lr 1 \
--n_patch_num 0.2 \
--l_patch_num 0.4 \
--epsilon_list "[64/255, ]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"

python llava_eval_mme.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2, 6, 15]" \
--target_layer_idx 2 \
--adv_steps_list "[120]" \
--adv_lr 1 \
--n_patch_num 0.2 \
--l_patch_num 0.5 \
--epsilon_list "[16/255, 64/255, 32/255, ]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True" \
--w_bpr_l 20 --w_bpr_p 20 --w_dir_l 5 --w_dir_p 5 --w_bpr_lp 10 --w_ori_l 10 --w_ori_p 10 \
--stop_ratio 0.8 --stop_teps 20