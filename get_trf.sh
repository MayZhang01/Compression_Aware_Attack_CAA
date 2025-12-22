# python llava_eval_pope.py \
# --model_path "../../model_weight/llava-v1.5-7b" \
# --layer_list "[2]" \
# --target_layer_idx 2 \
# --adv_steps_list "[100]" \
# --adv_lr 1 \
# --n_patch_num 0.15 \
# --l_patch_num 0.15 \
# --epsilon_list "[128/255]" \
# --bpr_ths 0.1 \
# --cut_num 3 \
# --re_scale "True" \
# --atk "trf" \
# --patch_or_noise "False" \
# --most_or_least "True" \
# --most_flag "True" \
# --w_ori_l 1 --w_ori_p 1 --w_bpr_lp 10 \
# --w_dir_l 5 --w_dir_p 5  --w_bpr_p 20 --w_bpr_l 20 --adv_path ''

# python llava_eval_pope.py \
# --model_path "../../model_weight/llava-v1.5-7b" \
# --layer_list "[2]" \
# --target_layer_idx 2 \
# --adv_steps_list "[100]" \
# --adv_lr 1 \
# --n_patch_num 0.15 \
# --l_patch_num 0.15 \
# --epsilon_list "[64/255]" \
# --bpr_ths 0.1 \
# --cut_num 3 \
# --re_scale "True" \
# --atk "trf" \
# --patch_or_noise "False" \
# --most_or_least "True" \
# --most_flag "True" \
# --w_ori_l 1 --w_ori_p 1 --w_bpr_lp 10 \
# --w_dir_l 5 --w_dir_p 5  --w_bpr_p 20 --w_bpr_l 20 --adv_path ''

# python llava_eval_vqa.py \
# --model_path "../../model_weight/llava-v1.5-7b" \
# --layer_list "[2]" \
# --target_layer_idx 2 \
# --adv_steps_list "[100]" \
# --adv_lr 1 \
# --n_patch_num 0.15 \
# --l_patch_num 0.15 \
# --epsilon_list "[128/255]" \
# --bpr_ths 0.1 \
# --cut_num 3 \
# --re_scale "True" \
# --atk "trf" \
# --patch_or_noise "False" \
# --most_or_least "True" \
# --most_flag "True" \
# --w_ori_l 1 --w_ori_p 1 --w_bpr_lp 10 \
# --w_dir_l 5 --w_dir_p 5  --w_bpr_p 20 --w_bpr_l 20 --adv_path ''

# python llava_eval_mme.py \
# --model_path "../../model_weight/llava-v1.5-7b" \
# --layer_list "[2]" \
# --target_layer_idx 2 \
# --adv_steps_list "[100]" \
# --adv_lr 1 \
# --n_patch_num 0.15 \
# --l_patch_num 0.15 \
# --epsilon_list "[64/255]" \
# --bpr_ths 0.1 \
# --cut_num 3 \
# --re_scale "True" \
# --atk "trf" \
# --patch_or_noise "False" \
# --most_or_least "True" \
# --most_flag "True" \
# --w_ori_l 1 --w_ori_p 1 --w_bpr_lp 10 \
# --w_dir_l 5 --w_dir_p 5  --w_bpr_p 20 --w_bpr_l 20 --adv_path ''

# python llava_eval_vqa.py \
# --model_path "../../model_weight/llava-v1.5-7b" \
# --layer_list "[2]" \
# --target_layer_idx 2 \
# --adv_steps_list "[100]" \
# --adv_lr 1 \
# --n_patch_num 0.15 \
# --l_patch_num 0.15 \
# --epsilon_list "[128/255]" \
# --bpr_ths 0.1 \
# --cut_num 3 \
# --re_scale "True" \
# --atk "trf" \
# --patch_or_noise "False" \
# --most_or_least "True" \
# --most_flag "True" \
# --w_ori_l 1 --w_ori_p 1 --w_bpr_lp 10 \
# --w_dir_l 5 --w_dir_p 5  --w_bpr_p 20 --w_bpr_l 20 --adv_path ''

# python llava_eval_mme.py \
# --model_path "../../model_weight/llava-v1.5-7b" \
# --layer_list "[2]" \
# --target_layer_idx 2 \
# --adv_steps_list "[100]" \
# --adv_lr 1 \
# --n_patch_num 0.15 \
# --l_patch_num 0.15 \
# --epsilon_list "[64/255]" \
# --bpr_ths 0.1 \
# --cut_num 3 \
# --re_scale "True" \
# --atk "trf" \
# --patch_or_noise "False" \
# --most_or_least "True" \
# --most_flag "True" \
# --w_ori_l 1 --w_ori_p 1 --w_bpr_lp 10 \
# --w_dir_l 5 --w_dir_p 5  --w_bpr_p 20 --w_bpr_l 20 --adv_path ''

# python llava_eval_pope.py \
# --model_path "../../model_weight/llava-v1.5-7b" \
# --layer_list "[2, 6, 15]" \
# --target_layer_idx 2 \
# --adv_steps_list "[100]" \
# --adv_lr 1 \
# --n_patch_num 0.15 \
# --l_patch_num 0.15 \
# --epsilon_list "[64/255]" \
# --bpr_ths 0.1 \
# --cut_num 3 \
# --re_scale "True" \
# --atk "trf" \
# --patch_or_noise "False" \
# --most_or_least "True" \
# --most_flag "True" \
# --w_ori_l 1 --w_ori_p 1 --w_bpr_lp 10 \
# --w_dir_l 5 --w_dir_p 5  --w_bpr_p 20 --w_bpr_l 20 --adv_path ''

python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[100]" \
--adv_lr 1 \
--n_patch_num 0.15 \
--l_patch_num 0.15 \
--epsilon_list "[64/255]" \
--bpr_ths 0.1 \
--cut_num 3 \
--re_scale "True" \
--atk "trf" \
--patch_or_noise "False" \
--most_or_least "True" \
--most_flag "True" \
--w_ori_l 1 --w_ori_p 1 --w_bpr_lp 10 \
--w_dir_l 5 --w_dir_p 5  --w_bpr_p 20 --w_bpr_l 20 --adv_path ''

python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2, 6, 15]" \
--target_layer_idx 2 \
--adv_steps_list "[100]" \
--adv_lr 1 \
--n_patch_num 0.15 \
--l_patch_num 0.15 \
--epsilon_list "[64/255]" \
--bpr_ths 0.1 \
--cut_num 3 \
--re_scale "True" \
--atk "trf" \
--patch_or_noise "False" \
--most_or_least "True" \
--most_flag "True" \
--w_ori_l 1 --w_ori_p 1 --w_bpr_lp 10 \
--w_dir_l 5 --w_dir_p 5  --w_bpr_p 20 --w_bpr_l 20 --adv_path ''

python llava_eval_mme.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[100]" \
--adv_lr 1 \
--n_patch_num 0.15 \
--l_patch_num 0.15 \
--epsilon_list "[64/255]" \
--bpr_ths 0.1 \
--cut_num 3 \
--re_scale "True" \
--atk "trf" \
--patch_or_noise "False" \
--most_or_least "True" \
--most_flag "True" \
--w_ori_l 1 --w_ori_p 1 --w_bpr_lp 10 \
--w_dir_l 5 --w_dir_p 5  --w_bpr_p 20 --w_bpr_l 20 --adv_path ''

python llava_eval_mme.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2, 6, 15]" \
--target_layer_idx 2 \
--adv_steps_list "[100]" \
--adv_lr 1 \
--n_patch_num 0.15 \
--l_patch_num 0.15 \
--epsilon_list "[64/255]" \
--bpr_ths 0.1 \
--cut_num 3 \
--re_scale "True" \
--atk "trf" \
--patch_or_noise "False" \
--most_or_least "True" \
--most_flag "True" \
--w_ori_l 1 --w_ori_p 1 --w_bpr_lp 10 \
--w_dir_l 5 --w_dir_p 5  --w_bpr_p 20 --w_bpr_l 20 --adv_path ''