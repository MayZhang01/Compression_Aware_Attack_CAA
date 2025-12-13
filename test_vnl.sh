# python llava_eval_pope.py \
# --model_path "../../model_weight/llava-v1.5-7b" \
# --layer_list "[2]" \
# --target_layer_idx -1 \
# --adv_steps_list "[100]" \
# --adv_lr 1 \
# --n_patch_num 1 \
# --epsilon_list "[32/255, 16/255, 64/255, ]" \
# --bpr_ths 0.1 \
# --cut_num 1 \
# --re_scale "False" \
# --atk "vnl" \
# --patch_or_noise "False" \
# --most_or_least "True" \
# --mask_layer -1

# python llava_eval_pope.py \
# --model_path "../../model_weight/llava-v1.5-7b" \
# --layer_list "[2, 6, 15]" \
# --target_layer_idx -1 \
# --adv_steps_list "[100]" \
# --adv_lr 1 \
# --n_patch_num 1 \
# --epsilon_list "[32/255, 16/255, 64/255, ]" \
# --bpr_ths 0.1 \
# --cut_num 1 \
# --re_scale "False" \
# --atk "vnl" \
# --patch_or_noise "False" \
# --most_or_least "True" \
# --mask_layer -1

# python llava_eval_pope.py \
# --model_path "../../model_weight/llava-v1.5-7b" \
# --layer_list "[8, 16, 24]" \
# --target_layer_idx -1 \
# --adv_steps_list "[100]" \
# --adv_lr 1 \
# --n_patch_num 1 \
# --epsilon_list "[32/255, 16/255, 64/255, ]" \
# --bpr_ths 0.1 \
# --cut_num 1 \
# --re_scale "False" \
# --atk "vnl" \
# --patch_or_noise "False" \
# --most_or_least "True" \
# --mask_layer -1

# python llava_eval_mme.py \
# --model_path "../../model_weight/llava-v1.5-7b" \
# --layer_list "[2]" \
# --target_layer_idx -1 \
# --adv_steps_list "[100]" \
# --adv_lr 1 \
# --n_patch_num 1 \
# --epsilon_list "[32/255, 16/255, 64/255, ]" \
# --bpr_ths 0.1 \
# --cut_num 1 \
# --re_scale "False" \
# --atk "vnl" \
# --patch_or_noise "False" \
# --most_or_least "True" \
# --mask_layer -1

# python llava_eval_mme.py \
# --model_path "../../model_weight/llava-v1.5-7b" \
# --layer_list "[2, 6, 15]" \
# --target_layer_idx -1 \
# --adv_steps_list "[100]" \
# --adv_lr 1 \
# --n_patch_num 1 \
# --epsilon_list "[32/255, 16/255, 64/255, ]" \
# --bpr_ths 0.1 \
# --cut_num 1 \
# --re_scale "False" \
# --atk "vnl" \
# --patch_or_noise "False" \
# --most_or_least "True" \
# --mask_layer -1

# python llava_eval_mme.py \
# --model_path "../../model_weight/llava-v1.5-7b" \
# --layer_list "[8, 16, 24]" \
# --target_layer_idx -1 \
# --adv_steps_list "[100]" \
# --adv_lr 1 \
# --n_patch_num 1 \
# --epsilon_list "[32/255, 16/255, 64/255, ]" \
# --bpr_ths 0.1 \
# --cut_num 1 \
# --re_scale "False" \
# --atk "vnl" \
# --patch_or_noise "False" \
# --most_or_least "True" \
# --mask_layer -1


# python llava_eval_vqa.py \
# --model_path "../../model_weight/llava-v1.5-7b" \
# --layer_list "[2]" \
# --target_layer_idx -1 \
# --adv_steps_list "[100]" \
# --adv_lr 1 \
# --n_patch_num 1 \
# --epsilon_list "[32/255, 16/255, 64/255, ]" \
# --bpr_ths 0.1 \
# --cut_num 1 \
# --re_scale "False" \
# --atk "vnl" \
# --patch_or_noise "False" \
# --most_or_least "True" \
# --mask_layer -1

# python llava_eval_vqa.py \
# --model_path "../../model_weight/llava-v1.5-7b" \
# --layer_list "[2, 6, 15]" \
# --target_layer_idx -1 \
# --adv_steps_list "[100]" \
# --adv_lr 1 \
# --n_patch_num 1 \
# --epsilon_list "[32/255, 16/255, 64/255, ]" \
# --bpr_ths 0.1 \
# --cut_num 1 \
# --re_scale "False" \
# --atk "vnl" \
# --patch_or_noise "False" \
# --most_or_least "True" \
# --mask_layer -1

# python llava_eval_vqa.py \
# --model_path "../../model_weight/llava-v1.5-7b" \
# --layer_list "[8, 16, 24]" \
# --target_layer_idx -1 \
# --adv_steps_list "[100]" \
# --adv_lr 1 \
# --n_patch_num 1 \
# --epsilon_list "[32/255, 16/255, 64/255, ]" \
# --bpr_ths 0.1 \
# --cut_num 1 \
# --re_scale "False" \
# --atk "vnl" \
# --patch_or_noise "False" \
# --most_or_least "True" \
# --mask_layer -1


python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2]" \
--target_layer_idx -1 \
--adv_steps_list "[100]" \
--adv_lr 1 \
--n_patch_num 1 \
--epsilon_list "[32/255, 16/255, 64/255, ]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "vnl" \
--patch_or_noise "False" \
--most_or_least "True" \
--mask_layer -1

python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2, 6, 15]" \
--target_layer_idx -1 \
--adv_steps_list "[100]" \
--adv_lr 1 \
--n_patch_num 1 \
--epsilon_list "[32/255, 16/255, 64/255, ]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "vnl" \
--patch_or_noise "False" \
--most_or_least "True" \
--mask_layer -1

python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[8, 16, 24]" \
--target_layer_idx -1 \
--adv_steps_list "[100]" \
--adv_lr 1 \
--n_patch_num 1 \
--epsilon_list "[32/255, 16/255, 64/255, ]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "vnl" \
--patch_or_noise "False" \
--most_or_least "True" \
--mask_layer -1

python llava_eval_mme.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2]" \
--target_layer_idx -1 \
--adv_steps_list "[100]" \
--adv_lr 1 \
--n_patch_num 1 \
--epsilon_list "[32/255, 16/255, 64/255, ]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "vnl" \
--patch_or_noise "False" \
--most_or_least "True" \
--mask_layer -1

python llava_eval_mme.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2, 6, 15]" \
--target_layer_idx -1 \
--adv_steps_list "[100]" \
--adv_lr 1 \
--n_patch_num 1 \
--epsilon_list "[32/255, 16/255, 64/255, ]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "vnl" \
--patch_or_noise "False" \
--most_or_least "True" \
--mask_layer -1

python llava_eval_mme.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[8, 16, 24]" \
--target_layer_idx -1 \
--adv_steps_list "[100]" \
--adv_lr 1 \
--n_patch_num 1 \
--epsilon_list "[32/255, 16/255, 64/255, ]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "vnl" \
--patch_or_noise "False" \
--most_or_least "True" \
--mask_layer -1


python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2]" \
--target_layer_idx -1 \
--adv_steps_list "[100]" \
--adv_lr 1 \
--n_patch_num 1 \
--epsilon_list "[32/255, 16/255, 64/255, ]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "vnl" \
--patch_or_noise "False" \
--most_or_least "True" \
--mask_layer -1

python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2, 6, 15]" \
--target_layer_idx -1 \
--adv_steps_list "[100]" \
--adv_lr 1 \
--n_patch_num 1 \
--epsilon_list "[32/255, 16/255, 64/255, ]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "vnl" \
--patch_or_noise "False" \
--most_or_least "True" \
--mask_layer -1

python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[8, 16, 24]" \
--target_layer_idx -1 \
--adv_steps_list "[100]" \
--adv_lr 1 \
--n_patch_num 1 \
--epsilon_list "[32/255, 16/255, 64/255, ]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "vnl" \
--patch_or_noise "False" \
--most_or_least "True" \
--mask_layer -1