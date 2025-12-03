# pope llava fastv
python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[0]" \
--adv_lr 1 \
--n_patch_num 1 \
--l_patch_num 0 \
--epsilon_list "[0/255, 16/255, 32/255, 64/255]" \
--image_token_ratio_list "[1.0, 0.7, 0.5, 0.3, 0.2, 0.1]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"

# pope llava pdrop
python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[8, 16]" \
--target_layer_idx 2 \
--adv_steps_list "[0]" \
--adv_lr 1 \
--n_patch_num 1 \
--l_patch_num 0 \
--epsilon_list "[0/255, 16/255, 32/255, 64/255]" \
--image_token_ratio_list "[1.0, 0.7, 0.5, 0.3, 0.2, 0.1]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"

# pope llava sparse
python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2, 8, 16]" \
--target_layer_idx 2 \
--adv_steps_list "[0]" \
--adv_lr 1 \
--n_patch_num 1 \
--l_patch_num 0 \
--epsilon_list "[0/255, 16/255, 32/255, 64/255]" \
--image_token_ratio_list "[1.0, 0.7, 0.5, 0.3, 0.2, 0.1]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"

# mme llava fastv
python llava_eval_mme.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[0]" \
--adv_lr 1 \
--n_patch_num 1 \
--l_patch_num 0 \
--epsilon_list "[0/255, 16/255, 32/255, 64/255]" \
--image_token_ratio_list "[1.0, 0.7, 0.5, 0.3, 0.2, 0.1]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"

# mme llava pdrop
python llava_eval_mme.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[8, 16]" \
--target_layer_idx 2 \
--adv_steps_list "[0]" \
--adv_lr 1 \
--n_patch_num 1 \
--l_patch_num 0 \
--epsilon_list "[0/255, 16/255, 32/255, 64/255]" \
--image_token_ratio_list "[1.0, 0.7, 0.5, 0.3, 0.2, 0.1]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"

# mme llava sparse
python llava_eval_mme.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2, 8, 16]" \
--target_layer_idx 2 \
--adv_steps_list "[0]" \
--adv_lr 1 \
--n_patch_num 1 \
--l_patch_num 0 \
--epsilon_list "[0/255, 16/255, 32/255, 64/255]" \
--image_token_ratio_list "[1.0, 0.7, 0.5, 0.3, 0.2, 0.1]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"

# vqa llava fastv
python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[0]" \
--adv_lr 1 \
--n_patch_num 1 \
--l_patch_num 0 \
--epsilon_list "[0/255, 16/255, 32/255, 64/255]" \
--image_token_ratio_list "[1.0, 0.7, 0.5, 0.3, 0.2, 0.1]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"

# vqa llava pdrop
python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[8, 16]" \
--target_layer_idx 2 \
--adv_steps_list "[0]" \
--adv_lr 1 \
--n_patch_num 1 \
--l_patch_num 0 \
--epsilon_list "[0/255, 16/255, 32/255, 64/255]" \
--image_token_ratio_list "[1.0, 0.7, 0.5, 0.3, 0.2, 0.1]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"

# vqa llava sparse
python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2, 8, 16]" \
--target_layer_idx 2 \
--adv_steps_list "[0]" \
--adv_lr 1 \
--n_patch_num 1 \
--l_patch_num 0 \
--epsilon_list "[0/255, 16/255, 32/255, 64/255]" \
--image_token_ratio_list "[1.0, 0.7, 0.5, 0.3, 0.2, 0.1]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"



# pope llava fastv
python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[0]" \
--adv_lr 1 \
--n_patch_num 1 \
--l_patch_num 0 \
--epsilon_list "[0/255, 16/255, 32/255, 64/255]" \
--image_token_ratio_list "[1.0, 0.7, 0.5, 0.3, 0.2, 0.1]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"

# pope llava pdrop
python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[8, 16]" \
--target_layer_idx 2 \
--adv_steps_list "[0]" \
--adv_lr 1 \
--n_patch_num 1 \
--l_patch_num 0 \
--epsilon_list "[0/255, 16/255, 32/255, 64/255]" \
--image_token_ratio_list "[1.0, 0.7, 0.5, 0.3, 0.2, 0.1]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"

# pope llava sparse
python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2, 8, 16]" \
--target_layer_idx 2 \
--adv_steps_list "[0]" \
--adv_lr 1 \
--n_patch_num 1 \
--l_patch_num 0 \
--epsilon_list "[0/255, 16/255, 32/255, 64/255]" \
--image_token_ratio_list "[1.0, 0.7, 0.5, 0.3, 0.2, 0.1]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"

# mme llava fastv
python llava_eval_mme.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[0]" \
--adv_lr 1 \
--n_patch_num 1 \
--l_patch_num 0 \
--epsilon_list "[0/255, 16/255, 32/255, 64/255]" \
--image_token_ratio_list "[1.0, 0.7, 0.5, 0.3, 0.2, 0.1]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"

# mme llava pdrop
python llava_eval_mme.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[8, 16]" \
--target_layer_idx 2 \
--adv_steps_list "[0]" \
--adv_lr 1 \
--n_patch_num 1 \
--l_patch_num 0 \
--epsilon_list "[0/255, 16/255, 32/255, 64/255]" \
--image_token_ratio_list "[1.0, 0.7, 0.5, 0.3, 0.2, 0.1]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"

# mme llava sparse
python llava_eval_mme.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2, 8, 16]" \
--target_layer_idx 2 \
--adv_steps_list "[0]" \
--adv_lr 1 \
--n_patch_num 1 \
--l_patch_num 0 \
--epsilon_list "[0/255, 16/255, 32/255, 64/255]" \
--image_token_ratio_list "[1.0, 0.7, 0.5, 0.3, 0.2, 0.1]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"

# vqa llava fastv
python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[0]" \
--adv_lr 1 \
--n_patch_num 1 \
--l_patch_num 0 \
--epsilon_list "[0/255, 16/255, 32/255, 64/255]" \
--image_token_ratio_list "[1.0, 0.7, 0.5, 0.3, 0.2, 0.1]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"

# vqa llava pdrop
python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[8, 16]" \
--target_layer_idx 2 \
--adv_steps_list "[0]" \
--adv_lr 1 \
--n_patch_num 1 \
--l_patch_num 0 \
--epsilon_list "[0/255, 16/255, 32/255, 64/255]" \
--image_token_ratio_list "[1.0, 0.7, 0.5, 0.3, 0.2, 0.1]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"

# vqa llava sparse
python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2, 8, 16]" \
--target_layer_idx 2 \
--adv_steps_list "[0]" \
--adv_lr 1 \
--n_patch_num 1 \
--l_patch_num 0 \
--epsilon_list "[0/255, 16/255, 32/255, 64/255]" \
--image_token_ratio_list "[1.0, 0.7, 0.5, 0.3, 0.2, 0.1]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"