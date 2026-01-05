python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[100]" \
--adv_lr 1 \
--n_patch_num 0.2 \
--l_patch_num 0 \
--epsilon_list "[32/255]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "HJS" \
--patch_or_noise "False" \
--most_or_least "True"

python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[100]" \
--adv_lr 1 \
--n_patch_num 0.2 \
--l_patch_num 0 \
--epsilon_list "[32/255]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "HJS" \
--patch_or_noise "False" \
--most_or_least "True"

python llava_eval_mme.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[100]" \
--adv_lr 1 \
--n_patch_num 0.2 \
--l_patch_num 0 \
--epsilon_list "[32/255]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "HJS" \
--patch_or_noise "False" \
--most_or_least "True"

python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[100]" \
--adv_lr 1 \
--n_patch_num 0.2 \
--l_patch_num 0 \
--epsilon_list "[32/255]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "rays" \
--patch_or_noise "False" \
--most_or_least "True"

python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[100]" \
--adv_lr 1 \
--n_patch_num 0.2 \
--l_patch_num 0 \
--epsilon_list "[32/255]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "rays" \
--patch_or_noise "False" \
--most_or_least "True"

python llava_eval_mme.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[100]" \
--adv_lr 1 \
--n_patch_num 0.2 \
--l_patch_num 0 \
--epsilon_list "[32/255]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "rays" \
--patch_or_noise "False" \
--most_or_least "True"