python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[100,0]" \
--adv_lr 1 \
--n_patch_num 0.2 \
--l_patch_num 0.4 \
--epsilon_list "[32/255, 16/255, 64/255]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"

python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[8]" \
--target_layer_idx 8 \
--adv_steps_list "[100,0]" \
--adv_lr 1 \
--n_patch_num 0.2 \
--l_patch_num 0.4 \
--epsilon_list "[32/255, 16/255, 64/255]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"


python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[100,0]" \
--adv_lr 1 \
--n_patch_num 0.1 \
--l_patch_num 0.3 \
--epsilon_list "[32/255, 16/255, 64/255]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"

python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[8]" \
--target_layer_idx 8 \
--adv_steps_list "[100,0]" \
--adv_lr 1 \
--n_patch_num 0.1 \
--l_patch_num 0.3 \
--epsilon_list "[32/255, 16/255, 64/255]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"


python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[0]" \
--adv_lr 1 \
--n_patch_num 0.1 \
--l_patch_num 0.3 \
--epsilon_list "[0/255]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True" \
--rlt_path "../qwen2vl_compress/saved_data/Qwen2.5-VL-3B-Instruct_pope_filter_rule1_llava_qwen25_3b_adv_image_0.1_0.1_128_100_1.0_2_False_True_1_trf_False_311.pt"


# qwen2.5-vl-7b normal pope eval
python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[150]" \
--adv_lr 1 \
--n_patch_num 0.2 \
--l_patch_num 0.4 \
--epsilon_list "[32/255, 16/255, 64/255, ]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True"

# qwen2.5-vl-7b 全图攻击
python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[150]" \
--adv_lr 1 \
--n_patch_num 0.3 \
--l_patch_num 0.6 \
--epsilon_list "[32/255]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True" \
--stop_teps 10 --most_flag "True"

# qwen2.5-vl-7b 无BPR
python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[150]" \
--adv_lr 1 \
--n_patch_num 0.2 \
--l_patch_num 0.4 \
--epsilon_list "[32/255]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True" \
--w_bpr_l 0 --w_bpr_p 0

# qwen2.5-vl-7b 无hierarchical least
python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[150]" \
--adv_lr 1 \
--n_patch_num 0.4 \
--l_patch_num 0 \
--epsilon_list "[32/255]" \
--re_scale "False" \
--atk "ca" --stop_teps 30

python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[150]" \
--adv_lr 1 \
--n_patch_num 0.3 \
--l_patch_num 0.5 \
--epsilon_list "[32/255]" \
--re_scale "False" \
--atk "ca"

python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[150]" \
--adv_lr 1 \
--n_patch_num 0.5 \
--l_patch_num 0.7 \
--epsilon_list "[32/255]" \
--re_scale "False" \
--atk "ca"