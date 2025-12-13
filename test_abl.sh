# qwen2.5-vl-7b 全图攻击
python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
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
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
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
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
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
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[150]" \
--adv_lr 1 \
--n_patch_num 0.1 \
--l_patch_num 0.3 \
--epsilon_list "[32/255]" \
--re_scale "False" \
--atk "ca"

python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
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
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[150]" \
--adv_lr 1 \
--n_patch_num 0.5 \
--l_patch_num 0.7 \
--epsilon_list "[32/255]" \
--re_scale "False" \
--atk "ca"


# qwen2.5-vl-7b 全图攻击
python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[100]" \
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
python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[100]" \
--adv_lr 1 \
--n_patch_num 0.1 \
--l_patch_num 0.2 \
--epsilon_list "[32/255]" \
--bpr_ths 0.1 \
--cut_num 1 \
--re_scale "False" \
--atk "ca" \
--patch_or_noise "False" \
--most_or_least "True" \
--w_bpr_l 0 --w_bpr_p 0

# qwen2.5-vl-7b 无hierarchical least
python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[100]" \
--adv_lr 1 \
--n_patch_num 0.2 \
--l_patch_num 0 \
--epsilon_list "[32/255]" \
--re_scale "False" \
--atk "ca" --stop_teps 30

python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[150]" \
--adv_lr 1 \
--n_patch_num 0.1 \
--l_patch_num 0.3 \
--epsilon_list "[32/255]" \
--re_scale "False" \
--atk "ca"

python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[150]" \
--adv_lr 1 \
--n_patch_num 0.3 \
--l_patch_num 0.5 \
--epsilon_list "[32/255]" \
--re_scale "False" \
--atk "ca"

python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[150]" \
--adv_lr 1 \
--n_patch_num 0.5 \
--l_patch_num 0.7 \
--epsilon_list "[32/255]" \
--re_scale "False" \
--atk "ca"