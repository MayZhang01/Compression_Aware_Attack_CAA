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
--w_bpr_l 20 --w_bpr_p 20 --w_dir_l 5 --w_dir_p 5 --w_bpr_lp 10 --w_ori_l 0 --w_ori_p 0

python llava_eval_mme.py \
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
--w_bpr_l 20 --w_bpr_p 20 --w_dir_l 5 --w_dir_p 5 --w_bpr_lp 10 --w_ori_l 0 --w_ori_p 0

python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[150]" \
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
--w_bpr_l 20 --w_bpr_p 20 --w_dir_l 5 --w_dir_p 5 --w_bpr_lp 10 --w_ori_l 0 --w_ori_p 0


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
--w_bpr_l 20 --w_bpr_p 20 --w_dir_l 0 --w_dir_p 0 --w_bpr_lp 10 --w_ori_l 10 --w_ori_p 10

python llava_eval_mme.py \
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
--w_bpr_l 20 --w_bpr_p 20 --w_dir_l 0 --w_dir_p 0 --w_bpr_lp 10 --w_ori_l 10 --w_ori_p 10

python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--adv_steps_list "[150]" \
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
--w_bpr_l 20 --w_bpr_p 20 --w_dir_l 0 --w_dir_p 0 --w_bpr_lp 10 --w_ori_l 10 --w_ori_p 10