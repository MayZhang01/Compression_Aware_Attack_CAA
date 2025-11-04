python llava_eval_mme.py \
--model_path "../../model_weight/llava-v1.5-7b" \
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


python llava_eval_mme.py \
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
--rlt_path "../qwen2vl_compress/saved_data/Qwen2.5-VL-3B-Instruct_qwen25_3b_llava_mme_filter_rule1_adv_image_0.1_0.1_128_100_1.0_2_False_True_1_trf_False_307.pt"