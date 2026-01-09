python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--epsilon_list "[128/255]" \
--adv_path "['../qwen2vl_compress/saved_data/Qwen2.5-VL-7B-Instruct_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.25_0.25_64_150_1.0_20_False_True_1_trf_full_True_157_True_20.0_20.0_5.0_5.0_10.0_10.0_10.0_80_0.8_False_1.pt']"

python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2,8]" \
--epsilon_list "[128/255]" \
--adv_path "['../qwen2vl_compress/saved_data/Qwen2.5-VL-7B-Instruct_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.25_0.25_64_150_1.0_20_False_True_1_trf_full_True_157_True_20.0_20.0_5.0_5.0_10.0_10.0_10.0_80_0.8_False_1.pt']"

python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2,8,16]" \
--epsilon_list "[128/255]" \
--adv_path "['../qwen2vl_compress/saved_data/Qwen2.5-VL-7B-Instruct_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.25_0.25_64_150_1.0_20_False_True_1_trf_full_True_157_True_20.0_20.0_5.0_5.0_10.0_10.0_10.0_80_0.8_False_1.pt']"


python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2]" \
--epsilon_list "[16/255]" \
--adv_path "['../qwen2vl_compress/saved_data/Qwen2.5-VL-7B-Instruct_qwen25_3b_7b_llava_vqa_filter_rule1_adv_image_0.15_0.15_64_150_1.0_20_False_True_1_trf_full_True_157_True_20.0_20.0_5.0_5.0_10.0_10.0_10.0_60_0.78_False_1.pt']"

python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2,8]" \
--epsilon_list "[16/255]" \
--adv_path "['../qwen2vl_compress/saved_data/Qwen2.5-VL-7B-Instruct_qwen25_3b_7b_llava_vqa_filter_rule1_adv_image_0.15_0.15_64_150_1.0_20_False_True_1_trf_full_True_157_True_20.0_20.0_5.0_5.0_10.0_10.0_10.0_60_0.78_False_1.pt']"

python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--layer_list "[2,8,16]" \
--epsilon_list "[16/255]" \
--adv_path "['../qwen2vl_compress/saved_data/Qwen2.5-VL-7B-Instruct_qwen25_3b_7b_llava_vqa_filter_rule1_adv_image_0.15_0.15_64_150_1.0_20_False_True_1_trf_full_True_157_True_20.0_20.0_5.0_5.0_10.0_10.0_10.0_60_0.78_False_1.pt']"
