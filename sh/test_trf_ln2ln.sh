python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--epsilon_list "[128/255]" \
--adv_path "['../token-compress-pyramid/saved_data/llava-v1.6-vicuna-7b_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.15_0.15_64_100_1.0_2_False_True_3_trf_True_113_True_20.0_20.0_5.0_5.0_10.0_1.0_1.0_40_0.8_False.pt']"

python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2, 6, 15]" \
--target_layer_idx 2 \
--epsilon_list "[128/255]" \
--adv_path "['../token-compress-pyramid/saved_data/llava-v1.6-vicuna-7b_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.15_0.15_64_100_1.0_2_False_True_3_trf_True_113_True_20.0_20.0_5.0_5.0_10.0_1.0_1.0_40_0.8_True.pt']"

python llava_eval_mme.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--epsilon_list "[128/255]" \
--adv_path "['../token-compress-pyramid/saved_data/llava-v1.6-vicuna-7b_qwen25_7b_llava_mme_filter_rule1_adv_image_0.15_0.15_64_100_1.0_2_False_True_3_trf_True_113_True_20.0_20.0_5.0_5.0_10.0_1.0_1.0_40_0.8_False.pt']"

python llava_eval_mme.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2, 6, 15]" \
--target_layer_idx 2 \
--epsilon_list "[128/255]" \
--adv_path "['../token-compress-pyramid/saved_data/llava-v1.6-vicuna-7b_qwen25_7b_llava_mme_filter_rule1_adv_image_0.15_0.15_64_100_1.0_2_False_True_3_trf_True_113_True_20.0_20.0_5.0_5.0_10.0_1.0_1.0_40_0.8_True.pt']"

python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2]" \
--target_layer_idx 2 \
--epsilon_list "[128/255]" \
--adv_path "['../token-compress-pyramid/saved_data/llava-v1.6-vicuna-7b_qwen25_3b_7b_llava_vqa_filter_rule1_adv_image_0.15_0.15_64_100_1.0_2_False_True_3_trf_True_113_True_20.0_20.0_5.0_5.0_10.0_1.0_1.0_20_0.75_False.pt']"

python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2, 6, 15]" \
--target_layer_idx 2 \
--epsilon_list "[128/255]" \
--adv_path "['../token-compress-pyramid/saved_data/llava-v1.6-vicuna-7b_qwen25_3b_7b_llava_vqa_filter_rule1_adv_image_0.15_0.15_64_100_1.0_2_False_True_3_trf_True_113_True_20.0_20.0_5.0_5.0_10.0_1.0_1.0_20_0.75_True.pt']"