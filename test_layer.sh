
# llava pope
# llava-v1.5-7b_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.2_0.4_64_150_1.0_2_False_True_1_ca_False_311_False_20_20_10_5_5_10_10_40_0.8.pt
python eval_load.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--res_file "saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.2_0.4_64_150_1.0_2_False_True_1_ca_False_311_False_20_20_10_5_5_10_10_40_0.8.pt"

# llava-v1.5-7b_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.2_0.4_32_150_1.0_8_False_True_1_ca_False_157_False_20_20_10_5_5_10_10_40_0.8.pt
# python eval_load.py \
# --model_path "../../model_weight/llava-v1.5-7b" \
# --res_file "saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.2_0.4_32_150_1.0_8_False_True_1_ca_False_157_False_20_20_10_5_5_10_10_40_0.8.pt"

# llava vqa
# llava-v1.5-7b_qwen25_3b_7b_llava_vqa_filter_rule1_adv_image_0.1_0.2_32_100_1.0_2_False_True_1_ca_False_311_False_20_20_10_5_5_10_10_40_0.8.pt
python eval_load.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--res_file "saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_vqa_filter_rule1_adv_image_0.1_0.2_32_100_1.0_2_False_True_1_ca_False_311_False_20_20_10_5_5_10_10_40_0.8.pt"

# llava-v1.5-7b_qwen25_3b_7b_llava_vqa_filter_rule1_adv_image_0.1_0.2_32_100_1.0_8_False_True_1_ca_False_311_False_20_20_10_5_5_10_10_20_0.75.pt
python eval_load.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--res_file "saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_vqa_filter_rule1_adv_image_0.1_0.2_32_100_1.0_8_False_True_1_ca_False_311_False_20_20_10_5_5_10_10_20_0.75.pt"

# llava mme
# llava-v1.5-7b_qwen25_7b_llava_mme_filter_rule1_adv_image_0.2_0.4_32_150_1.0_2_False_True_1_ca_False_157_False_20_20_10_5_5_10_10_40_0.8.pt
python eval_load.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--res_file "saved_data/llava-v1.5-7b_qwen25_7b_llava_mme_filter_rule1_adv_image_0.2_0.4_32_150_1.0_2_False_True_1_ca_False_157_False_20_20_10_5_5_10_10_40_0.8.pt"

# llava-v1.5-7b_qwen25_7b_llava_mme_filter_rule1_adv_image_0.2_0.4_32_150_1.0_8_False_True_1_ca_False_311_False_20_20_10_5_5_10_10_40_0.8.pt
python eval_load.py \
--model_path "../../model_weight/llava-v1.5-7b" \
--res_file "saved_data/llava-v1.5-7b_qwen25_7b_llava_mme_filter_rule1_adv_image_0.2_0.4_32_150_1.0_8_False_True_1_ca_False_311_False_20_20_10_5_5_10_10_40_0.8.pt"
