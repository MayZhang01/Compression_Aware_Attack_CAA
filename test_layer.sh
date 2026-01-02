
# # llava pope
python eval_load.py \
--model_path "../../model_weight/llava-v1.5-7b" --benign_flag "False" \
--res_file "saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.2_0.4_64_150_1.0_2_False_True_1_ca_False_311_False_20_20_10_5_5_10_10_40_0.8.pt"
python eval_load.py \
--model_path "../../model_weight/llava-v1.5-7b" --benign_flag "False" \
--res_file "saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.2_0.4_32_150_1.0_2_False_True_1_ca_False_311_False_20_20_10_5_5_10_10_40_0.8.pt"

python eval_load.py \
--model_path "../../model_weight/llava-v1.5-7b" --benign_flag "True" \
--res_file "saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.2_0.4_32_150_1.0_8_False_True_1_ca_False_157_False_20_20_10_5_5_10_10_40_0.8.pt"

# # llava vqa
python eval_load.py \
--model_path "../../model_weight/llava-v1.5-7b" --benign_flag "True" \
--res_file "saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_vqa_filter_rule1_adv_image_0.05_0.1_32_100_1.0_2_False_True_1_ca_False_113_False_20.0_20.0_5.0_5.0_10.0_5.0_5.0_10.0_0.65.pt"

python eval_load.py \
--model_path "../../model_weight/llava-v1.5-7b" --benign_flag "True" \
--res_file "saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_vqa_filter_rule1_adv_image_0.1_0.2_32_100_1.0_8_False_True_1_ca_False_113_False_20.0_20.0_5.0_5.0_10.0_5.0_5.0_10.0_0.75_False.pt"

# # llava mme
python eval_load.py \
--model_path "../../model_weight/llava-v1.5-7b" --benign_flag "True" \
--res_file "saved_data/llava-v1.5-7b_qwen25_7b_llava_mme_filter_rule1_adv_image_0.2_0.4_32_150_1.0_2_False_True_1_ca_False_157_False_20_20_10_5_5_10_10_40_0.8.pt"

python eval_load.py \
--model_path "../../model_weight/llava-v1.5-7b" --benign_flag "True" \
--res_file "saved_data/llava-v1.5-7b_qwen25_7b_llava_mme_filter_rule1_adv_image_0.2_0.4_32_150_1.0_8_False_True_1_ca_False_311_False_20_20_10_5_5_10_10_40_0.8.pt"
