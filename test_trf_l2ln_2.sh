python llava_eval_pope.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2]" \
--epsilon_list "[128/255]" \
--adv_path "['../token-compress-pyramid/saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.25_0.25_64_150_1.0_20_False_True_3_trf_True_157_True_20.0_20.0_5.0_5.0_10.0_10.0_10.0_40_0.8_False.pt']"
# python llava_eval_pope.py \
# --model_path "../../model_weight/llava-v1.6-vicuna-7b" \
# --layer_list "[4]" \
# --epsilon_list "[128/255]" \
# --adv_path "['../token-compress-pyramid/saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.25_0.25_64_150_1.0_20_False_True_3_trf_True_157_True_20.0_20.0_5.0_5.0_10.0_10.0_10.0_40_0.8_False.pt']"
# python llava_eval_pope.py \
# --model_path "../../model_weight/llava-v1.6-vicuna-7b" \
# --layer_list "[6]" \
# --epsilon_list "[128/255]" \
# --adv_path "['../token-compress-pyramid/saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.25_0.25_64_150_1.0_20_False_True_3_trf_True_157_True_20.0_20.0_5.0_5.0_10.0_10.0_10.0_40_0.8_False.pt']"
# python llava_eval_pope.py \
# --model_path "../../model_weight/llava-v1.6-vicuna-7b" \
# --layer_list "[8]" \
# --epsilon_list "[128/255]" \
# --adv_path "['../token-compress-pyramid/saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.25_0.25_64_150_1.0_20_False_True_3_trf_True_157_True_20.0_20.0_5.0_5.0_10.0_10.0_10.0_40_0.8_False.pt']"
# python llava_eval_pope.py \
# --model_path "../../model_weight/llava-v1.6-vicuna-7b" \
# --layer_list "[10]" \
# --epsilon_list "[128/255]" \
# --adv_path "['../token-compress-pyramid/saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.25_0.25_64_150_1.0_20_False_True_3_trf_True_157_True_20.0_20.0_5.0_5.0_10.0_10.0_10.0_40_0.8_False.pt']"
# python llava_eval_pope.py \
# --model_path "../../model_weight/llava-v1.6-vicuna-7b" \
# --layer_list "[2,6]" \
# --epsilon_list "[128/255]" \
# --adv_path "['../token-compress-pyramid/saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.25_0.25_64_150_1.0_20_False_True_3_trf_True_157_True_20.0_20.0_5.0_5.0_10.0_10.0_10.0_40_0.8_False.pt']"
# python llava_eval_pope.py \
# --model_path "../../model_weight/llava-v1.6-vicuna-7b" \
# --layer_list "[2,8]" \
# --epsilon_list "[128/255]" \
# --adv_path "['../token-compress-pyramid/saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.25_0.25_64_150_1.0_20_False_True_3_trf_True_157_True_20.0_20.0_5.0_5.0_10.0_10.0_10.0_40_0.8_False.pt']"
# python llava_eval_pope.py \
# --model_path "../../model_weight/llava-v1.6-vicuna-7b" \
# --layer_list "[2,10]" \
# --epsilon_list "[128/255]" \
# --adv_path "['../token-compress-pyramid/saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.25_0.25_64_150_1.0_20_False_True_3_trf_True_157_True_20.0_20.0_5.0_5.0_10.0_10.0_10.0_40_0.8_False.pt']"
# python llava_eval_pope.py \
# --model_path "../../model_weight/llava-v1.6-vicuna-7b" \
# --layer_list "[8,16]" \
# --epsilon_list "[128/255]" \
# --adv_path "['../token-compress-pyramid/saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.25_0.25_64_150_1.0_20_False_True_3_trf_True_157_True_20.0_20.0_5.0_5.0_10.0_10.0_10.0_40_0.8_False.pt']"
# python llava_eval_pope.py \
# --model_path "../../model_weight/llava-v1.6-vicuna-7b" \
# --layer_list "[2,8,16]" \
# --epsilon_list "[128/255]" \
# --adv_path "['../token-compress-pyramid/saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.25_0.25_64_150_1.0_20_False_True_3_trf_True_157_True_20.0_20.0_5.0_5.0_10.0_10.0_10.0_40_0.8_False.pt']"
# python llava_eval_pope.py \
# --model_path "../../model_weight/llava-v1.6-vicuna-7b" \
# --layer_list "[8,16,24]" \
# --epsilon_list "[128/255]" \
# --adv_path "['../token-compress-pyramid/saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_pope_filter_rule1_adv_image_0.25_0.25_64_150_1.0_20_False_True_3_trf_True_157_True_20.0_20.0_5.0_5.0_10.0_10.0_10.0_40_0.8_False.pt']"


python llava_eval_vqa.py \
--model_path "../../model_weight/llava-v1.6-vicuna-7b" \
--layer_list "[2]" \
--epsilon_list "[16/255]" \
--adv_path "['../token-compress-pyramid/saved_data/llava-v1.5-7b_qwen25_3b_7b_llava_vqa_filter_rule1_adv_image_0.25_0.25_64_150_1.0_20_False_True_3_trf_True_157_True_20.0_20.0_5.0_5.0_10.0_1.0_1.0_20_0.75_False.pt']"