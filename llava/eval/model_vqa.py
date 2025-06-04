import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
import random
from llava.eval.find_adv import opt_adv_pro, judge_adv


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if args.layer_list is not None:
        pdrop_infer = True  # whether to use pdrop infer
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        args.model_base, 
        model_name,
        pdrop_infer
    )

    model_class_name = type(model).__name__

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    random.seed(124)
    random.shuffle(questions)
    questions = questions[0:20]
    # questions = questions[150:200]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # answers_file = os.path.expanduser(args.answers_file)
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # ans_file = open(answers_file, "w")
    count, count2=0, 0
    total=0
    succ_list=[]
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]
        image_tensor=image_tensor.unsqueeze(0).cuda()
        image_sizes = image.size
        with torch.no_grad():
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                new_input_embeds,
                _,  
            ) = model.prepare_inputs_labels_for_multimodal_pdrop(       
                input_ids, position_ids=None, attention_mask=None, past_key_values=None, labels=None,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True), 
                image_sizes=image_sizes
            )
            ori_prompt_len=model.model.prompt_len[0]
        
        adv_input_embeds, adv_num, flag=opt_adv_pro(
            model, new_input_embeds, position_ids, attention_mask,
            image_shape=576,        
            token_length_list=[new_input_embeds.shape[1]],  
            pre_prompt_length_list=[torch.where(input_ids==-200)[1][0].item()], 
            target_layer_idx=1, 
            adv_steps=50, #30
            adv_lr=2e-3,
            stop_ths=[0.55, 0.5, 0.0005],
            alpha=1
        )

        if flag:
            total+=1

        with torch.inference_mode():
            if model_class_name == "LlavaLlamaForCausalLM_PDrop":
                model.model.layer_list = [2]
                model.model.image_token_ratio_list = [1]
                model.model.image_token_ratio_list.insert(0, 1)
            model.model.prompt_len[0]=ori_prompt_len+adv_num
            adv_output_ids = model.generate(
                # input_ids,
                # images=image_tensor.unsqueeze(0).half().cuda(),
                inputs_embeds=adv_input_embeds,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                # temperature=args.temperature,
                # top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )

        adv_outputs = tokenizer.batch_decode(adv_output_ids, skip_special_tokens=True)[0].strip()

        
        with torch.inference_mode():
            if model_class_name == "LlavaLlamaForCausalLM_PDrop":
                model.model.layer_list = [2]
                model.model.image_token_ratio_list = [0.3]
                model.model.image_token_ratio_list.insert(0, 1.0)
            model.model.prompt_len[0]=ori_prompt_len+adv_num
            adv_output_ids2 = model.generate(
                # input_ids,
                # images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                inputs_embeds=adv_input_embeds,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                # temperature=args.temperature,
                # top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )
        adv_outputs2 = tokenizer.batch_decode(adv_output_ids2, skip_special_tokens=True)[0].strip()

        with torch.inference_mode():
            if model_class_name == "LlavaLlamaForCausalLM_PDrop":
                model.model.layer_list = []
                model.model.image_token_ratio_list = []
                # model.model.image_token_ratio_list.insert(0, 1.0)
            model.model.prompt_len[0]=ori_prompt_len
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                # inputs_embeds=new_input_embeds,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                # temperature=args.temperature,
                # top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(idx, ':', cur_prompt)
        print(outputs, '---', adv_outputs, '---', adv_outputs2)
        if judge_adv(output_nc=outputs, output_c=adv_outputs):# or adv_outputs.find(outputs.lower())>-1
            count+=1
        if judge_adv(output_nc=outputs, output_c=adv_outputs2):# or adv_outputs.find(outputs.lower())>-1
            count2+=1
        else:
            print()
            succ_list.append(idx)
    print(1-count/total, 1-count2/total, total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
