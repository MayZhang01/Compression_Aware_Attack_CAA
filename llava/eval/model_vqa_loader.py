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
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
from llava.eval.find_adv import opt_adv_pro
import random

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4, conv_mode=''):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, conv_mode)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

def judge_adv(output_c, output_nc):
    if output_c==output_nc:
        return True
    elif output_c.find(output_nc)>-1:# and len(output_c)<len(output_nc)*2:
        return True
    return False

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
    random.seed(123)
    random.shuffle(questions)
    questions = questions[0:20]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, conv_mode=args.conv_mode)
    count, count2=0, 0
    succ_list=[]
    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]
        # if idx not in [10001551, 10000073, 10002494, 10002462, 794 ]:
        #     continue
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        # input_ids = torch.cat([
        #     input_ids,
        #     torch.randint(0,tokenizer.vocab_size,(1,1),device=input_ids.device)
        # ], dim=1)
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
        
        adv_input_embeds, adv_num=opt_adv_pro(
            model, new_input_embeds, position_ids, attention_mask,
            image_shape=576,        
            token_length_list=[new_input_embeds.shape[1]],  
            pre_prompt_length_list=[torch.where(input_ids==-200)[1][0].item()], 
            target_layer_idx=1, 
            adv_steps=10, 
            adv_lr=1e-2
        )

        with torch.inference_mode():
            if model_class_name == "LlavaLlamaForCausalLM_PDrop":
                model.model.layer_list = [2]
                model.model.image_token_ratio_list = [1]
                model.model.image_token_ratio_list.insert(0, 1.0)
            model.model.prompt_len[0]=ori_prompt_len+adv_num
            adv_output_ids = model.generate(
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
        adv_outputs = tokenizer.batch_decode(adv_output_ids, skip_special_tokens=True)[0].strip()
        
        with torch.inference_mode():
            if model_class_name == "LlavaLlamaForCausalLM_PDrop":
                model.model.layer_list = [2]
                model.model.image_token_ratio_list = [0.1]
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
    print(1-count/len(questions), 1-count2/len(questions))
    for id in succ_list:
        print(id)

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
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--layer_list", type=str, default= None)
    parser.add_argument("--image_token_ratio_list", type=str, default= None)
    parser.add_argument("--pdrop_infer", action='store_true', help="use this to apply pdrop inference to ori llava-1.5")
    args = parser.parse_args()

    eval_model(args)
