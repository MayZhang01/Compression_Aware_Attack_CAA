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
import random
from llava.eval.find_adv import opt_adv_pro, opt_adv_image_token, opt_adv_image_text_pro
from llava.eval.find_adv_patch import opt_adv_image_patch
from llava.eval.utils import Logger, extract_name_from_path
from llava.eval.find_adv_util import judge_adv
import numpy as np
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator

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
        gd = line["ground_truth"]
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

        return input_ids, image_tensor, image.size, gd

    def __len__(self):
        return len(self.questions)
class Counter_acc:
    def __init__(self, cp_list=[], evaluator=None):
        self.score={'ori':0}
        self.least_inter_ratio, self.patch_inter_ratio=[],[]
        self.rej, self.nonsense={}, {}
        self.total=0
        self.evaluator=evaluator
        for cp in cp_list:
            self.add_key(str(cp))
            # self.score[str(cp)]=0
            # self.rej[str(cp)]=0
            # self.nonsense[str(cp)]=0
    
    def add_key(self, key):
        self.score[key]=0
        self.rej[key]=0
        self.nonsense[key]=0
    
    def add_rlt(self, gt, rlt):
        self.total+=1
        for key in rlt:
            if key in ['least_inter_ratio', 'patch_inter_ratio']:
                continue
            if key not in self.score:
                self.add_key(key)
            tmp_s=self.evaluator.eval_pred_per_sample(rlt[key], gt)
            self.score[key]+=tmp_s
            rlt[key]=(tmp_s, rlt[key], )
                
        if 'least_inter_ratio' in rlt:
            self.least_inter_ratio.append(rlt['least_inter_ratio'])
            self.patch_inter_ratio.append(rlt['patch_inter_ratio'])
        return rlt
    
    def get_metric(self, logger=None):
        metric={}
        m_list=["ACC"]
        for key in self.score:
            metric[key]={}
            metric[key]['ACC']=round(self.score[key]/self.total,4)
            
        if logger is not None:
            logger.print_log('', '\t'.join(m for m in m_list))
            logger.print_log('least_inter_ratio', np.mean(self.least_inter_ratio))
            logger.print_log('patch_inter_ratio', np.mean(self.patch_inter_ratio))
            for key in metric:
                logger.print_log(key,  metric[key]['ACC'])
        return metric

class Counter:
    def __init__(self, cp_list=[]):
        self.TP, self.FP, self.TN, self.FN={'ori':0}, {'ori':0}, {'ori':0}, {'ori':0}
        self.least_inter_ratio, self.patch_inter_ratio=[],[]
        self.rej, self.nonsense={}, {}
        self.total=0
        for cp in cp_list:
            self.add_key(str(cp))
            # self.TP[str(cp)]=0
            # self.FP[str(cp)]=0
            # self.TN[str(cp)]=0
            # self.FN[str(cp)]=0
            # self.rej[str(cp)]=0
            # self.nonsense[str(cp)]=0
    
    def get_metric(self, logger=None):
        metric={}
        m_list=["TPR", "FNR","TNR", "FPR", "ACC", "Rej", "Nos"]
        for key in self.TP:
            metric[key]={}
            total=self.TP[key]+self.FN[key]+self.TN[key]+self.FP[key]
            metric[key]['TPR']=round(self.TP[key]/(self.TP[key]+self.FN[key]+1e-8),4)
            metric[key]['FNR']=round(self.FN[key]/(self.TP[key]+self.FN[key]+1e-8),4)
            metric[key]['TNR']=round(self.TN[key]/(self.TN[key]+self.FP[key]+1e-8),4)
            metric[key]['FPR']=round(self.FP[key]/(self.TN[key]+self.FP[key]+1e-8),4)
            metric[key]['ACC']=round((self.TP[key]+self.TN[key])/(total+1e-8),4)
            if key !='ori':
                metric[key]['Rej']=round((self.rej[key])/(total+1e-8),4)
                metric[key]['Nos']=round((self.nonsense[key])/(total+1e-8),4)
        if logger is not None:
            logger.print_log('', '\t'.join(m for m in m_list))
            logger.print_log('least_inter_ratio', np.mean(self.least_inter_ratio))
            logger.print_log('patch_inter_ratio', np.mean(self.patch_inter_ratio))
            for key in metric:
                if key =='ori':
                    logger.print_log(key, metric[key]['TPR'], metric[key]['FNR'], metric[key]['TNR'], metric[key]['FPR'], metric[key]['ACC'])
                else:
                    logger.print_log(key, metric[key]['TPR'], metric[key]['FNR'], metric[key]['TNR'], metric[key]['FPR'], metric[key]['ACC'], metric[key]['Rej'], metric[key]['Nos'])
        return metric

    def add_key(self, key):
        self.TP[key]=0
        self.FP[key]=0
        self.TN[key]=0
        self.FN[key]=0
        self.rej[key]=0
        self.nonsense[key]=0
            
    def add_rlt(self, gt, rlt):
        self.total+=1
        if gt=='Yes':
            for key in rlt:
                if key in ['least_inter_ratio', 'patch_inter_ratio']:
                    continue
                if key not in self.TP:
                    self.add_key(key)
                if judge_adv(rlt[key], gt):
                    self.TP[key]+=1
                else:
                    self.FN[key]+=1
        else:
            for key in rlt:
                if key in ['least_inter_ratio', 'patch_inter_ratio']:
                    continue
                if key not in self.TP:
                    self.add_key(key)
                if judge_adv(rlt[key], gt):
                    self.TN[key]+=1
                else:
                    self.FP[key]+=1
        for key in rlt:
            if key in ['least_inter_ratio', 'patch_inter_ratio']:
                continue
            if key=='ori':
                continue
            if rlt[key]=='':
                self.rej[key]+=1
            elif (judge_adv(rlt[key], 'Yes') or judge_adv(rlt[key], 'No'))==False:
                self.nonsense[key]+=1
        
        if 'least_inter_ratio' in rlt:
            self.least_inter_ratio.append(rlt['least_inter_ratio'])
            self.patch_inter_ratio.append(rlt['patch_inter_ratio'])


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, gd = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes, gd


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4, conv_mode=''):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, conv_mode)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader