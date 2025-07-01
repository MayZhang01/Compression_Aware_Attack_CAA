import logging
import time
import os
import torch
import torch.nn.functional as F
import json
class Logger:
    def __init__(self):
        # 创建以时间戳命名的日志文件名
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")#
        self.log_filename = f"{self.timestamp}.log"

        # 创建日志目录（可选）
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, self.log_filename)

        # 配置日志系统
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,  # 或 logging.DEBUG / ERROR 等
            format="%(message)s"#%(asctime)s - %(levelname)s - 
        )

    def print_log(*args):
        message = ' '.join(str(arg) for arg in args[1:])
        # for i, arg in enumerate(args):
        #     print(f"Argument {i + 1}: {arg}")
        print(message)
        logging.info(message)
        # logging.warning("这是一个警告")
        # logging.error("这是一个错误")

def loss_norm(ground_logits, pred_logits, eps=1e-8):
    # soft_targets = F.softmax(ground_logits, dim=1)  # shape: (batch_size, num_classes)
    # soft_preds = F.softmax(pred_logits, dim=1)
    # soft_preds = soft_preds.clamp(min=eps)

    # loss = -torch.sum(soft_targets * torch.log(soft_preds), dim=1)  # shape: (batch_size,)
    loss = torch.norm(ground_logits-pred_logits)#, dim=2)/pred_logits.shape[2]
    mean_loss = loss.mean()
    return mean_loss

def load_json(file_path):
    with open(file_path, 'r') as file:
        dict_list=json.load(file)
    return dict_list

def load_jsonl(file_path):
    dict_list=[]
    with open(file_path, "r") as f:
        for line in f:
            dict_list.append(json.loads(line))
    return dict_list

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def save_jsonl(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + "\n")