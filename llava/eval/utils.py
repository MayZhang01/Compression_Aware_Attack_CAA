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

def intersection_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute the intersection ratio between two integer tensors after sorting.
    Intersection is defined as the number of common elements (set-wise), 
    divided by the total number of unique elements (from either tensor).

    Args:
        a (torch.Tensor): First tensor, must be 1D or flattenable
        b (torch.Tensor): Second tensor, same shape as a

    Returns:
        float: Ratio of intersection count to total number of elements in a
    """
    # Flatten both tensors to 1D
    # a = a.flatten()
    # b = b.flatten()
    
    # # Sort both tensors
    # a_sorted, _ = torch.sort(a)
    # b_sorted, _ = torch.sort(b)

    # Compute set-wise intersection
    intersect = torch.isin(a, b)
    num_match = intersect.sum().item()
    total = min(a.numel(),b.numel())

    # Compute ratio
    ratio = num_match / total
    return ratio

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

def add_square_patch(image, top_left, bottom_right=None, p=None, patch_value=None):
    """
    Add a square patch to the input image.
    
    Args:
        image: torch.Tensor, shape [1, C, H, W]
            The input image tensor.
        p: float
            The proportion of the image area that the patch should occupy (between 0 and 1).
        top_left: tuple (x, y)
            The coordinates of the top-left corner of the patch.
        patch_value: float or tensor
            The value to fill the patch with (default: 1.0).
    
    Returns:
        torch.Tensor
            The image tensor with the patch applied.
    """
    if bottom_right is None and p is not None:
        assert image.dim() == 4 and image.shape[0] == 1, "image must be of shape [1, C, H, W]"
        _, C, H, W = image.shape
        
        # Compute the total area of the image
        total_area = H * W
        # Compute the area of the patch based on the proportion
        patch_area = total_area * p
        # Compute the side length of the square patch
        side_len = int(round(patch_area ** 0.5))
        
        x0, y0 = top_left
        # Make sure the patch does not go out of bounds
        x1 = min(x0 + side_len, H)
        y1 = min(y0 + side_len, W)
    else:
        x0, y0 = top_left
        x1, y1 = bottom_right
        
    # Create a copy of the image to avoid modifying the original
    image_with_patch = image.clone()
    
    # Fill the patch area with the specified value
    if patch_value is None:
        patch_value=image.min()
    image_with_patch[:, :, x0:x1, y0:y1] = patch_value
    
    return image_with_patch