import logging
import time
import os
import torch
import torch.nn.functional as F
import json
from PIL import Image
class Logger:
    def __init__(self, dir_name=None):
        # 创建以时间戳命名的日志文件名
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")#
        self.log_filename = f"{self.timestamp}.log"

        self.log_dir = "logs"
        if dir_name is not None:
            self.log_dir = os.path.join(self.log_dir, dir_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, self.log_filename)

        # # 配置日志系统
        # logging.basicConfig(
        #     filename=self.log_path,
        #     level=logging.INFO,  # 或 logging.DEBUG / ERROR 等
        #     format="%(message)s"#%(asctime)s - %(levelname)s - 
        # )

        # 创建独立的 logger 和 handler
        self.logger = logging.getLogger(self.log_path)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_path)
        handler.setFormatter(logging.Formatter("%(message)s"))
        # 防止重复添加 handler
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)

    def print_log(self, *args):
        message = '\t'.join(str(arg) for arg in args)
        # for i, arg in enumerate(args):
        #     print(f"Argument {i + 1}: {arg}")
        print(message)
        # logging.info(message)
        self.logger.info(message)
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
    ratio = num_match / (total+1e-3)
    return round(ratio,3)

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

def get_patch_mask(selected_indices, image_size=336, patch_size=14):
    """
    Generate a (image_size, image_size) mask where pixels corresponding to selected patch tokens are set to 1,
    and all other pixels are 0.

    Args:
        selected_indices (torch.Tensor): 1D tensor containing the indices of selected tokens (values in [0, num_patches-1]).
        image_size (int): Size of the input image (height and width). Default is 336.
        patch_size (int): Size of each patch (height and width). Default is 14.

    Returns:
        torch.Tensor: A 2D tensor mask of shape (image_size, image_size) with 1s in selected regions.
    """
    num_patches_per_row = image_size // patch_size

    # Initialize mask with zeros
    mask = torch.zeros((image_size, image_size), dtype=torch.float16)

    # Iterate over each selected token index
    for idx in selected_indices:
        # Compute the row and column index of the patch
        row = idx // num_patches_per_row
        col = idx % num_patches_per_row

        # Compute the pixel coordinates of the patch
        row_start = row * patch_size
        row_end = (row + 1) * patch_size
        col_start = col * patch_size
        col_end = (col + 1) * patch_size

        # Set the corresponding region in the mask to 1
        mask[row_start:row_end, col_start:col_end] = 1.0

    return mask


def intersect_topk_range(t1, t2, k1, k2):
    """
    Compute the intersection of top-k1 to top-k2 indices between two tensors.

    Args:
        t1 (Tensor): A tensor of shape (1, 576).
        t2 (Tensor): A tensor of shape (1, 576).
        k1 (int): Start of the top-k range (inclusive).
        k2 (int): End of the top-k range (inclusive).

    Returns:
        Tensor: A 1D tensor containing the indices that are common in the top-k1 to top-k2 
                range of both t1 and t2.
    """
    # Validate shapes
    assert t1.shape == t2.shape == (1, 576), "Both tensors must have shape (1, 576)"
    assert 0 <= k1 <= k2 <= t1.shape[1], "k1 and k2 must define a valid range"

    # Get top-k2 indices for each tensor (returns sorted in descending order of values)
    t1_topk = torch.topk(t1, k=k2, dim=1).indices[0]
    t2_topk = torch.topk(t2, k=k2, dim=1).indices[0]

    # Slice from k1 to k2 to get the desired rank range
    t1_range = t1_topk[k1:k2]
    t2_range = t2_topk[k1:k2]

    # Compute the intersection of the index sets and convert to a tensor
    intersection = torch.tensor(
        list(set(t1_range.tolist()) & set(t2_range.tolist())),
        dtype=torch.long
    )

    return intersection

def save_clip_tensor_as_image(tensor, filepath):
    """
    Save a tensor preprocessed by CLIP back as an image.

    Args:
        tensor (torch.Tensor): A tensor of shape (3, H, W) or (1, 3, H, W).
        filepath (str): Path to save the output image (e.g., 'output.png').
    """
    # If tensor has batch dimension, remove it
    tensor=tensor.cpu()
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # CLIP preprocessing normalization values
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    # Unnormalize
    tensor = tensor * std + mean

    # Clip to [0,1]
    tensor = torch.clamp(tensor, 0.0, 1.0)

    # Convert to numpy and scale to [0,255]
    np_image = tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy()

    # Convert to PIL and save
    img = Image.fromarray(np_image)
    img.save(filepath)

def extract_name_from_path(s):
    last_slash = s.rfind('/')
    last_dot = s.rfind('.')
    if 0 <= last_slash < last_dot:
        return s[last_slash + 1:last_dot]
    else:
        return None

def print_args(args, logger):
    
    key_dict={}
    for key, value in args.__dict__.items():
        if not key.startswith('__') and key not in key_dict:
            if key=='epsilon':
                key_dict[key]=int(args.epsilon*255)
            else:
                key_dict[key]=value
    for key, value in args.__class__.__dict__.items():
        if not key.startswith('__') and key not in key_dict:
            key_dict[key]=value
    for key in key_dict:
        logger.print_log(f"{key}: {key_dict[key]}")


def add_border(image, padding, value=0):
    # image: [C, H, W]
    C, H, W = image.shape
    # padding = (pad_width, pad_width, pad_width, pad_width)  # (left, right, top, bottom)
    # Add border to the image using constant padding
    image_with_border = F.pad(image, padding, mode='constant', value=value)
    # Create a border mask: 1 for border, 0 for original image
    mask = torch.zeros((H, W), dtype=torch.uint8, device=image.device)
    mask_padded = F.pad(mask, padding, mode='constant', value=1)  # 边框为1，原图为0
    eps = 1e-2
    zero_pos = (image_with_border.abs() < eps).any(dim=0)
    mask_padded[zero_pos] = 1
    return image_with_border, mask_padded

def crop_and_resize_image(image, cut_loc, cut_size):
    """
    Crop a region from one of the four corners of the image and resize it back to the original size.
    Args:
        image: torch.Tensor, shape [C, H, W]
        cut_loc: int, 0~3, 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
        cut_size: float, (0,1], the ratio of cropped region size to original size
    Returns:
        image_cropped_resized: torch.Tensor, shape [C, H, W]
    """
    if len(image.shape)==2:
        image=image.unsqueeze(0).float()
    C, H, W = image.shape
    crop_h = int(H * cut_size)
    crop_w = int(W * cut_size)
    # Ensure at least 1 pixel
    crop_h = max(1, crop_h)
    crop_w = max(1, crop_w)

    if cut_loc == 0:  # top-left
        h_start, w_start = 0, 0
    elif cut_loc == 1:  # top-right
        h_start, w_start = 0, W - crop_w
    elif cut_loc == 2:  # bottom-left
        h_start, w_start = H - crop_h, 0
    elif cut_loc == 3:  # bottom-right
        h_start, w_start = H - crop_h, W - crop_w
    else:
        raise ValueError("cut_loc must be 0, 1, 2, or 3.")

    h_end = h_start + crop_h
    w_end = w_start + crop_w

    cropped = image[:, h_start:h_end, w_start:w_end]
    # Resize back to original size
    cropped = cropped.unsqueeze(0)  # [1, C, crop_h, crop_w]
    resized = F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
    return resized.squeeze(0)  # [C, H, W]


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, process_images
def process_input(text, image_tensor, tokenizer, image_processor, model_config, conv_mode):
    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + text

    conv = conv_mode.copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if len(image_tensor.shape)==3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = process_images(image_tensor, image_processor, model_config)
    image_sizes=image_tensor.size
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).to(device='cuda')
    return input_ids, image_tensor, image_sizes