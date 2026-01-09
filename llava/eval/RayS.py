import torch
import numpy as np
import torch.nn.functional as F
from llava.eval.find_adv_util import first_token_id_llava
from llava.eval.utils import save_clip_tensor_as_image

def is_adv_fn(model, input_ids, image_tensor: torch.Tensor, layer_list, cp, most_or_least=True, target_token_ids=None, avoid_token_ids=None) -> bool:
    # img: (1,3,H,W) in [0,1]
    token_id = first_token_id_llava(
        model=model,
        input_ids=input_ids,
        image_tensor=image_tensor,
        max_new_tokens=1,
        layer_list=layer_list,
        cp=cp,
        most_or_least=most_or_least,
    )
    return (token_id in target_token_ids) and (token_id not in avoid_token_ids)

class RaySAttack:
    """
    A simplified RayS-style hard-label (decision-based) attack under L_infinity norm.

    It searches for an adversarial example x_adv = x0 + alpha * d,
    where d is a sign vector in {-1, +1}^D, and alpha is minimized via binary search.
    """

    def __init__(
        self,
        model=None,
        layer_list=[2,4,6,8],
        cp=1,
        max_queries=200,
        max_iters=200,             # number of direction refinement iterations
        alpha_init=0.25,   
        epsilon=0.5,                  # initial upper bound for alpha expansion
        device="cuda",
        target_token_ids=None,               # token IDs to move toward
        avoid_token_ids=None,                # token IDs to move away from
        bounds=(0.0, 1.0),
        diversity_prob=0
    ):
        self.lower, self.upper = bounds
        self.model = model
        self.layer_list = layer_list
        self.cp = cp
        self.max_queries = max_queries
        self.max_iters = max_iters
        self.alpha_init = alpha_init
        self.epsilon = epsilon
        self.device = device
        self.queries = 0
        self.target_token_ids = target_token_ids
        self.avoid_token_ids = avoid_token_ids
        self.diversity_prob = diversity_prob
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(3, 1, 1)

    def _clip(self, x):

        x.clamp(self.lower, self.upper)
        
        return x

    def _query(self, x, input_ids) -> bool:
        """
        [修改点 3] 引入输入多样性 (Input Diversity)
        在攻击过程中随机对输入图像进行变换，迫使攻击寻找更鲁棒的方向。
        """
        self.queries += 1
        
        x_input = (x.clone() - self.mean) / self.std

        res = bool(is_adv_fn(
            self.model, input_ids, x_input, self.layer_list, self.cp, True, self.target_token_ids, self.avoid_token_ids
        ))
        return res

    def _binary_search_alpha(self, x0, input_ids, d, lo, hi, steps=20):
        """
        Binary search the minimal alpha in [lo, hi] such that x0 + alpha*d is adversarial.
        Assumes that hi is adversarial and lo is not adversarial.
        """
        for _ in range(steps):
            if self.queries >= self.max_queries:
                break
            mid = (lo + hi) / 2.0
            xm = self._clip(x0 + mid * d)
            if self._query(xm, input_ids):
                hi = mid
            else:
                lo = mid
            # if hi<self.epsilon:
            #     break
        return hi

    def _find_initial_alpha(self, x0, text,d):
        """
        Find an initial alpha_high such that x0 + alpha_high*d is adversarial,
        by exponentially increasing alpha. Returns None if not found.
        """
        alpha = self.alpha_init
        # Ensure alpha does not trivially exceed the input bounds too aggressively
        for _ in range(30):
            if self.queries >= self.max_queries:
                return None
            x = self._clip(x0 + alpha * d)
            if self._query(x, text):
                return alpha
            alpha *= 2.0
            if alpha>1:
                return None
        return None

    def run(self, image_tensor: torch.Tensor, input_ids: torch.Tensor, logger=None):
        """
        Main entry. x0 is a torch tensor of shape (1,3,H,W), assumed to be in [0,1].
        Returns an adversarial example (or x0 if failed).
        """

        assert image_tensor.ndim == 4 and image_tensor.shape[0] == 1
        x0 = image_tensor.detach().clone().to(self.device).float()
        x0 = x0* self.std + self.mean
        x0 = self._clip(x0)

        # If x0 already satisfies the target decision, we can flip the goal externally.
        # Here we assume the goal is: make is_adv_fn(x) == True.
        if self._query(x0, input_ids):
            logger.print_log(f"[RayS] The clean input is already adversarial under the criterion. Queries: {self.queries}")
            # x_tmp=(x0 - self.mean) / self.std
            # save_clip_tensor_as_image(x_tmp.cpu(),'3.png')
            # return x0, True

        # Initialize a random sign direction d in {-1, +1}^D
        d = torch.sign(torch.randn_like(x0))
        d[d == 0] = 1.0

        best_alpha = float("inf")
        best_adv = None

        # 1) Find an initial adversarial point along the current ray
        alpha_hi = self._find_initial_alpha(x0, input_ids, d)
        if alpha_hi is None:
            logger.print_log(f"[RayS] Failed to find an initial adversarial point along the initial ray. Queries: {self.queries}")
            x_check = self._clip(x0 + self.epsilon * d)
            x_check=(x_check - self.mean) / self.std
            save_clip_tensor_as_image(x_check.cpu(),'3.png')
            return x_check, False
            
        alpha = self._binary_search_alpha(x0, input_ids, d, lo=0.0, hi=alpha_hi)
        best_alpha = alpha
        best_adv = self._clip(x0 + best_alpha * d)

        logger.print_log(f"[RayS] Init alpha = {best_alpha:.6f}, queries = {self.queries}")

        # if best_alpha <= self.epsilon:   
        #     logger.print_log(f"[RayS] Early Success! Alpha {best_alpha:.4f} <= {self.epsilon}, Queries: {self.queries}")
        #     best_adv=(best_adv - self.mean) / self.std
        #     save_clip_tensor_as_image(best_adv.cpu(),'3.png')
        #     return best_adv, True

        # 2) Iteratively refine the direction by flipping signs
        #    This is a simplified coordinate-flip strategy.
        flat_d = d.view(-1)
        D = flat_d.numel()

        curr_level = 0
        curr_block_idx = 0

        for it in range(self.max_iters):
            if self.queries >= self.max_queries:
                break

            # Pick a block to flip (you can schedule block size as in the paper)
            # Here we use a simple decreasing block size schedule.
            # block = max(1, D // (2 ** (1 + it // 100)))
            # start = (it * block) % D
            # idx = torch.arange(start, min(start + block, D), device=self.device)

            # 1. 计算当前层级总共有多少块 (2^level)
            num_blocks_at_level = 2 ** curr_level

            # 2. 计算每一块的大小 (向上取整，防止除不尽漏掉像素)
            block_size = (D + num_blocks_at_level - 1) // num_blocks_at_level

            # 3. 计算当前块的起始和结束索引
            start = curr_block_idx * block_size
            end = min(start + block_size, D) # 确保不越界

            # 生成需要翻转的索引
            if start >= D: 
                curr_block_idx = 0
                curr_level += 1
                continue
            idx = torch.arange(start, end, device=self.device)

            #随机反转
            # n_flip = end-start #max(1, D // (2 ** (1 + it // 500)))
            # idx = torch.randperm(D, device=self.device)[:n_flip]

            # Try flipping this block
            d_try_flat = flat_d.clone()
            d_try_flat[idx] *= -1.0
            d_try = d_try_flat.view_as(d)

            # Check Phase (Fast Check)
            x_check = self._clip(x0 + best_alpha * d_try)
            if self._query(x_check, input_ids):
                alpha_try = self._binary_search_alpha(x0, input_ids, d_try, lo=0.0, hi=best_alpha)

                if alpha_try < best_alpha:
                    best_alpha = alpha_try
                    best_adv = self._clip(x0 + best_alpha * d_try)
                    flat_d = d_try_flat 
                    d = d_try 

                    logger.print_log(f"[RayS] iter={it} improved alpha={best_alpha:.4f} query times={self.queries}")
                    
                    # 循环中一旦达标，立即算作成功返回
                    if best_alpha <= self.epsilon:
                        logger.print_log(f"[RayS] Success! Alpha reached {best_alpha:.4f}, Queries: {self.queries}")
                        best_adv = self._clip(x0 + self.epsilon * d_try)
                        best_adv=(best_adv - self.mean) / self.std
                        save_clip_tensor_as_image(best_adv.cpu(),'3.png')
                        return best_adv, True

            # # --- [新增] 更新状态，准备下一次迭代 ---
            # # 如果这层的所有块都遍历完了，进入下一层（块变得更小更细）
            curr_block_idx += 1
            if curr_block_idx >= num_blocks_at_level:
                curr_block_idx = 0
                curr_level += 1
            # # -----------------------------------

        # [修改] 循环结束后的最终审判
        # 虽然我们找到了 best_adv，但如果它的距离依然很大，这不算“攻击成功”
        is_success = (best_alpha <= self.epsilon)
        
        status = "SUCCESS" if is_success else "FAILED"
        logger.print_log(f"[RayS] Finished. Result: {status}. Best Alpha: {best_alpha:.4f} (Threshold: {self.epsilon}), Queries: {self.queries}, iter:{it}")

        if is_success==False:
            best_adv = self._clip(x0 + self.epsilon * d_try)
        best_adv=(best_adv - self.mean) / self.std
        save_clip_tensor_as_image(best_adv.cpu(),'3.png')
        return best_adv, is_success

