import numpy as np
import torch
import foolbox as fb
import eagerpy as ep
from llava.eval.utils import save_clip_tensor_as_image
from llava.eval.find_adv_util import first_token_id_llava

class ScalarTargetedMisclassification(fb.criteria.Criterion):
    """
    Targeted criterion with a *scalar* target class.

    This avoids shape-mismatch issues when the attack internally evaluates
    a dynamic batch size (e.g., HSJ testing multiple candidates per step).
    """
    def __init__(self, target_class: int):
        self.target_class = int(target_class)

    def __call__(self, perturbed: ep.Tensor, outputs: ep.Tensor) -> ep.Tensor:
        # outputs: (N, num_classes)
        pred = outputs.argmax(axis=-1)   # (N,)
        return pred == self.target_class

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target_class={self.target_class})"

def HJS(
    model,
    image_tensor: torch.Tensor,          # shape: (1,3,H,W), float in [0,1]
    input_ids: torch.Tensor,  
    target_token_ids=None,               # token IDs to move toward
    avoid_token_ids=None,                # token IDs to move away from
    mode: str = "toward_and_away",                # "toward", "away", or "toward_and_away"
    norm: str = "linf",                    # perturbation norm: "l2" or "linf"
    steps: int = 5,
    max_queries: int = 20,
    init_search_tries: int = 100,
    init_noise_eps: float = 128/255,
    epsilon: float = 32/255,
    device: str = "cuda",
    layer_list=[2],
    cp=0.2,
    logger=None,
):
    """
    Decision-based HopSkipJump attack for Large Vision-Language Models (LVLMs).

    The attack perturbs the input image such that the *first generated token*
    satisfies a specified discrete decision criterion, without access to logits,
    gradients, or probabilities.

    The decision function is defined over the first generated token only.
    """

    target_token_ids = set(target_token_ids or [])
    avoid_token_ids = set(avoid_token_ids or [])

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(3, 1, 1)

    assert mode in ["toward", "away", "toward_and_away"]
    assert norm in ["l2", "linf"]

    bound_min, bound_max = 0.0, 1.0

    # Clean input
    x0 = image_tensor.detach().clone().float()
    x0 = x0.to(device) * std + mean
    x0 = torch.clamp(x0, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Decision function:
    # Input  : batch of images (numpy array)
    # Output : boolean array indicating whether each input
    #          satisfies the target decision
    # ------------------------------------------------------------------
    def decision_fn(x_batch_np: np.ndarray) -> np.ndarray:
        x_batch = (
            torch.from_numpy(x_batch_np)
            .to(device)
            .float()
        )
        
        x_batch=torch.clamp(x_batch, bound_min, bound_max)
        x_batch=(x_batch - mean)/std

        decisions = []
        for i in range(x_batch.shape[0]):
            token_id = first_token_id_llava(
                model=model,
                input_ids=input_ids,
                image_tensor=x_batch[i:i + 1],
                max_new_tokens=1,
                layer_list=layer_list,
                cp=cp,
            )

            if mode == "toward":
                ok = token_id.item()  in target_token_ids
            elif mode == "away":
                ok = token_id.item()  not in avoid_token_ids
            else:
                ok = (
                    token_id.item()  in target_token_ids
                    and token_id.item()  not in avoid_token_ids
                )

            decisions.append(ok)

        return np.array(decisions, dtype=bool)

    # Minimal Foolbox model wrapper for decision-only access
    class DecisionModel:
        """
        A minimal Foolbox-compatible decision-only model wrapper.

        Foolbox expects:
        - a `.bounds` property returning a `foolbox.types.Bounds` object
        - a callable `__call__` that returns logits (or scores)

        We return 2-class logits whose argmax matches the boolean decision:
        class 1 -> decision True
        class 0 -> decision False
        """
        def __init__(self, bounds=(0, 1)):
            self._bounds = fb.types.Bounds(bounds[0], bounds[1])

        @property
        def bounds(self):
            return self._bounds

        def __call__(self, inputs):
            x_ep, restore_type = ep.astensor_(inputs)   # ensure EagerPy tensor
            x_np = x_ep.raw                            # numpy array (N,3,H,W)

            ok = decision_fn(x_np).astype(np.bool_)     # (N,) boolean
            logits = np.zeros((ok.shape[0], 2), dtype=np.float32)
            logits[:, 0] = (~ok).astype(np.float32)     # class 0: False
            logits[:, 1] = ok.astype(np.float32)        # class 1: True

            return restore_type(ep.astensor(logits))


    # Evaluate decision on clean input
    clean_ok = bool(decision_fn(x0.cpu().numpy())[0])

    # Target decision: flip the current decision
    target_ok = not clean_ok

    logger.print_log(f"[HJS] clean decision={clean_ok}, target decision={target_ok}")

    # ------------------------------------------------------------------
    # Step 1: Find an initial point on the target side of the decision
    # boundary. This is required for decision-based attacks such as HSJ.
    # ------------------------------------------------------------------
    x_init = None
    rng = torch.Generator(device=device)
    rng.manual_seed(0)

    # noise=torch.zeros_like(x0)
    for t in range(init_search_tries):
        noise = torch.empty_like(x0).uniform_(
            -init_noise_eps, init_noise_eps, generator=rng
        )
        # x_try = (x0 + noise).clamp(0, 1)
        x_try=torch.clamp(x0 + noise, bound_min, bound_max)
        ok = bool(decision_fn(x_try.cpu().numpy())[0])

        if ok == target_ok:
            x_init = x_try
            logger.print_log(f"[HJS] found initialization at try={t}")
            break

    if x_init is None:
        logger.print_log(
            "[HJS] Failed to find a valid initialization. "
            "Consider increasing init_noise_eps or init_search_tries."
        )
        x_try=(x_try-mean)/std
        save_clip_tensor_as_image(x_try.cpu(),'3.png')
        return x_try

    # ------------------------------------------------------------------
    # Step 2: Run HopSkipJump using Foolbox
    # ------------------------------------------------------------------
    bounds = (bound_min, bound_max)

    attack = fb.attacks.HopSkipJumpAttack(
        steps=steps,
        initial_gradient_eval_steps=max_queries,
        max_gradient_eval_steps=max_queries,
        constraint=norm,
    )
    # 1) Define the criterion (TARGETED)
    # criterion = fb.criteria.TargetedMisclassification(
    #     ep.astensor(np.array([list(target_token_ids)], dtype=np.int64))
    # )
    criterion = ScalarTargetedMisclassification(1)

    x0_ep = ep.astensor(x0.cpu().numpy())
    xinit_ep = ep.astensor(x_init.cpu().numpy())


    fmodel = DecisionModel(bounds=bounds)

    # 2) Run attack: pass criterion as the 3rd argument
    try:
        adv_ep = attack(
            fmodel,
            inputs=x0_ep,
            criterion=criterion,
            epsilons = epsilon,
            starting_points=xinit_ep,
            # bound_min=bound_min, bound_max=bound_max,
        )

        adv = (
            torch.from_numpy(adv_ep[1].raw)
            .to(device)
            .float()
            .clamp(bound_min, bound_max)
        )

        adv_ok = bool(decision_fn(adv.cpu().numpy())[0])
        logger.print_log(f"[HJS] adversarial decision={adv_ok}")
        adv=(adv-mean)/std
        save_clip_tensor_as_image(adv.cpu(),'3.png')
        return adv
    except ValueError as e:
        logger.print_log(f"[HJS] Attack failed: {e}")
        x_try=(x_try-mean)/std
        save_clip_tensor_as_image(x_try.cpu(),'3.png')
        return x_try
