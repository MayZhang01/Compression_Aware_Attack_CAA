import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Step 1: Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"  # Replace with a smaller model if out of memory
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
model.eval()

# Step 2: Tokenize benign input and get token embeddings
benign_text = "Explain how a nuclear reactor works."
inputs = tokenizer(benign_text, return_tensors="pt").to("cuda")
with torch.no_grad():
    ben_token_embed = model.model.embed_tokens(inputs["input_ids"]).squeeze(0)  # [L, D]

# Step 3: Initialize adv_tokens as one-hot vectors with requires_grad
vocab_embed = model.model.embed_tokens.weight  # [V, D]
vocab_size, hidden_size = vocab_embed.shape
n = 5  # number of adversarial tokens
k = 10  # top-k tokens by gradient
m = 8   # number of candidates per row

adv_token_ids = torch.randint(0, vocab_size, (n,), device="cuda")
adv_tokens = F.one_hot(adv_token_ids, num_classes=vocab_size).float()
adv_tokens.requires_grad_()

# Optimization loop
num_iters = 30
loss_fn = nn.CrossEntropyLoss()

for step in range(num_iters):
    # Step 4: adv_embed = adv_tokens @ vocab_embed
    adv_embed = adv_tokens @ vocab_embed  # [n, D]

    # Step 5: Concatenate with benign embedding
    adv_input_embed = torch.cat([ben_token_embed, adv_embed], dim=0).unsqueeze(0)  # [1, L+n, D]

    # Step 6: Forward and compute loss
    output = model(inputs_embeds=adv_input_embed)
    logits = output.logits[:, -n-1:-1, :]  # predict next tokens
    target_ids = inputs["input_ids"][:, 1:1+n]
    loss = loss_fn(logits.reshape(-1, vocab_size), target_ids.reshape(-1))

    # Backward to get gradients w.r.t. adv_tokens
    model.zero_grad()
    loss.backward()
    adv_grads = adv_tokens.grad.detach()  # [n, V]

    # Step 7: For each row, select top-k tokens by gradient, sample m candidates
    candi_adv_token_ids = []
    for i in range(n):
        topk = torch.topk(adv_grads[i], k).indices  # [k]
        sampled = topk[torch.randint(0, k, (m,))]    # [m]
        candi_adv_token_ids.append(sampled)
    candi_adv_token_ids = torch.stack(candi_adv_token_ids, dim=0).T  # [m, n]

    # Step 8: Convert candi_adv_token_ids to one-hot tensor of shape (m, n, V)
    candi_adv_tokens = F.one_hot(candi_adv_token_ids, num_classes=vocab_size).float()  # [m, n, V]

    # Step 9: Evaluate each candidate and compute loss
    best_loss = float("inf")
    best_tokens = None

    for j in range(m):
        this_adv_embed = candi_adv_tokens[j] @ vocab_embed  # [n, D]
        this_input_embed = torch.cat([ben_token_embed, this_adv_embed], dim=0).unsqueeze(0)
        output = model(inputs_embeds=this_input_embed)
        logits = output.logits[:, -n-1:-1, :]
        loss = loss_fn(logits.reshape(-1, vocab_size), target_ids.reshape(-1))
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_tokens = candi_adv_tokens[j]

    # Step 10: Update adv_tokens with best found
    adv_tokens = best_tokens.detach().clone()
    adv_tokens.requires_grad_()

# Final optimized adversarial token one-hot matrix
adv_tokens_final = adv_tokens
adv_token_ids_final = adv_tokens.argmax(dim=-1)
adv_token_ids_final
