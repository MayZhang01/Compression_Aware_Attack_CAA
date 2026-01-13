import torch
from transformers import GPT2Model, GPT2Tokenizer
import matplotlib.pyplot as plt

# 使用cuda或cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练模型
model_name = "gpt2"
model = GPT2Model.from_pretrained(model_name, output_attentions=True).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 设定测试文本
text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(text, return_tensors="pt").to(device)

# 获取embedding
with torch.no_grad():
    embedding = model.wte(inputs["input_ids"])

print("Original embedding norm per token:")
print(torch.norm(embedding, dim=-1))

# 不同scale
scales = [1.0, 3.0, 10.0]

for scale in scales:
    print("\n"+"="*60)
    print(f"Embedding scale factor: {scale}")

    # 放大embedding
    scaled_embedding = embedding * scale

    # 前向传播
    with torch.no_grad():
        outputs = model(inputs_embeds=scaled_embedding)

    # outputs.attentions是一个list: [num_layers x (batch, num_heads, seq_len, seq_len)]
    attentions = outputs.attentions

    num_layers = len(attentions)
    seq_len = attentions[0].shape[-1]

    # 打印每层第一头注意力分布
    for layer_idx, layer_attn in enumerate(attentions):
        # 取第一头
        attn_head = layer_attn[0,0].detach().cpu()
        entropy = -torch.sum(attn_head * torch.log(attn_head + 1e-8), dim=-1)

        print(f"Layer {layer_idx:2d} attention entropy (mean over tokens): {entropy.mean().item():.4f}")

        # 可视化
        plt.figure(figsize=(6,5))
        plt.imshow(attn_head.numpy(), cmap="viridis")
        plt.colorbar()
        plt.title(f"Layer {layer_idx} Head 0 Attention (scale={scale})")
        plt.xlabel("Key token index")
        plt.ylabel("Query token index")
        plt.tight_layout()
        plt.savefig('figs/'+str(scale)+'_'+str(layer_idx)+'.pdf')
