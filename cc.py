import torch
import torch.nn as nn
import torch.nn.functional as F


#验证可学习的过滤算法层

class PatchTokenRetention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 简单的映射层，将每个patch token的embedding映射成一个标量概率
        self.token_proj = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (batch_size, num_tokens, dim)
        # 得到每个token的logit
        logits = self.token_proj(x).squeeze(-1)  # (batch_size, num_tokens)
        # sigmoid映射到概率 [0,1]
        probs = torch.sigmoid(logits)
        return probs

# 用法示例
batch_size, num_tokens, dim = 8, 196, 512
patch_tokens = torch.randn(batch_size, num_tokens, dim)

retention_module = PatchTokenRetention(dim)
token_probs = retention_module(patch_tokens)  # (batch_size, num_tokens)

# 根据阈值筛选tokens，例如阈值0.5
threshold = 0.5
mask = token_probs > threshold  # (batch_size, num_tokens), bool值表示保留的tokens

print("Token retention probabilities:", token_probs)
print("Mask of retained tokens:", mask)

