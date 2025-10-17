import torch
from modelscope import AutoImageProcessor, AutoModel, AutoConfig
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

pretrained_model_name = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(pretrained_model_name, device_map="auto")
cfg = AutoConfig.from_pretrained(pretrained_model_name)

inputs = processor(images=image, return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = model(**inputs)

visual_tokens = outputs.last_hidden_state  # 或者你确认的最后一层token输出
print("Original token shape:", visual_tokens.shape)

r = visual_tokens.shape[1] // 2

def bipartite_soft_matching_merge(metric: torch.Tensor, r: int, x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    protected = 0
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)
    if r <= 0:
        return x
    
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)

# 合并50% token
merged_tokens = bipartite_soft_matching_merge(visual_tokens, r, visual_tokens)

print("Merged token shape:", merged_tokens.shape)
