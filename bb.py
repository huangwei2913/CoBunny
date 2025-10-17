import os
import torch
import matplotlib.pyplot as plt
from modelscope import AutoImageProcessor, AutoModel, AutoConfig
from transformers.image_utils import load_image

# 加载图像
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

# 配置模型和处理器
pretrained_model_name = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(pretrained_model_name, device_map="auto")
cfg = AutoConfig.from_pretrained(pretrained_model_name)

inputs = processor(images=image, return_tensors="pt").to(model.device)

# 自定义函数提取所有stage输出
def forward_with_stages(model, inputs):
    x = inputs.pixel_values
    stage_outputs = []
    for stage in model.stages:
        x = stage(x)
        stage_outputs.append(x)
    return stage_outputs

with torch.inference_mode():
    stage_feats = forward_with_stages(model, inputs)

# 取stage 1特征，去除batch维度 (C, H, W)
stage1_feat = stage_feats[1].cpu().squeeze(0)

# 创建保存目录
save_dir = "./stage1_channels"
os.makedirs(save_dir, exist_ok=True)

# 遍历384个通道，保存为图片
for i in range(stage1_feat.shape[0]):
    channel_img = stage1_feat[i]
    channel_img = (channel_img - channel_img.min()) / (channel_img.max() - channel_img.min() + 1e-5)
    plt.imshow(channel_img, cmap='viridis')
    plt.axis('off')
    file_path = os.path.join(save_dir, f"channel_{i}.png")
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.clf()  # 清理当前figure，避免内存占用过大

print(f"共保存 {stage1_feat.shape[0]} 个通道的特征图到目录: {save_dir}")
