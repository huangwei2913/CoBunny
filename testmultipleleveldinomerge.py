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

# 修改forward函数获取每个stage输出
def forward_with_intermediate_outputs(model, inputs):
    x = inputs.pixel_values  # 输入tensor
    stage_outputs = []
    
    for idx, stage in enumerate(model.stages):
        x = stage(x)
        stage_outputs.append(x)
    return stage_outputs

with torch.inference_mode():
    stage_feats = forward_with_intermediate_outputs(model, inputs)

for i, feat in enumerate(stage_feats):
    print(f"Stage {i} output shape:", feat.shape)

# 最后可以获取pool输出
with torch.inference_mode():
    outputs = model(**inputs)
print("Pooled output shape:", outputs.pooler_output.shape)




# 在每个stage的输出中，最后两个维度（高度H和宽度W）是随着网络层数增加逐步缩小的，这是由于卷积下采样（stride）导致空间分辨率逐渐减半，体现多尺度抽象。

# 但通道数C（即feature维度）是每个stage固定的，且一般会随着stage的增加逐渐加大（例如192->384->768->1536），以增加网络表达能力。

# 因此，输入图像大小如果不同，stage的H和W会随着输入尺寸变化按比例变化，但C保持不变。

# 举例说明：
# 输入图像越大，stage 0输出的H,W也会越大（比例对应初始空间），而C依然是192。

# 总结：

# H,W维度依赖输入尺寸和stage下采样。

# C维度固定，与stage层数相关。

# 这种设计方便模型同时具备空间细节和语义表达能力，也便于多尺度特征的利用和融合。