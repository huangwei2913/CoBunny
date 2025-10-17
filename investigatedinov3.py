import torch
from modelscope import AutoImageProcessor, AutoModel,AutoConfig
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

pretrained_model_name = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(
    pretrained_model_name, 
    device_map="auto", 
)
cfg = AutoConfig.from_pretrained(pretrained_model_name)
print(cfg)
inputs = processor(images=image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

pooled_output = outputs.pooler_output
print("Pooled output shape:", pooled_output.shape)
print(model)
_patch_size = model.stages[0].downsample_layers[0].stride[0]
# 读取最后归一化层的输出维度作为hidden size
_hidden_size = model.layer_norm.normalized_shape[0]
# 图像尺寸按照config
_image_size = cfg.image_size

print("_patch_size , _hidden_size ,_image_size ", _patch_size, _hidden_size,_image_size)


######这个脚本是为了测试和知道dinov3编码器结构和模型权重的
print(model.stages[0].downsample_layers[0])
print(model.layer_norm)
