import torch
import torch.nn.functional as F
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

# 本地模型路径（包含权重和配置文件的目录）
local_weight_path = '/home/huangwei/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin'

# 从本地路径加载模型和预处理器
model, preprocess = create_model_from_pretrained(
    model_name='ViT-H-14',
    pretrained=local_weight_path
)


print(model)  # 打印整体模型结构
print("..............")
print(model.visual)  # 打印视觉编码器部分
print("..............")
print(model.transformer)  # 打印文本编码器部分（如果有）
print("..............")
print("Vision feature dim:", model.visual.ln_post.normalized_shape[0])  # 特征向量大小

# 也可以打印模型中的参数名确认是否有logit_bias
print("Model parameters:")
for name, param in model.named_parameters():
    print(name)

print("..............")
tokenizer = get_tokenizer('ViT-H-14')

# 加载网络图片进行测试（也可以替换成本地图片路径）
image = Image.open(
    'beignets-task-guide.png'
)
image = preprocess(image).unsqueeze(0)

labels_list = ["a dog", "a cat", "a donut", "a beignet"]
text = tokenizer(labels_list, context_length=model.context_length)

with torch.no_grad(), torch.amp.autocast('cuda'):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    print("Image features shape:", image_features.shape)
    print("Text features shape:", text_features.shape)
    print("..............")
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    logit_scale = model.logit_scale.exp()
    logits = image_features @ text_features.T * logit_scale
    if getattr(model, "logit_bias", None) is not None:
        logits = logits + model.logit_bias

    text_probs = torch.sigmoid(logits)

zipped_list = list(zip(labels_list, [round(p.item(), 3) for p in text_probs[0]]))
print("Label probabilities: ", zipped_list)

