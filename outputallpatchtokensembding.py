import torch
import torch.nn.functional as F
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

# 本地模型权重路径
local_weight_path = '/home/huangwei/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin'

# 设备管理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和预处理器
model, preprocess = create_model_from_pretrained(
    model_name='ViT-H-14',
    pretrained=local_weight_path
)
model = model.to(device)

# 打印模型结构（可选）
print(model)
print("..............")
print(model.visual)
print("..............")
print(model.transformer)
print("Vision feature dim:", model.visual.ln_post.normalized_shape[0])

print("Model parameters:")
for name, _ in model.named_parameters():
    print(name)
print("..............")

# 初始化tokenizer
tokenizer = get_tokenizer('ViT-H-14')

# 加载并预处理图片
image = Image.open('beignets-task-guide.png')
image = preprocess(image).unsqueeze(0).to(device)

# 文本标签
labels_list = ["a dog", "a cat", "a donut", "a beignet"]
text = tokenizer(labels_list, context_length=model.context_length).to(device)

# 开启视觉编码器输出所有patch tokens
model.visual.output_tokens = True

with torch.no_grad(), torch.amp.autocast(device.type):
    # 获取tokens和视觉特征序列
    tokens, image_patches = model.visual(image)

    print("Text tokens shape:", tokens.shape)  # 例如 (1, 1024)
    print("Image patches shape:", image_patches.shape)  # 例如 (1, 256, 1280)

    # 取CLS token作为全局图像特征
    global_image_feature = image_patches[:, 0, :]  # shape: (1, 1280)
    print("Global image feature shape:", global_image_feature.shape)

    text_features = model.encode_text(text)
    print("Text features shape:", text_features.shape)  # 例如 (4, 1024)

    # 打印视觉映射矩阵形状，确认维度
    print("proj shape:", model.visual.proj.shape)

    # 根据proj权重矩阵形状选择是否转置
    if model.visual.proj.shape[0] == global_image_feature.shape[1]:
        # proj形状是 (1280, 1024)
        projected_image_feature = torch.matmul(global_image_feature, model.visual.proj)
    else:
        # proj形状是 (1024, 1280)
        projected_image_feature = torch.matmul(global_image_feature, model.visual.proj.T)

    projected_image_feature = F.normalize(projected_image_feature, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    logit_scale = model.logit_scale.exp()
    logits = projected_image_feature @ text_features.T * logit_scale

    if getattr(model, "logit_bias", None) is not None:
        logits = logits + model.logit_bias

    text_probs = torch.sigmoid(logits)

# 打印结果
zipped_list = list(zip(labels_list, [round(p.item(), 3) for p in text_probs[0]]))
print("Label probabilities: ", zipped_list)

