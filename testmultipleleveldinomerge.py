import torch
from modelscope import AutoImageProcessor, AutoModel, AutoConfig
from transformers.image_utils import load_image
#from imagepreprocessor import  ImageProcessor


from PIL import Image
import math

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return abs(a * b) // gcd(a, b)

def list_lcm(numbers):
    from functools import reduce
    return reduce(lcm, numbers)

class ImageProcessorMultipleEncoders:
    def __init__(self, patch_size_list, max_size=1152, min_no_scale=384):
        self.patch_size_list = patch_size_list
        self.max_size = max_size
        self.min_no_scale = min_no_scale
        self.patch_lcm = list_lcm(patch_size_list)

    def process_image(self, image: Image.Image) -> Image.Image:
        # image is PIL.Image.Image
        W, H = image.size

        # 小于最小阈值，直接返回
        if H <= self.min_no_scale and W <= self.min_no_scale:
            return image

        # 384 ~ 1152 范围内保持不变
        if self.min_no_scale < H <= self.max_size and self.min_no_scale < W <= self.max_size:
            return image

        # 大于1152，重采样到不超过1152且为patch_lcm的最大倍数
        if H > self.max_size or W > self.max_size:
            new_H = (self.max_size // self.patch_lcm) * self.patch_lcm
            new_W = (self.max_size // self.patch_lcm) * self.patch_lcm
            image = image.resize((new_W, new_H), Image.BILINEAR)
            return image

        # 小于384但不满足最小公倍数倍数条件的，调整尺寸
        if H % self.patch_lcm != 0 or W % self.patch_lcm != 0:
            new_H = (H // self.patch_lcm) * self.patch_lcm
            new_W = (W // self.patch_lcm) * self.patch_lcm
            if new_H < 1: new_H = self.patch_lcm
            if new_W < 1: new_W = self.patch_lcm
            image = image.resize((new_W, new_H), Image.BILINEAR)

        return image





url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

patch_size_list = [14, 16]
processor_ = ImageProcessorMultipleEncoders(patch_size_list)
processed_img = processor_.process_image(image)


print("Original image size:", image.size)  # (宽度, 高度)

print("Processed image size:", processed_img.size)




pretrained_model_name = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(pretrained_model_name, device_map="auto")
cfg = AutoConfig.from_pretrained(pretrained_model_name)

inputs = processor(images=processed_img, return_tensors="pt").to(model.device)

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

# 打印last_hidden_state信息
if hasattr(outputs, "last_hidden_state"):
    print("last_hidden_state shape:", outputs.last_hidden_state.shape)
    print("last_hidden_state sample tokens:", outputs.last_hidden_state[0, :5, :5])
else:
    print("No last_hidden_state in outputs.")

print("Pooled output shape:", getattr(outputs, "pooler_output", "None"))



#Stage 0 output shape: torch.Size([1, 192, 56, 56])
#Stage 1 output shape: torch.Size([1, 384, 28, 28])
#Stage 2 output shape: torch.Size([1, 768, 14, 14])
#Stage 3 output shape: torch.Size([1, 1536, 7, 7])
#Pooled output shape: torch.Size([1, 1536])
#last_hidden_state shape: torch.Size([1, 50, 1536])
#last_hidden_state sample tokens: tensor([[-2.2186e+00, -5.9434e-01, -2.3002e+00, -9.5732e-01, -5.2036e-01],
#        [-1.4762e+00, -2.1649e-01, -3.1283e+00,  4.1952e-01,  3.3477e-01],
#        [ 4.8986e-01, -6.2382e-01, -2.1964e+00,  1.0055e+00, -1.4469e-02],
#        [ 8.4573e-01, -9.7807e-01, -1.2511e+00,  7.9045e-04, -2.6689e-01],
#        [-1.7200e+00, -5.4456e-02, -3.2549e+00,  1.7529e+00,  1.0417e+00]],
#       device='cuda:0')
#Pooled output shape: tensor([[-2.2186, -0.5943, -2.3002,  ..., -0.4853, -2.7568,  0.0075]],
#       device='cuda:0')


#返回的张量last_hidden_state是[1, 50, 1536] b ,n, c


# 在每个stage的输出中，最后两个维度（高度H和宽度W）是随着网络层数增加逐步缩小的，这是由于卷积下采样（stride）导致空间分辨率逐渐减半，体现多尺度抽象。

# 但通道数C（即feature维度）是每个stage固定的，且一般会随着stage的增加逐渐加大（例如192->384->768->1536），以增加网络表达能力。

# 因此，输入图像大小如果不同，stage的H和W会随着输入尺寸变化按比例变化，但C保持不变。

# 举例说明：
# 输入图像越大，stage 0输出的H,W也会越大（比例对应初始空间），而C依然是192。

# 总结：

# H,W维度依赖输入尺寸和stage下采样。

# C维度固定，与stage层数相关。

# 这种设计方便模型同时具备空间细节和语义表达能力，也便于多尺度特征的利用和融合。