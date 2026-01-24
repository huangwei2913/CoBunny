import torch
from transformers import SiglipVisionModel, SiglipImageProcessor
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)



model_name = "/mnt/siglip-so400m-patch14-384"
processor = SiglipImageProcessor.from_pretrained(model_name)
model = SiglipVisionModel.from_pretrained(model_name)
model.eval()

# 假设image是经过ImageProcessorMultipleEncoders处理的PIL图像
inputs = processor(images=image, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

for idx, feat in enumerate(outputs.hidden_states):
    print(f"Stage {idx} output shape: {feat.shape}")

print("Pooled output shape:", outputs.pooler_output.shape)
print("Last hidden state shape:", outputs.last_hidden_state.shape)
print("Sample tokens:", outputs.last_hidden_state[0, :5, :5])
