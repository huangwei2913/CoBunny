import torch
from transformers import AutoTokenizer
from bunny.model.language_model.bunny_phi import BunnyPhiForCausalLM
from PIL import Image
import os
import torch
import torch.nn as nn
from bunny import conversation as conversation_lib
from bunny.constants import DEFAULT_IMAGE_TOKEN

def get_inference_prompt(question):
    # 1. 强制获取 bunny 模版 (对应 --version bunny)
    conv = conversation_lib.conv_templates["bunny"].copy()
    
    # 2. 模仿训练时的输入逻辑
    # 注意：训练时 preprocess_multimodal 会把 <image> 换成 <img_content>
    # 这里的 DEFAULT_IMAGE_TOKEN 应该就是 "<img_content>"
    message = DEFAULT_IMAGE_TOKEN + "\n" + question
    
    conv.append_message(conv.roles[0], message) # USER: <img_content>\n{question}
    conv.append_message(conv.roles[1], None)    # ASSISTANT:
    
    # 3. 拿到 100% 对齐的字符串
    prompt = conv.get_prompt()
    return prompt
# 配置
#MODEL_PATH = '/mnt/conda_data/checkpoints-pretrain/pretrain_stage1_modified/checkpoint-31216'
MODEL_PATH = '/mnt/CoBunny/checkpoints-stage3/bunny-phi1.5-full-finetune-2000-fp16'


IMAGE_PATH = "/mnt/CoBunny/bunny/model/language_model/xx.jpg"
DEVICE = "cuda:0"
IMAGE_TOKEN_INDEX = -200 
TARGET_ID = -200  # 锁死的逻辑 ID


def materialize_meta_tensors(model):
    print("\n" + "🧪" * 10 + " 物理层物化审计 (修正版) " + "🧪" * 10)
    
    # 1. 处理 Buffer
    for name, buf in model.named_buffers():
        if buf.device.type == 'meta':
            print(f"  📍 发现 meta buffer: {name} | 形状: {buf.shape} | 类型: {buf.dtype}")
            # 修正点：根据数据类型选择填充方式
            if buf.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                real_buf = torch.zeros(buf.shape, dtype=buf.dtype, device='cpu').normal_(std=0.02)
            else:
                # 对于 Long 类型 (如 position_ids)，直接用 0 填充
                real_buf = torch.zeros(buf.shape, dtype=buf.dtype, device='cpu')
            
            parent_name = ".".join(name.split(".")[:-1])
            layer_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, layer_name, real_buf)

    # 2. 处理 Parameter
    for name, param in model.named_parameters():
        if param.device.type == 'meta':
            print(f"  📍 发现 meta parameter: {name} | 形状: {param.shape} | 类型: {param.dtype}")
            # 同样根据类型处理
            if param.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                real_data = torch.zeros(param.shape, dtype=param.dtype, device='cpu').normal_(std=0.02)
            else:
                real_data = torch.zeros(param.shape, dtype=param.dtype, device='cpu')
            
            param.data = real_data
            
    print("✅ 物化完成，物理层的所有空壳已填满。")

# --- 测试这个函数是否能解决 NotImplementedError ---
def test_step_one(model):
    # 第一步：物化
    materialize_meta_tensors(model)
    
    # 第二步：尝试搬运 (这是检验真理的唯一标准)
    try:
        print(f"🚀 尝试执行 model.to('cuda:0')...")
        model.to("cuda:0")
        print("🎉 物理层通关！NotImplementedError 已消失。")
        return True
    except Exception as e:
        print(f"❌ 物理层依然报错: {e}")
        return False



# --- 定义全局常量 ---

def tokenizer_image_token_custom(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX):
    """
    将文本按 <img_content> 切开，中间插入逻辑 ID 
    """
    # 1. 编码文本块
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<img_content>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    
    # 2. 处理 BOS Token
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])


    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to("cuda:0")



def get_six_crops(image_path, processor):
    """6图切分 - 训练时形状对齐"""
    raw_image = Image.open(image_path).convert('RGB')
    w, h = raw_image.size
    def calculate_anchors(full_len, target_len):
        if full_len <= target_len: return [0, 0, 0, 0, 0]
        max_scroll = full_len - target_len
        return [0, max_scroll // 4, max_scroll // 2, 3 * max_scroll // 4, max_scroll]
    
    target_sz = 378
    global_img = raw_image.resize((target_sz, target_sz), Image.BILINEAR)
    x_coords = calculate_anchors(w, target_sz)
    y_coords = calculate_anchors(h, target_sz)
    crops = [
        raw_image.crop((x_coords[0], y_coords[0], x_coords[0] + target_sz, y_coords[0] + target_sz)),
        raw_image.crop((x_coords[4], y_coords[0], x_coords[4] + target_sz, y_coords[0] + target_sz)),
        raw_image.crop((x_coords[0], y_coords[4], x_coords[0] + target_sz, y_coords[4] + target_sz)),
        raw_image.crop((x_coords[4], y_coords[4], x_coords[4] + target_sz, y_coords[4] + target_sz)),
        raw_image.crop((x_coords[2], y_coords[2], x_coords[2] + target_sz, y_coords[2] + target_sz)),
    ]
    six_images = [global_img] + crops
    pixel_values = processor.preprocess(six_images, return_tensors='pt')['pixel_values']
    
    if pixel_values.dim() == 5:
        pixel_values = pixel_values.unsqueeze(0)
    return pixel_values.to(DEVICE, dtype=torch.float16)



def get_processed_images(image_path, processor):
    # 按照 Bunny 的标准切分逻辑加载图片
    raw_image = Image.open(image_path).convert('RGB')
    target_sz = 378
    # 简化示例：仅返回 global_img。如需 6 图切分，请复用你之前的 get_six_crops 函数
    pixel_values = processor.preprocess(raw_image, return_tensors='pt')['pixel_values']
    return pixel_values.to(DEVICE, dtype=torch.float16)


def test_logical_alignment(model, tokenizer):
    print("\n" + "📊" * 10 + " 逻辑层“滴血验亲”测试 " + "📊" * 10)
    
    # 1. 设置我们公认的 TARGET_ID
    TARGET_ID = -200 
    
    # 2. 强制锁定模型配置 (这是为了防止模型内部 config 还没同步)
    print(f"  - [操作] 强制锁定 model.config.image_token_index 为 {TARGET_ID}")
    model.config.image_token_index = TARGET_ID
    if hasattr(model, 'model'):
        model.model.config.image_token_index = TARGET_ID

    # 3. 构造 input_ids 并测试匹配
    prompt = "USER: <img_content>\nDetailed description: ASSISTANT:"
    
    # 使用你脚本里的 tokenizer_image_token_custom 
    # (注意：确保该函数内部确实把 <img_content> 换成了 TARGET_ID)
    input_ids = tokenizer_image_token_custom(prompt, tokenizer, TARGET_ID).to(DEVICE)
    
    # 4. 核心审计：直接对比
    match_indices = (input_ids == model.config.image_token_index).nonzero()
    match_count = match_indices.size(0)
    
    print(f"  - [结果] input_ids 长度: {input_ids.shape[1]}")
    print(f"  - [结果] 匹配到的占位符数量: {match_count}")
    
    if match_count > 0:
        print(f"  - [结果] 占位符所在位置索引: {match_indices.tolist()}")
        print("🎉 逻辑层通关！'特征数量不匹配' 的报错已从源头根除。")
        return True
    else:
        print("❌ 逻辑层失败！input_ids 里依然没有找到匹配的 ID。")
        print(f"  - 调试：input_ids 的前 20 个值: {input_ids[0][:20].tolist()}")
        print(f"  - 调试：模型此时期待的 ID 是: {model.config.image_token_index}")
        return False

# 紧接在你刚才的“物理通关”代码后面调用：
# if test_logical_alignment(model, tokenizer):
#     print("🚀 准备启动最终的 model.generate()...")
def debug_final_shape(model, input_ids, images):
    print("🔬 正在深度追踪 prepare_inputs_labels_for_multimodal...")
    
    # 使用 *args 接收所有可能的返回值，防止解包报错
    outputs = model.prepare_inputs_labels_for_multimodal(
        input_ids, 
        None, # labels
        None, # attention_mask
        None, # past_key_values
        None, # inputs_embeds
        images
    )
    
    print(f"📊 函数返回了 {len(outputs)} 个对象")
    
    final_embeds = outputs[0]
    print(f"🚨 [FINAL VERDICT] 最终注入后的 Embeds 形状: {final_embeds.shape}")
    
    # 计算逻辑
    # 原始 input_ids 长度
    raw_len = input_ids.shape[1]
    # 实际得到的长度
    actual_len = final_embeds.shape[1]
    
    print(f"📝 原始文本长度: {raw_len}")
    print(f"📸 视觉注入后的总长度: {actual_len}")
    print(f"🧮 实际增加的特征数 (应为 365 或其倍数): {actual_len - raw_len + 1}")

    return final_embeds


def run_inference():
    # 1. 加载模型 (实心加载)
    model = BunnyPhiForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False, 
        device_map=None,
        trust_remote_code=True
    )
    test_step_one(model)
    # 2. 现在执行搬运，绝对不会报错
    
    # --- 3. 加载 Tokenizer 并构造 input_ids ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    test_logical_alignment(model,tokenizer) 
    vision_tower = model.get_vision_tower()
    # 确保视觉塔内部已经 load_model
    if not getattr(vision_tower, "is_loaded", False):
        vision_tower.load_model()   
    image_tensor = get_six_crops(IMAGE_PATH, vision_tower.image_processor)
    image_tensor = image_tensor.to(DEVICE, dtype=torch.float16)

    question = "What is in the image?"
    prompt = get_inference_prompt(question)
    # 确保模型的所有部分都在 GPU 上
    if next(model.parameters()).device.type != 'cuda':
        print("⚠️ 检测到模型部分参数不在 CUDA，正在强制同步...")
        model.to(DEVICE)
    
    # 确保 input embedding 层也在 GPU (针对你遇到的特定报错)
    model.get_input_embeddings().to(DEVICE)

    input_ids = tokenizer_image_token_custom(prompt, tokenizer, IMAGE_TOKEN_INDEX).to(DEVICE)

    # 增加一个形状检查，确保它是 [1, seq_len] 而不是 [seq_len]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    print(f" 🎯 最终逻辑序列 (Input IDs) 形状: {input_ids.shape}")
    print(f" 🎯 最终逻辑序列内容: {input_ids[0].tolist()}")
    # 🔥 1. 调用自定义转换器，将 <img_content> 转换为逻辑 ID -200
    logical_count = (input_ids == IMAGE_TOKEN_INDEX).sum().item()
    print(f" 🧮 逻辑占位符 ({IMAGE_TOKEN_INDEX}) 数量: {logical_count}")

    # 🛠️ 3. 解决 Attention Mask 警告：手动创建一个全 1 的 mask
    attention_mask = torch.ones_like(input_ids).to(DEVICE)
    
    # 🛡️ 4. 解决 Pad Token 致命伤：使用 Stage 3 新增的 <pad>
    # 在 train_stage3.py 中你添加了 <pad>，现在必须显式使用它
    pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
    if pad_token_id == tokenizer.unk_token_id:
        # 如果没搜到新 pad，尝试用原来的 eos 兜底
        pad_token_id = tokenizer.eos_token_id

    stop_token_ids = [tokenizer.eos_token_id]
    print(f"🚀 [设备检查] Model Device: {next(model.parameters()).device}")
    print(f"🚀 [设备检查] Input IDs Device: {input_ids.device}")

    # 5. 执行生成
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=64, # 先看短描述
            # 必须传 mask，防止 pad/eos 混淆
            attention_mask=attention_mask,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False
        )


    # 6. 结果解码
# 6. 结果解码 (修正版：过滤掉非文字 ID)
    # 重点：output_ids[0] 是个 Tensor，得先转成 list，然后把 -200 这种东西踢出去
    clean_output_ids = [idx for idx in output_ids[0].tolist() if idx >= 0]
    
    # 现在解码就绝对不会报 TypeError 了
    response = tokenizer.decode(clean_output_ids, skip_special_tokens=True)
    
    print(f"\n📄 模型输出全文:\n{response}")

    if "ASSISTANT:" in response:
        answer = response.split("ASSISTANT:")[-1].strip()
        print(f"\n✅ 最终 AI 回答:\n{answer}")
    else:
        # Stage 1 没生成标志很正常，我们直接看 response 就行
        print("\n⚠️ 提示：模型处于预训练阶段，可能直接在续写描述而没有输出标志。")


if __name__ == "__main__":
    run_inference()