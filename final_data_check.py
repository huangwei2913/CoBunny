import os
import torch
import transformers
import json
from PIL import Image

# 1. 导入你的类
from bunny.util.data_utils import LazySupervisedDataset, DataArguments
# 手动模拟一个 ImageProcessor，绕过 OSError
class MockProcessor:
    def __init__(self, size=378):
        self.size = size
    def preprocess(self, images, return_tensors='pt'):
        # 模拟 ClipProcessor 的返回格式
        if not isinstance(images, list):
            images = [images]
        # 返回维度 [N, 3, 378, 378]
        return {'pixel_values': torch.zeros(len(images), 3, self.size, self.size)}
def run_strict_check():
    print("="*60)
    print("🚀 [数据硬核扫描 V2] 目标：绕过模型加载，直击 6967 核心")
    print("="*60)

    # --- 你的真实路径 ---
    MODEL_PATH = "/mnt/conda_data/microsoft/phi-1_5"
    DATA_PATH = "/mnt/conda_data/Bunny-v1.1-data/finetune/bunny_stage3_mixed_2M.json"
    IMAGE_FOLDER = "/mnt/conda_data/Bunny-v1.1-data/finetune/images" 

    # --- Step 1: 模拟配置 ---
    data_args = DataArguments()
    data_args.data_path = DATA_PATH
    data_args.image_folder = IMAGE_FOLDER
    data_args.is_multimodal = True
    data_args.mm_vision_tokens = 365

    # 使用 Mock 处理器，不再去 phi-1_5 目录下找配置文件
    data_args.image_processor = MockProcessor(size=378)

    print(f"📦 正在加载分词器: {MODEL_PATH}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.model_max_length = 2048 

    # --- Step 2: 运行防火墙 ---
    print("\n" + "-"*40)
    print("🔍 执行 LazySupervisedDataset.__init__ (防火墙扫描)")
    print("-"*40)

    try:
        # 这里会触发扫描，如果路径不对或长度超限，这里就会显示保留数
        dataset = LazySupervisedDataset(
            data_path=DATA_PATH,
            tokenizer=tokenizer,
            data_args=data_args
        )
        
        total_in_dataset = len(dataset)
        print(f"\n📊 防火墙结果报告:")
        print(f"   - 最终保留样本数: {total_in_dataset}")
        
        # 尝试读取原始文件总量对比
        try:
            with open(DATA_PATH, 'r') as f:
                raw_json = json.load(f)
                raw_json_count = len(raw_json)
            print(f"   - JSON 原始总量: {raw_json_count}")
            print(f"   - 拦截/丢失率: {(raw_json_count - total_in_dataset)/raw_json_count:.2%}")
        except:
            print("   - 无法读取原始 JSON 总量进行对比")

    except Exception as e:
        print(f"❌ 初始化阶段崩溃: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Step 3: 深度抽查 __getitem__ ---
    print("\n" + "-"*40)
    print("🔍 深度抽查：验证第 0 个和最后一个有效样本")
    print("-"*40)

    for idx in [0, total_in_dataset - 1]:
        if idx < 0: continue
        try:
            print(f"\n[测试索引 {idx}]")
            item = dataset[idx]
            print(f"   - 加载成功！")
            print(f"   - 图像 Tensor 维度: {item['image'].shape}") # 应该是 [6, 2, 3, 378, 378]
            print(f"   - Input_ids 长度: {item['input_ids'].shape}")
            
            # 这里的 valid_tokens 指的是非 -100 的部分，即模型要学习的部分
            valid_tokens = (item['labels'] != -100).sum().item()
            print(f"   - 有效训练 Token 数: {valid_tokens}")
            
        except Exception as e:
            print(f"   - ❌ 加载失败！")
            print(f"   - 错误详情: {e}")
            if hasattr(dataset, 'list_data_dict'):
                bad_entry = dataset.list_data_dict[idx]
                print(f"   - 原始数据预览: {bad_entry.get('id', 'N/A')}")

    print("\n" + "="*60)
    print("🏁 [测试结束] 重点看‘最终保留样本数’是否接近 200w")
    print("="*60)

if __name__ == "__main__":
    run_strict_check()