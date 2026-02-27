我们完成了第二个阶段recipe2时候，要先合并权重
合并 Recipe-2 零件： 运行 merge_lora_weights.py
指定第二个阶段的权重存储目录，指定phi-1.5目录，指定合合并后的目录
另外一个要注意的事情是，在merge之前，先要把/mnt/CoBunny/checkpoints-finetune/phi-1.5-lora-finetune-multinode-recipe2第二阶段
权重中的配置文件config.json中加入"continuous_training": true 这一步至关重要，它告诉代码不要去线上下载 SigLIP，而是加载你本地合并好的混合编码器。
python script/merge_lora_weights.py   --model-path /mnt/CoBunny/checkpoints-finetune/phi-1.5-lora-finetune-multinode-recipe2   --model-base /mnt/conda_data/microsoft/phi-1_5   --model-type phi-1.5   --save-model-path /mnt/CoBunny/checkpoints-finetune/phi-1.5-bunny-mixed-final
检查第二个阶段recipe2时的目录，确认有如下文件
(base) huangwei@ecs-53704537-002:/mnt/CoBunny/checkpoints-finetune/phi-1.5-lora-finetune-multinode-recipe2$ ls -alh
total 1.4G
drwxrwxr-x 3 huangwei huangwei 4.0K Dec 20 08:53 .
drwxrwxr-x 6 huangwei huangwei 4.0K Dec 20 09:27 ..
-rw-rw-r-- 1 huangwei huangwei  918 Dec 16 21:41 adapter_config.json
-rw-rw-r-- 1 huangwei huangwei 109M Dec 16 21:41 adapter_model.safetensors
drwxrwxr-x 3 huangwei huangwei 4.0K Dec 16 21:41 checkpoint-30872
-rw-rw-r-- 1 huangwei huangwei 1.2K Dec 20 08:53 config.json
-rw-r--r-- 1 huangwei huangwei  12K Dec 20 08:48 .config.json.swp
-rw-rw-r-- 1 huangwei huangwei 165K Dec 16 21:41 log.txt
-rw-rw-r-- 1 huangwei huangwei 1.3G Dec 16 21:41 non_lora_trainables.bin
-rw-rw-r-- 1 huangwei huangwei 5.1K Dec 16 21:41 README.md
-rw-rw-r-- 1 huangwei huangwei 529K Dec 16 21:41 trainer_state.json
检查合并后的权重文件
(base) huangwei@ecs-53704537-002:/mnt/CoBunny/checkpoints-finetune/phi-1.5-bunny-mixed-final$ ls -alh
total 3.9G
drwxrwxr-x 2 huangwei huangwei 4.0K Dec 20 13:55 .
drwxrwxr-x 6 huangwei huangwei 4.0K Dec 20 09:27 ..
-rw-rw-r-- 1 huangwei huangwei 1.1K Dec 20 09:34 added_tokens.json
-rw-rw-r-- 1 huangwei huangwei 2.2K Dec 20 13:07 cleaner.py
-rw-rw-r-- 1 huangwei huangwei 1.4K Dec 20 13:55 config.json
-rw-rw-r-- 1 huangwei huangwei 1.3K Dec 20 13:19 config.json_bak
-rw-rw-r-- 1 huangwei huangwei 1.3K Dec 20 13:59 configuration_bunny_phi.py
-rw-rw-r-- 1 huangwei huangwei 446K Dec 20 09:34 merges.txt
-rw-rw-r-- 1 huangwei huangwei 251K Dec 20 15:28 modeling_bunny_phi.py
-rw-rw-r-- 1 huangwei huangwei 3.9G Dec 20 09:34 pytorch_model.bin
-rw-rw-r-- 1 huangwei huangwei  441 Dec 20 09:34 special_tokens_map.json
-rw-rw-r-- 1 huangwei huangwei 7.3K Dec 20 09:34 tokenizer_config.json
-rw-rw-r-- 1 huangwei huangwei 3.4M Dec 20 09:34 tokenizer.json
-rw-rw-r-- 1 huangwei huangwei 780K Dec 20 09:34 vocab.json
将config.josn中的"continuous_training": false 

如果在这个郭晨中发生了
报错的核心在这一行： TypeError: AdaptiveConcatenationVisionTower.__init__() got an unexpected keyword argument 'delay_load'
你需要显式地增加 delay_load=False 参数，或者通过 **kwargs 吸收掉它（建议显式增加，这样更清晰）：
Python

class AdaptiveConcatenationVisionTower(nn.Module):
    # 增加 delay_load 参数，默认值为 False
    def __init__(self, vision_tower, args, delay_load=False, **kwargs):
        super().__init__()
        self.is_loaded = False 
        
        # 保存这个变量，虽然合并脚本可能不需要它
        self.delay_load = delay_load
        
        # ... 你原本的初始化逻辑 ...
        
        # 如果 delay_load 为 False，通常需要立即加载模型
        if not self.delay_load:
            self.load_model()
在 Bunny/LLaVA 的逻辑中：
训练时：delay_load 通常为 False，因为需要立刻加载权重进行微调。
推理或合并时：代码有时会先初始化一个空的结构，然后再手动填充权重，这时它会尝试传入 delay_load=True

class AdaptiveConcatenationVisionTower(nn.Module):
    def __init__(self, 
                 vision_tower, 
                 args, 
                 delay_load=False,  # <--- 必须加上这个参数名，并给个默认值 False
                 grid_size=32):
        super().__init__()
        self.is_loaded = False
        
        # ... 你之前的代码保持不变 ...
        
        # 将原本直接运行的 load_vision_towers 逻辑，改为受 delay_load 控制
        if not delay_load:
            self.load_vision_towers(vision_tower_name_list, args)
        else:
            # 如果是延迟加载，我们只需要保存变量，等之后手动调 load_model()
            self.vision_tower_name_list = vision_tower_name_list
            self.args = args

    # ... 你的 load_vision_towers 定义 ...

    def load_model(self):
        # 这个函数是给外部调用的（比如在合并权重时）
        if not self.is_loaded:
            # 这里的 vision_tower_name_list 和 args 需要确保能访问到
            # 建议在 __init__ 里用 self. 保存一下这两个变量
            self.load_vision_towers(self.vision_tower_name_list, self.args)
        
        # 确保你的断言依然有效
        assert self.is_loaded, "All the vision encoders should be loaded during initialization!"

如果出现在merge的过程中token不匹配问题，则需要修改 merge_lora_weights.py 绕过 pad_token_id 报错。
要彻底解决这连环三个问题（TypeError 参数冲突、IndexError 分片错误、AttributeError 配置缺失），我们不能只改脚本了，必须对 builder.py 底层代码进行一次“手术”。这是最稳妥、也是唯一的终极解决方案。
找到bunny/model/builder.py代码中的 load_pretrained_model 函数中约 第 46 行
model = BunnyPhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
解决 AttributeError (pad_token 报错)修改这个builder.py中的
if model.generation_config is not None:
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
else:
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config.pad_token_id = tokenizer.pad_token_id



这里的核心矛盾在于 Safetensors (新格式) vs PyTorch Bin (旧格式) 对“共享张量”的处理方式：

Safetensors 的“洁癖”： 为了极致的加载速度和安全性，Safetensors 规定：模型文件中每一个张量的内存地址必须是唯一的。 在你的代码中，你为了方便管理，把同一个 dino_vision_tower 既放到了 self.dino_vision_tower，又放到了 self.vision_towers[0]。

结果：两个不同的“名字”指向了内存里同一个“对象”。

报错：Safetensors 检查到这种“多重映射”时会报错，因为它怕在加载时重复分配内存导致混乱。

PyTorch Bin 的“包容”： 当你设置 safe_serialization=False 时，保存的是传统的 pytorch_model.bin（本质是 Python 的 Pickle 格式）。

逻辑：它不管内存地址是否重复，它只负责按照模型的 state_dict 顺序把权重吐出来。

结果：虽然在保存时可能因为引用关系多写了一点冗余数据，或者仅仅是记录了映射关系，但它不会阻拦你保存。

请执行命令：cat /mnt/CoBunny/checkpoints-finetune/phi-1.5-bunny-mixed-final/config.json 你应该能看到类似 model_type: "bunny-phi"，并且在 architectures 里有你自定义的模型类名。这意味着当你加载这个 pytorch_model.bin 时，程序会自动去找你的 AdaptiveConcatenationVisionTower 结构。


还有一个关键的点需要注意的是，我们的混合编码器使用的dino3的权重必须保保存在和训练时候指定的目录一样下面
self.pretrained_path = "/mnt/facebook/dinov3-convnext-large-pretrain-lvd1689m"
self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)

Processor 本地化：local_processor_path = "/mnt/conda_data/openai/clip-vit-large-patch14"。你直接指定了绝对路径，这意味着模型在预处理图像时，会直接读取你硬盘上的 preprocessor_config.json，不再联网。

（后续，我们会修改这个问题）

我们还可以运行校验权重代码，看合并后的权重是否是正确的
python checkfinalmodelweights.py 
(/mnt/conda-envs/bunny) huangwei@ecs-53704537-002:/mnt/CoBunny$ python checkfinalmodelweights.py 
/mnt/conda-envs/bunny/lib/python3.10/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
🔍 开始扫描模型指纹: /mnt/CoBunny/checkpoints-finetune/phi-1.5-bunny-mixed-final/pytorch_model.bin

==================================================
组件名称                      | 检测结果      
--------------------------------------------------
语言模型 (LLM)                | ✅ 存在
投影层 (Projector)           | ✅ 存在
DINOv3 视觉塔                | ✅ 存在
Oryx-ViT 视觉塔              | ✅ 存在
自定义 Cross-Attn 融合层        | ✅ 存在
可学习的 Pseudo-CLS 头         | ✅ 存在
==================================================

🎊 校验通过！你的 3.9G 模型是一个完整的“混合动力”多模态模型。
📊 投影层维度采样: torch.Size([2048, 1024]) (符合预期)
(/mnt/conda-envs/bunny) huangwei@ecs-53704537-002:/mnt/CoBunny$ 

你的 3.9G 模型文件夹（例如：phi-1.5-bunny-mixed-final/）里现在必须包含以下这些“家庭成员”：

pytorch_model.bin：你合并出的 3.9G 权重。

modeling_bunny_phi.py：你刚拼好的单体代码。

configuration_bunny_phi.py：里面定义了 BunnyPhiConfig 类（这个很简单，基本就是继承 PhiConfig 并改个 model_type）。

config.json：这是最关键的一步，你需要手动修改它，加入 auto_map 链接。

请确保 config.json 包含以下内容：

JSON

{
  "model_type": "bunny-phi",
  "auto_map": {
    "AutoConfig": "configuration_bunny_phi.BunnyPhiConfig",
    "AutoModelForCausalLM": "modeling_bunny_phi.BunnyPhiForCausalLM"
  },
  "mm_vision_tower": "mixedencoder", 
  "mm_projector_type": "mlp2x_gelu",
  ... 
}
注：mm_vision_tower 的值一定要对应你代码里 build_vision_tower 判断的那个字符串。

第二步：验证模型能否正常“睁眼”
在跑大规模评测前，先用几行 Python 代码测试一下你的 Flatten 工作是否完美。

创建一个 test_load.py：

Python

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./你的模型文件夹路径"

# 1. 测试加载
print("⏳ 正在加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    torch_dtype=torch.float16, 
    device_map="cpu" # 先用 CPU 测，省显存
)

# 2. 测试视觉塔初始化
print("👁️ 正在初始化视觉塔...")
vision_tower = model.get_model().get_vision_tower()
vision_tower.load_model() # 看看会不会报路径错误

print("✅ 恭喜！模型代码完全自洽，可以独立运行。")
第三步：配置 MME 推理脚本
现在回到你之前的 model_vqa_loader_mme.py。因为你已经做了 Flatten，加载代码变得极其简单：

修改 eval_model 函数中的加载部分：

Python

def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    
    # 只要有了 auto_map，这就成了万能加载语句
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    
    # 必须手动调用一次视觉塔加载，以载入 DINO/Oryx 的权重
    model.get_model().get_vision_tower().load_model()
    
    # ... 后面的 DataLoader 和 Inference 逻辑保持不变 ...
第四步：执行 MME 评测脚本
在终端运行你的评测命令。注意：因为你的模型现在是 phi-1.5 架构，且逻辑都在模型文件夹里，所以命令可以很清爽：

Bash


python -m bunny.eval.model_vqa_loader_mme_mixencoders     --model-path /mnt/CoBunny/checkpoints-finetune/phi-1.5-bunny-mixed-final     --image-folder ./eval/mme/MME_Benchmark_release_version/MME_Benchmark     --question-file ./eval/mme/bunny_mme.jsonl     --answers-file ./eval/mme/answers/mixed_phi1.5_mme_results.jsonl     --temperature 0     --conv-mode bunny


第五步：结果后处理（分数的诞生）
MME 跑完后会生成一个大 JSONL。你还需要运行 MME 官方的脚本来算出最后的得分（感知分 + 推理分）：

转换格式：运行 convert_answer_to_mme.py。

计算分数：运行 calculation_mme.py。



我们修改了预训练时候，要将混合编码器中除了子编码器之外的跨塔注意力模块以及伪cls模块全部导出和合并在投影层中的代码

例如在config.json加入对vision_tower_dino和vision_tower_oryx模型的引用
(base) huangwei@ecs-53704537-002:/mnt/CoBunny/checkpoints-pretrain/bunny-phi1.5-mixed-pretrain-v2/checkpoint-100$ cat config.json 
{
  "architectures": [
    "PhiForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 50256,
  "dtype": "float32",
  "embd_pdrop": 0.0,
  "eos_token_id": 50256,
  "freeze_mm_mlp_adapter": false,
  "hidden_act": "gelu_new",
  "hidden_size": 2048,
  "image_aspect_ratio": null,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 2048,
  "mm_hidden_size": 1024,
  "mm_projector_lr": null,
  "mm_projector_type": "mlp2x_gelu",
  "mm_resampler_type": null,
  "mm_vision_select_feature": "patch",
  "mm_vision_select_layer": -1,
  "mm_vision_tower": "mixedencoder",
  "model_type": "bunny-phi",
  "vision_tower_dino": "/mnt/facebook/dinov3-convnext-large-pretrain-lvd1689m", 
  "vision_tower_oryx": "oryx_vit:/mnt/THUdyhOryx-ViT/oryx_vit.pth",    
  "num_attention_heads": 32,
  "num_hidden_layers": 24,
  "num_key_value_heads": 32,
  "pad_token_id": 50256,
  "partial_rotary_factor": 0.5,
  "qk_layernorm": false,
  "resid_pdrop": 0.0,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "tokenizer_model_max_length": 2048,
  "tokenizer_padding_side": "right",
  "transformers_version": "4.57.1",
  "tune_mm_mlp_adapter": true,
  "unfreeze_vision_tower": false,
  "use_cache": false,
  "use_mm_proj": true,
  "use_s2": false,
  "vocab_size": 51200
}


//下面这个验证代码是非常重要的
import os
import sys

# 关键：强制指定单卡环境，彻底解决 Runtime Error: Expected all tensors to be on the same device
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import torch
from PIL import Image
from transformers import AutoConfig, logging
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin

# 确保能找到 bunny 模块
sys.path.append(os.getcwd())

from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from bunny.model.language_model.phi import PhiForCausalLM

def test_inference():
    disable_torch_init()

    # --- 1. 路径设置 ---
    checkpoint_path = '/mnt/CoBunny/checkpoints-pretrain/bunny-phi1.5-mixed-pretrain-v2/checkpoint-100'
    base_llm_path = '/mnt/conda_data/microsoft/phi-1_5' 
    dino_path = "/mnt/facebook/dinov3-convnext-large-pretrain-lvd1689m"
    oryx_path = "oryx_vit:/mnt/THUdyhOryx-ViT/oryx_vit.pth"    
    model_name = 'bunny-phi-1.5'
    model_type = 'phi-1.5'

    print(f"🔄 正在读取配置并注入混合编码器参数...")
    from transformers.cache_utils import DynamicCache
    
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None

    if not hasattr(DynamicCache, "get_usable_length"):
        print("🔧 正在修复 DynamicCache 兼容性 (get_usable_length 严谨版)...")
        def get_usable_length(self, seq_length=None, layer_idx=None):
            # 关键修复：如果 layer_idx 是 None，直接调用不带参数的 get_seq_length
            if layer_idx is None:
                return self.get_seq_length()
            return self.get_seq_length(layer_idx)
        
        DynamicCache.get_usable_length = get_usable_length

    # --- 2. 加载模型 ---
    print("🔄 正在通过混合逻辑加载模型 (强制单卡模式)...")
    # 注意：这里我们传入 config=cfg_pretrained 确保路径生效
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=checkpoint_path,   
        model_base=base_llm_path,    
        model_name=model_name,
        model_type=model_type
    )

    # --- 3. 核心补丁：类结构重塑与 Cache 兼容性 ---
    print("🔧 执行类结构重塑与 Cache 兼容性补丁...")
    
    # 修复 DynamicCache 属性名缺失
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None

    # 动态重塑类继承关系，找回 generate 等缺失属性
    class FullyFixedBunnyModel(model.__class__, PhiForCausalLM, GenerationMixin):
        pass
    model.__class__ = FullyFixedBunnyModel

    # 修复视觉塔接口
    if not hasattr(model, 'get_vision_tower'):
        model.get_vision_tower = lambda: model.model.get_vision_tower()

    # 强制将整个模型移动到同一设备并设为 eval 模式
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    # --- 4. 准备图片 ---
    image_path = "Test.jpg"
    if not os.path.exists(image_path):
        print(f"❌ 找不到测试图片 {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    processed_output = image_processor.preprocess(image, return_tensors="pt")
    
    # 这里的 Key 必须与你定义的 SingleImageProcessor 对应
    image_tensor = processed_output["pixel_values"].to(device, dtype=torch.float16)
    print(f"✅ 图像 Tensor 准备就绪，形状: {image_tensor.shape}")

    # --- 5. 构建推理 ---
    prompt = "A picture of"
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, -200, return_tensors="pt")
        .unsqueeze(0)
        .to(device)
    )

    print("🚀 启动混合推理引擎...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=20,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # --- 6. 结果展示 ---
    output_text = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
    
    print("\n" + "=" * 40)
    print(f"🖼️ 模型推理结果: {output_text}")
    print("=" * 40)

    # 逻辑验证
    if len(output_text) < 3 or (output_text.count('!') > 5):
        print("🚩 警告：输出疑似异常（感叹号过多或过短）。可能需要检查 Projector 训练状态。")
    else:
        print("✅ 成功：模型输出了有效文本，混合编码器逻辑已跑通。")

if __name__ == "__main__":
    test_inference()


-------------------
第二个阶段的配置文件
{
  "architectures": [
    "BunnyPhiForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 50256,
  "dtype": "float16",
  "embd_pdrop": 0.0,
  "eos_token_id": 50256,
  "freeze_mm_mlp_adapter": false,
  "hidden_act": "gelu_new",
  "hidden_size": 2048,
  "image_aspect_ratio": null,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "layer_norm_eps": 1e-05,
  "lora_enable": false,
  "max_position_embeddings": 2048,
  "mm_hidden_size": 1024,
  "mm_projector_lr": null,
  "mm_projector_type": "mlp2x_gelu",
  "mm_resampler_type": null,
  "mm_vision_select_feature": "patch",
  "mm_vision_select_layer": -1,
  "mm_vision_tower": "mixedencoder",
  "mm_use_im_start_end": false,
  "mm_use_im_patch_token": false,
  "image_token_index": -200,
  "model_type": "bunny-phi",
  "num_attention_heads": 32,
  "num_hidden_layers": 24,
  "num_key_value_heads": 32,
  "pad_token_id": 50256,
  "partial_rotary_factor": 0.5,
  "qk_layernorm": false,
  "resid_pdrop": 0.0,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "tokenizer_model_max_length": 2048,
  "tokenizer_padding_side": "right",
  "transformers_version": "4.57.1",
  "tune_mm_mlp_adapter": true,
  "unfreeze_vision_tower": true,
  "use_cache": false,
  "use_mm_proj": true,
  "use_s2": false,
  "vision_tower_dino": "/mnt/facebook/dinov3-convnext-large-pretrain-lvd1689m",
  "vision_tower_siglip": "/mnt/siglip-so400m-patch14-384",
  "vocab_size": 50295
}
----------------------------
第二个阶段的推理代码

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import copy
from PIL import Image
from transformers import AutoTokenizer
from bunny.model.language_model.bunny_phi import BunnyPhiForCausalLM
from bunny.util.mm_utils import tokenizer_image_token
from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from bunny.model.multimodal_encoder.AdaptiveConcatenationVisionTower import ImageProcessorMultipleEncoders

def run_debug_inference():
    model_path = "/mnt/CoBunny/checkpoints-finetune/bunny-phi1.5-mixed-lora-695k/checkpoint-4000"
    image_path = "testt.jpg"
    device = "cuda"

    print(f"--- 🛠️ 开始深度诊断 ---")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = BunnyPhiForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)

    # --- 修复后的权重检查 ---
    print("\n🔍 [诊断 1: 融合层权重]")
    if hasattr(vision_tower, 'final_cls_weights'):
        weights = vision_tower.final_cls_weights.data
        print(f"融合层权重: {weights}")
        # 修复 dtype 不匹配报错
        is_initial = torch.allclose(weights, torch.tensor([0.5, 0.5], dtype=torch.float16, device=device), atol=1e-2)
        if is_initial:
            print("⚠️ 警告：权重接近初始值。")
        else:
            print("✅ 权重已偏离初始值，训练生效。")

    # --- 极简提示词 (针对小模型优化) ---
    # 格式：<image>\nUSER: What is in the image? ASSISTANT:
    question = "What is in the image?"
    prompt = f"{DEFAULT_IMAGE_TOKEN}\nUSER: {question} ASSISTANT:"
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    print("\n🔍 [诊断 2: Token 识别]")
    if IMAGE_TOKEN_INDEX in input_ids:
        pos = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1].item()
        print(f"✅ 成功识别图像占位符 (-200) 在位置: {pos}")
    else:
        print("❌ 错误：未识别到 -200")

    image = Image.open(image_path).convert("RGB")
    image_processor = ImageProcessorMultipleEncoders(patch_size_list=[14], target_size=384)
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(device, dtype=torch.float16)

    print("\n🚀 [诊断 3: 推理测试]")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=64, # 先看短描述
            repetition_penalty=1.5,
            # 必须传 mask，防止 pad/eos 混淆
            attention_mask=torch.ones_like(input_ids).to(device),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    response = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    print(f"\n✨ 推理结果:\n{response}")

if __name__ == "__main__":
    run_debug_inference()

--------------
# 阶段2：微调（包含 Recipe-1 和 Recipe-2）

#可以把 Recipe-1 和 Recipe-2 写在一个脚本里，用 && 连接，确保第一步成功后自动跑第二步：

#Recipe-1：--unfreeze_vision_tower False。先让语言模型学会多模态指令。

#Recipe-2：--unfreeze_vision_tower True。打开视觉塔，微调全链路。

#关键修正：全部统一使用 --version bunny，彻底告别 phi3。

#🎯 第二阶段（Stage 2）的核心定义：我们在练什么？
#你之前的理解部分正确，但不完全完整。在第二阶段，我们不再是简单的“训练映射层”，而是在进行一次**“三位一体”的协同进化**。

#具体来说，显存里发生的事情是这样的：

#🧠 大脑 (LLM - Phi-1.5)：

#本体：冻结 (Frozen)。

#挂件 (LoRA)：🔥 训练 (Trainable)。这是本阶段的重点。LoRA 模块插入在 LLM 的每一层中，学习如何处理复杂的指令逻辑（如“解释为什么”、“提取文字”）。

#👀 眼睛 (Vision Tower)：

#视网膜 (DINO/Oryx Backbone)：冻结 (Frozen)。保护基础视觉能力。

#神经束 (Fusion Layers 113 参数)：🔥 训练 (Trainable)。这是你独有的优势。它们必须继续进化，学会根据 LoRA 的指令需求，动态调整 DINO 和 Oryx 的融合权重（比如问颜色时多听 Oryx 的，看结构时多听 DINO 的）。

#🌉 桥梁 (Projector 4 参数)：

#本体：🔥 训练 (Trainable)。继续精调，修正 Stage 1 的“指鹿为马”现象。


#!/bin/bash

# ========================================================
# 1. 基础配置
# ========================================================
MASTER_ADDR=${MASTER_ADDR:-"192.168.0.3"}
MASTER_PORT=${MASTER_PORT:-"29501"}
# 你的 hostfile 配置
HOSTFILE="./script/deepspeed/hostfile"
# 确保所有卡都参与
INCLUDE_STR="192.168.0.3:0,1,2,3,4,5,6,7"

# ========================================================
# 2. 路径定义
# ========================================================
MODEL_TYPE="phi-1.5"
BASE_MODEL="/mnt/conda_data/microsoft/phi-1_5"
OUTPUT_DIR="./checkpoints-finetune/bunny-phi1.5-mixed-lora-695k"
# 关键：指向 Stage 1 跑出来的那个包含 117 个 Key 的文件
PRETRAIN_ADAPTER="/mnt/CoBunny/checkpoints-pretrain/bunny-phi1.5-mixed-pretrain/checkpoint-33300/mm_projector.bin"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export DS_SKIP_CUDA_CHECK=1
export DEEPSPEED_USE_TORCH_ADAM=1
export NCCL_DEBUG=INFO  # 开启调试模式，这样卡住时能看到为什么卡
export NCCL_SOCKET_IFNAME=eth0 
export GLOO_SOCKET_IFNAME=eth0
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=9600
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
# ========================================================
# 3. 启动训练 (Stage 2: Instruction Tuning)
# ========================================================
# 注意：这里我们使用 Zero-3 (如果显存够用 Zero-2 也可以，但 LoRA + 2M 数据建议 Zero-3 更稳)
# 增加了 --lora_enable 等参数，用的是经过精选后的数据

deepspeed \
    --hostfile $HOSTFILE \
    --include "$INCLUDE_STR" \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    bunny/train/train.py \
    --deepspeed ./script/deepspeed/zero2_mixencoders_finetune.json \
    --model_name_or_path $BASE_MODEL \
    --model_type $MODEL_TYPE \
    --version bunny \
    --data_path /mnt/conda_data/Bunny-v1.1-data/finetune/bunny_high_quality_final.json \
    --image_folder /mnt/conda_data/Bunny-v1.1-data/finetune/images \
    --vision_tower mixedencoder \
    --vision_tower_dino /mnt/facebook/dinov3-convnext-large-pretrain-lvd1689m \
    --vision_tower_siglip /mnt/siglip-so400m-patch14-384 \
    --pretrain_mm_mlp_adapter $PRETRAIN_ADAPTER \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --freeze_backbone False \
    --unfreeze_vision_tower True \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_bias "none" \
    --bf16 False \
    --fp16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to none 2>&1 | tee $OUTPUT_DIR/finetune.log


-------------------------------------------------
{
  "architectures": [
    "BunnyPhiForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 50256,
  "dtype": "float16",
  "embd_pdrop": 0.0,
  "eos_token_id": 50256,
  "freeze_mm_mlp_adapter": false,
  "hidden_act": "gelu_new",
  "hidden_size": 2048,
  "image_aspect_ratio": null,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "layer_norm_eps": 1e-05,
  "lora_enable": false,
  "max_position_embeddings": 2048,
  "mm_hidden_size": 1024,
  "mm_projector_lr": null,
  "mm_projector_type": "mlp2x_gelu",
  "mm_resampler_type": null,
  "mm_vision_select_feature": "patch",
  "mm_vision_select_layer": -1,
  "mm_vision_tower": "mixedencoder",
  "mm_use_im_start_end": false,
  "mm_use_im_patch_token": false,
  "image_token_index": -200,
  "model_type": "bunny-phi",
  "num_attention_heads": 32,
  "num_hidden_layers": 24,
  "num_key_value_heads": 32,
  "pad_token_id": 50256,
  "partial_rotary_factor": 0.5,
  "qk_layernorm": false,
  "resid_pdrop": 0.0,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "tokenizer_model_max_length": 2048,
  "tokenizer_padding_side": "right",
  "transformers_version": "4.57.1",
  "tune_mm_mlp_adapter": true,
  "unfreeze_vision_tower": true,
  "use_cache": false,
  "use_mm_proj": true,
  "use_s2": false,
  "vision_tower_dino": "/mnt/facebook/dinov3-convnext-large-pretrain-lvd1689m",
  "vision_tower_siglip": "/mnt/siglip-so400m-patch14-384",
  "vocab_size": 50295
}


------------------------
第二个阶段的训练代码如下
import os
from dataclasses import dataclass, field
import logging
import pathlib
from typing import Optional

import torch

import transformers

from bunny.train.bunny_trainer import BunnyTrainer

from bunny import conversation as conversation_lib
from bunny.model import *
from bunny.util.data_utils import make_supervised_data_module, DataArguments


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)  #选择何种LLM
    version: Optional[str] = field(default=None)  #选择何种对话模版
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    unfreeze_mm_vision_tower: bool = field(default=False)  
    vision_tower: Optional[str] = field(default=None)
    unfreeze_vision_tower: bool = field(default=False)
    use_s2: bool = field(default=False)  #是否使用S2
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')  #这个参数非常重要，它会指导如何建立投影层网络结构
    mm_resampler_type: Optional[str] = field(default=None) #采用何种重采样器
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    tune_mm_vision_resampler: bool = field(default=False)    
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_dense_connector_type: Optional[str] = field(default='dci')  #密集投影层类型
    vision_tower_dino: Optional[str] = field(default=None, metadata={"help": "DINOv2 子塔的权重路径"})
    vision_tower_siglip: Optional[str] = field(
        default=None, metadata={"help": "SigLIP 子塔的权重路径，例如 google/siglip-so400m-patch14-384"}
    )
    compression_K: int = field(default=8, metadata={"help": "ToMe 算法的压缩倍率"})
    mm_hidden_size: int = field(default=1024)



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    save_mm_vision_tower: bool = field(default=False) #增加一个是否保留视觉塔模型部分的参数
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.util.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def checkpoint_has_trainer_state(checkpoint_dir):
    return os.path.exists(os.path.join(checkpoint_dir, "trainer_state.json"))





def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    完整的、暴力可靠的权重保存函数。
    逻辑：
    1. 预训练阶段：自动抓取所有 requires_grad=True 的参数（含投影层和自定义融合层）。
    2. SFT 阶段：调用官方逻辑保存全量模型。
    """
    
    # 检查当前是否为“只练适配器”的预训练模式
    is_pretraining = getattr(trainer.args, "tune_mm_mlp_adapter", False)

    # ==========================================================
    # 场景 A: 预训练/对齐阶段 (只存增量参数)
    # ==========================================================
    if is_pretraining:
        if trainer.args.local_rank <= 0:
            print(f"\n[System] 启动暴力扫描保存模式...")

        # 暴力扫描：直接搜寻模型中所有开启了梯度的参数
        weight_to_save = {}
        for name, param in trainer.model.named_parameters():
            if param.requires_grad:
                # 兼容 DeepSpeed Zero2/Zero3，确保拿到 CPU 上的数据
                clean_data = torch.nan_to_num(param.data.detach().cpu(), nan=0.0, posinf=65500, neginf=-65500)
                weight_to_save[name] = clean_data.cpu()
              

        # 主进程负责物理写入磁盘
        if trainer.args.local_rank <= 0:
            # 1. 保存模型配置 (config.json)
            trainer.model.config.save_pretrained(output_dir)
            
            # 2. 保存增量权重 (mm_projector.bin)
            save_path = os.path.join(output_dir, "mm_projector.bin")
            torch.save(weight_to_save, save_path)
            
            # 3. 打印统计报告，确认是否漏掉 key
            vt_count = sum(1 for k in weight_to_save.keys() if 'vision_tower' in k)
            pj_count = sum(1 for k in weight_to_save.keys() if 'mm_projector' in k)
        # 预训练模式任务完成，直接返回，不再执行后续全量保存
        return

    # ==========================================================
    # 场景 B: 全量微调阶段 (SFT) 或 其它模式
    # ==========================================================
    
    # 兼容用户可能需要的独立 Vision Tower 保存开关
    if getattr(trainer.args, "save_mm_vision_tower", False):
        # 即使在全量微调，也可以单独拎出一份视觉塔权重
        vt_weights = {n: p.data.cpu() for n, p in trainer.model.named_parameters() if 'vision_tower' in n}
        if trainer.args.local_rank <= 0:
            torch.save(vt_weights, os.path.join(output_dir, 'vision_tower_standalone.bin'))

    # 执行 HuggingFace 官方的全量保存逻辑（保存数 GB 的 pytorch_model.bin）
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
    else:
        state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            trainer._save(output_dir, state_dict=cpu_state_dict)



def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))


    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    #跟序列的最大化长度相关，这里的padding同样最大长度max_length=10，输入7个token：,也就说model_max_length表示token的最大长度？？
    #当你输入的句子长度不足模型最大长度max_length时，需要用特殊的填充标记[PAD]把序列补齐到相同长度。这样，可以批量处理不等长的序列。
    assert model_args.vision_tower is not None
    if model_args.model_type in {'phi-1.5', 'phi-2', 'phi-3', 'qwen1.5-1.8b', 'minicpm', 'llama3-8b'}:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )
    elif model_args.model_type == 'stablelm-2':
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True
        )

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    if model_args.model_type == 'llama3-8b':
        tokenizer.eos_token_id = 128001  #该值不是随意确定的，而是对应模型词表中定义的特殊结束token。对于Llama3-8b模型，这个特殊token的id就是128001（根据模型词表和官方说明）。
        tokenizer.pad_token = tokenizer.eos_token 

    #看一下训练的时候，如何替代这些模型，任务13，非常重要，每一个模型都是多模态模型，因此，每一个模型都实现了类似于get_model().initialize_vision_modules(）
    #之类的函数，调用和得到对应的视觉编码器模块，重要的任务是在这里添加视觉或者模型块
    if model_args.model_type == 'phi-1.5' or model_args.model_type == 'phi-2':
        model = BunnyPhiForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **bnb_model_from_pretrained_args
        )
    elif model_args.model_type == 'phi-3':
        model = BunnyPhi3ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    elif model_args.model_type == 'stablelm-2':
        model = BunnyStableLMForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    elif model_args.model_type == 'qwen1.5-1.8b':
        model = BunnyQwen2ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    elif model_args.model_type == 'minicpm':
        model = BunnyMiniCPMForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    elif model_args.model_type == 'llama3-8b':
        model = BunnyLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **bnb_model_from_pretrained_args
        )
    else:
        raise ValueError(f"Unknown Model Type {model_args.model_type}")

    model.config.use_cache = False

    if model_args.freeze_backbone:   #是否冻结骨干
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()  #这是模型提供的一个方法，用来开启输入embedding层张量的requires_grad=True，允许对输入做梯度追踪。
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)  #这是模型中获取输入嵌入层（embedding layer）的接口，返回模型输入embedding模块，通常是一个nn.Embedding层


    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        # ---------------------------------------------------------
        # 🌟 关键加固：强制激活 LoRA 层梯度
        # ---------------------------------------------------------
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True # 确保 LoRA 层必开
            elif "mm_projector" in name:
                param.requires_grad = True # 确保投影层也必开
        # ---------------------------------------------------------

        # 打印一下，验证给学术论文看
        model.print_trainable_parameters()       


    #这段代码的作用正是为加载的大语言模型（LLM）选择对应的对话（聊天）模板
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["default"]



    # --- 在它下面插入这几行调试代码 ---
    rank0_print(f"\n" + "="*40)
    rank0_print(f"🔍 正在自检模板对齐情况...")
    rank0_print(f"🔥 命令行传入的 version: {model_args.version}")
    template_name = getattr(conversation_lib.default_conversation, 'version', 
                            getattr(conversation_lib.default_conversation, 'name', 'Unknown'))
    rank0_print(f"🔥 实际激活的模板名称: {template_name}")
    rank0_print(f"🔥 角色设定 (Roles): {conversation_lib.default_conversation.roles}")
    rank0_print(f"🔥 分隔符 (Sep): {repr(conversation_lib.default_conversation.sep)}")
    
    # 打印一个真实的预览，看看图片占位符和文字是怎么拼接的
    test_prompt = conversation_lib.default_conversation.get_prompt()
    rank0_print(f"🔥 模板预览:\n{test_prompt}")
    rank0_print("="*40 + "\n")


    model.get_model().initialize_vision_modules(model_args=model_args)
    model.resize_token_embeddings(len(tokenizer))

    # 2. 🛡️【核心修复】手动计算老词的均值，填补给新词
    input_embeddings = model.get_input_embeddings().weight
    output_embeddings = model.get_output_embeddings().weight
    # 计算老词（原生 50257 个词）的平均值
    # 这样新词就长得像老词一样，不会惊吓到模型
    current_size = input_embeddings.shape[0]
    SAFE_VOCAB_SIZE = 50257
    if current_size > SAFE_VOCAB_SIZE:
        rank0_print(f"🚨 检测到词表差异！当前: {current_size}, 原生安全区: {SAFE_VOCAB_SIZE}")
        with torch.no_grad():
            # 计算原生词表的均值
            in_avg = input_embeddings[:SAFE_VOCAB_SIZE].mean(dim=0, keepdim=True)
            out_avg = output_embeddings[:SAFE_VOCAB_SIZE].mean(dim=0, keepdim=True)
            # 【关键操作】：把 50257 之后的所有位置（不管是 38 个还是 900 个）全部初始化
            input_embeddings[SAFE_VOCAB_SIZE:] = in_avg
            output_embeddings[SAFE_VOCAB_SIZE:] = out_avg
            
        rank0_print(f"✅ 已清理并初始化 {current_size - SAFE_VOCAB_SIZE} 个潜在危险槽位。")



    if training_args.local_rank == 0:
        print("✅ 已手动初始化新增 Token！梯度爆炸隐患已清除。")

    # 3. 🛡️【双重保险】把所有参数强制转为 float32 进行一次清洗，再转回 float16
    # 这能保证即便刚才 resize 产生了细微的 NaN，也被洗掉了
    for p in model.parameters():
        if p.requires_grad:
            # 只处理参与训练的参数
            p.data = torch.nan_to_num(p.data, nan=0.0, posinf=65500, neginf=-65500)
    # ...
    #################### ⭐️ 插入调试代码 ⭐️ ####################
    if model_args.pretrain_mm_mlp_adapter:
        rank0_print("Checking mm_projector parameters after loading pretrain weights...")

        # 假设 mm_projector 至少有一个权重层 (比如 weight)
        mm_projector_first_weight = model.get_model().mm_projector.parameters().__next__()

        # 尝试计算该权重的L2范数或某个统计量，证明它不是随机初始化
        # 注意：这只在 local_rank 0 上安全，因为它需要同步
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            try:
                # 检查权重的范数，如果是一个加载的权重，它的值应该是非零且非极小的
                weight_norm = torch.linalg.norm(mm_projector_first_weight).item()
                rank0_print(f"✅ mm_projector first weight norm: {weight_norm:.4f}")
                if weight_norm < 1.0: # 经验值，加载的权重通常不会这么小
                    rank0_print("⚠️ Warning: Weight norm seems very small, check if weights were correctly loaded.")
            except Exception as e:
                rank0_print(f"❌ Error checking mm_projector weight norm: {e}")

    # ... (继续后面的 vision_tower.to(...) 等代码)
    ####################应该是在这里添加视觉编码器？？？？？    
    vision_tower = model.get_vision_tower()
    #设备移动：模型必须移动到指定的训练设备（通常是GPU），否则计算无法加速。
    # 该调用确保vision_tower使用正确的硬件资源和数据格式，为训练或推理做准备。
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    data_args.image_processor = vision_tower.image_processor
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    #的主要作用是实现微调时只训练模型中视觉多模态MLP适配器（mm_projector）部分，而冻结模型其余参数。具体含义说明如下
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        if not training_args.lora_enable:
            print("❄️ [System] 全量冻结 Backbone，仅练 Projector...")
            model.requires_grad_(False)
        else:
            print("🚀 [System] 检测到 LoRA 已开启，仅冻结非 LoRA 的 LLM 权重...")
            # 这种情况下不需要 model.requires_grad_(False)，因为 get_peft_model 内部已经处理好了
            pass
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        rank0_print("🔥 [Custom] Unfreezing AdaptiveConcatenationVisionTower fusion layers...")
        if hasattr(model.get_model(), "vision_tower"):
            print("🔥 Unfreezing custom fusion layers in Vision Tower...")

            rank0_print("🔥 [Custom] 正在精准解冻混合视觉塔融合层...")
            v_tower = model.get_model().get_vision_tower() # 使用 getter 比较安全
            for name, p in v_tower.named_parameters():
                if any(k in name for k in ['mlp_layers', 'cross_attn', 'cls_weights', 'pseudo', 'score_predictor']):
                    p.requires_grad = True
                    print(f"   -> Unfrozen: {name}")


    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_projector_lr = training_args.mm_projector_lr

    model.config.use_s2 = model_args.use_s2

    model.config.unfreeze_vision_tower = training_args.unfreeze_vision_tower = model_args.unfreeze_vision_tower
    #if training_args.unfreeze_vision_tower:
    #    for p in model.get_model().vision_tower.parameters():
    #        p.requires_grad = True


    if training_args.unfreeze_vision_tower:
            print("--- 🚀 尝试解冻视觉编码器参数 (Recipe-2) ---")
            # 实际解冻逻辑
            vision_tower = model.get_model().vision_tower
            for name, p in vision_tower.named_parameters():
                p.requires_grad = True
                


    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)


    ''' 
        #设置数据处理模块,这一部分是为了训练的时候，使用相关bunny数据集的
        data_module = make_supervised_data_module(tokenizer=tokenizer,
                                                data_args=data_args)

        # 2. 从训练集中切出一小部分作为验证集 (例如 2000 条，足够反映收敛情况)
        full_train_dataset = data_module['train_dataset']
        num_val_samples = 2000 
        num_train_samples = len(full_train_dataset) - num_val_samples

        # 使用 torch.utils.data.random_split 进行随机切分
        train_dataset, eval_dataset = torch.utils.data.random_split(
                                            full_train_dataset, 
                                            [num_train_samples, num_val_samples],
                                            generator=torch.Generator().manual_seed(42) # 固定随机种子，确保多机训练时行为一致
                                            )

        # 3. 更新 data_module
        data_module['train_dataset'] = train_dataset
        data_module['eval_dataset'] = eval_dataset
            
    ''''''''' 

    # 1. 直接调用修改后的函数，它会一次性返回切分好的训练集和验证集,这个是为了使用那个sharegpt4v的
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args
    )

    # 2. 原本在 train.py 里的 random_split 逻辑全部删掉
    # 因为我们在 data_utils.py 内部已经处理好了属性透传

    model.config.vision_tower_dino = model_args.vision_tower_dino
    model.config.vision_tower_siglip = model_args.vision_tower_siglip
    model.config.mm_projector_type = model_args.mm_projector_type
    model.config.model_type = model_args.model_type
    # 额外建议：把 lora_enable 也同步进去，虽然保存时我们会强制改它
    model.config.lora_enable = training_args.lora_enable

    #   返回dict(train_dataset=train_dataset,
    #            eval_dataset=None,
    #            data_collator=data_collator)

    #可以把data_collator看成是批整合器，把LazySupervisedDataset看成是，也就是train_dataset这个对象看成是如何每次训练获取样本的集中管理器
    #开启训练过程
    trainer = BunnyTrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           **data_module)
    

    if training_args.local_rank == 0 or training_args.local_rank == -1:
        rank0_print("\n" + "="*80)
        rank0_print("🔍 [Parameter Check] 正在扫描可训练参数...")
        rank0_print(f"{'参数名称':<60} | {'形状':<20} | {'梯度'}")
        rank0_print("-" * 95)
        
        # 1. 修复：必须先初始化计数器
        trainable_params_count = 0
        trainable_params = []
        
        for name, p in model.named_parameters():
            if p.requires_grad:
                trainable_params.append(name)
                trainable_params_count += 1
                # 2. 修复：变量名统一使用 p，而不是 param
                shape_str = str(list(p.shape))
                rank0_print(f"{name:<60} | {shape_str:<20} | {p.requires_grad}")
        
        rank0_print("-" * 95)
        rank0_print(f"📊 总计可训练参数项: {trainable_params_count}")
        
        # --- 逻辑验证 ---
        vision_tower_params = [n for n in trainable_params if "vision_tower" in n]
        projector_params = [n for n in trainable_params if "mm_projector" in n]
        
        fusion_found = len(vision_tower_params) > 0
        projector_found = len(projector_params) > 0
        
        if fusion_found and projector_found:
            rank0_print("🚀 状态确认：混合编码器融合层 和 Projector 已全部解冻！")
        else:
            if not fusion_found:
                rank0_print("❌ 警告：未发现 vision_tower 的可训练参数，请检查解冻逻辑！")
            if not projector_found:
                rank0_print("❌ 警告：未发现 mm_projector 的可训练参数！")
        
        rank0_print(f"🔍 逻辑明细：")
        rank0_print(f"   - 混合塔内部参数 (vision_tower): {len(vision_tower_params)} 项")
        rank0_print(f"   - 外部连接投影器 (mm_projector): {len(projector_params)} 项")
        rank0_print("="*80 + "\n")

# ==================== 🔍 更加稳健的自检 Debug 代码 ====================
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        print("\n" + "="*50)
        print("🚀 [Debug] 正在抽样检查喂给模型的数据格式...")
        
        try:
            # 获取一个 batch
            sample_batch = next(iter(trainer.get_train_dataloader()))
            
            # 1. 获取 Input IDs 并移至 CPU 转换为 list
            input_ids = sample_batch['input_ids'][0].detach().cpu().tolist()
            print(f"👉 [Input IDs 前 10 个 Token]: {input_ids[:10]}")
            if any(tid < 0 for tid in input_ids):
                print(f"✅ 发现特殊的 Image Token Index!")
            else:
                print(f"⚠️ 警告：Input IDs 里全是正数，说明 <image> 没被正确转换成特殊索引！")
            # 2. 检查 Labels 并处理 -100
            labels = sample_batch['labels'][0].detach().cpu().tolist()
            
            # 找到非 -100 的部分（即模型真正学习的部分）
            # 我们把 -100 过滤掉，或者替换成一个可见字符
            filtered_input_ids = [tid for tid in input_ids if tid >= 0]
            decoded_text = tokenizer.decode(filtered_input_ids, skip_special_tokens=False)

            # 找到模型计算 Loss 的部分
            loss_mask_tokens = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
            decoded_loss_part = tokenizer.decode(loss_mask_tokens, skip_special_tokens=False)

            print(f"\n👉 [完整输入流解码] (含 Image Token 占位符):\n{decoded_text[:1000]}") # 截断前1000字符防止刷屏
            print(f"\n👉 [计算 Loss 的文本内容]:\n{decoded_loss_part}")
            
            if 'images' in sample_batch:
                print(f"\n👉 [图像 Tensor 形状]: {sample_batch['images'].shape}")
            
            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            import traceback
            print(f"❌ [Debug] 抽样检查依然失败: {e}")
            traceback.print_exc()
    checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoints:
        # 选最近的checkpoint
        latest_ckpt = str(sorted(checkpoints)[-1])
        if checkpoint_has_trainer_state(latest_ckpt):
            print(f"Resuming from checkpoint: {latest_ckpt}")
            trainer.train(resume_from_checkpoint=latest_ckpt)
        else:
            print(f"Checkpoint {latest_ckpt} missing trainer_state.json, training from scratch.")
            trainer.train()
    else:
        print("No checkpoint found, training from scratch.")
        trainer.train()   

    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    #     trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    # 2. 只在主进程 (Rank 0) 执行全量合并保存
    if training_args.local_rank <= 0:
        print("📢 [全量保存启动] 正在收集分布式权重并物理合并 LoRA...")

        # 如果模型是 PeftModel (开启了 LoRA)
        if training_args.lora_enable:
            # 核心逻辑：物理合并
            # merge_and_unload 会把 BA 矩阵加回 W，并返回一个正常的 BunnyPhiForCausalLM 对象
            model = model.merge_and_unload()
            
            # 强制更新 config，关闭推理时的 lora 搜索，因为它已经合进去了
            model.config.lora_enable = False
            
            # 此时的 model.state_dict() 已经包含了：
            # - 合并后的全量 LLM 权重
            # - 微调后的 Projector 权重 (无需手动 replace)
            # - 微调后的 Vision Tower 权重
            
            # 保存整个文件夹
            model.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            
            print(f"✅ 全量模型已保存至: {training_args.output_dir}")
            print("ℹ️ 推理说明：直接使用 BunnyPhiForCausalLM.from_pretrained 加载此目录即可。")
        else:
            # 如果是全量微调，正常保存即可
            trainer.save_model()

if __name__ == "__main__":
    train()



----------------

#!/bin/bash

# ========================================================
# 1. 基础配置
# ========================================================
MASTER_ADDR=${MASTER_ADDR:-"192.168.0.3"}
MASTER_PORT=${MASTER_PORT:-"29501"}
# 你的 hostfile 配置
HOSTFILE="./script/deepspeed/hostfile"
# 确保所有卡都参与
INCLUDE_STR="192.168.0.3:0,1,2,3,4,5,6,7"

# ========================================================
# 2. 路径定义
# ========================================================
MODEL_TYPE="phi-1.5"
BASE_MODEL="./checkpoints-finetune/bunny-phi1.5-mixed-lora-695k/checkpoint-23476"
OUTPUT_DIR="./checkpoints-stage3/bunny-phi1.5-full-finetune"
DATA_PATH="/mnt/conda_data/Bunny-v1.1-data/finetune/bunny_llava_allava_2m.json"
IMAGE_PATH="/mnt/conda_data/Bunny-v1.1-data/finetune/images"
# 关键：指向 Stage 1 跑出来的那个包含 117 个 Key 的文件
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export DS_SKIP_CUDA_CHECK=1
export DEEPSPEED_USE_TORCH_ADAM=1
export NCCL_DEBUG=INFO  # 开启调试模式，这样卡住时能看到为什么卡
export NCCL_SOCKET_IFNAME=eth0 
export GLOO_SOCKET_IFNAME=eth0
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=9600
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
# ========================================================
# 3. 启动训练 (Stage 2: Instruction Tuning)
# ========================================================
# 注意：这里我们使用 Zero-3 (如果显存够用 Zero-2 也可以，但 LoRA + 2M 数据建议 Zero-3 更稳)
# 增加了 --lora_enable 等参数，用的是经过精选后的数据

deepspeed \
    --hostfile $HOSTFILE \
    --include "$INCLUDE_STR" \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    bunny/train/train_stage3.py \
    --deepspeed ./script/deepspeed/zero3_mixedencoders_full.json \
    --model_name_or_path $BASE_MODEL \
    --model_type $MODEL_TYPE \
    --version bunny \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --vision_tower mixedencoder \
    --vision_tower_dino /mnt/facebook/dinov3-convnext-large-pretrain-lvd1689m \
    --vision_tower_siglip /mnt/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --freeze_backbone False \
    --unfreeze_vision_tower True \
    --lora_enable False \
    --bf16 False \
    --fp16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to none 2>&1 | tee $OUTPUT_DIR/finetunesharegpt.log

------------------------------------------------------
也就是说永远不要手写模版
from bunny import conversation as conversation_lib
from bunny.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from bunny.util.mm_utils import tokenizer_image_token
import torch

def generate_inference_input(question, tokenizer, device="cuda"):
    # 1. 复制官方预定义的模板 (一定要选对名字，通常是 "phi" 或 "bunny")
    # 对应你 data_utils.py 里的 conversation_lib.default_conversation.copy()
    conv = conversation_lib.default_conversation.copy()
    
    # 2. 构造符合规范的消息格式
    # 注意：DEFAULT_IMAGE_TOKEN 会被 preprocess_multimodal 替换为 <img_content>
    image_token = DEFAULT_IMAGE_TOKEN 
    message = f"{image_token}\n{question}"
    
    # 3. 按照对话轮次填充内容
    # roles[0] 是 Human/User, roles[1] 是 GPT/Assistant
    conv.append_message(conv.roles[0], message)
    conv.append_message(conv.roles[1], None)
    
    # 4. 获取最终的 Prompt 字符串
    # 这步会生成你 Log 里看到的 "A chat between a curious user..." 完整文本
    prompt = conv.get_prompt()
    
    # --- 调试打印：确认生成的字符串是否带有了 System Prompt ---
    # print(f"DEBUG PROMPT: {repr(prompt)}")
    
    # 5. 使用专用的 tokenizer 转换成 input_ids
    # 这样能确保图像占位符被正确识别，且 BOS 逻辑被处理
    input_ids = tokenizer_image_token(
        prompt, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors='pt'
    ).unsqueeze(0).to(device)
    
    return input_ids, prompt

# 使用方法：
question = "What is in the image?"
input_ids, final_prompt = generate_inference_input(question, tokenizer)