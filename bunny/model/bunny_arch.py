from abc import ABC, abstractmethod
import torch
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector
from bunny.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
import os
import glob

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

class BunnyMetaModel:
    def __init__(self, config):
        super(BunnyMetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            model_path = getattr(config, "_name_or_path", "")
            print(f" 🔍 [BunnyMetaModel 探测] 路径: {model_path}")
            self.stage = getattr(config, "training_stage", "inference") #先去找这个配置文件中的training_stage 得到它的值。如果没有的话，才是inference
            is_full_weight_checkpoint = False
            if model_path and os.path.isdir(model_path):
                # 兼容方案：只要有任何形式的权重文件存在，就视为全量 Checkpoint
                has_sharded = len(glob.glob(os.path.join(model_path, "pytorch_model-*.bin"))) > 0
                has_single_bin = os.path.exists(os.path.join(model_path, "pytorch_model.bin"))
                has_safetensors = os.path.exists(os.path.join(model_path, "model.safetensors"))
                if has_sharded or has_single_bin or has_safetensors:
                    is_full_weight_checkpoint = True
            if is_full_weight_checkpoint:  
                print("🏗️  检测到全量权重文件，强制构造视觉塔实体 (delay_load=False)...")
                delay_load = False
            else:
                # 只有在非全量 Checkpoint（如只存了 Projector 的 Stage 1/2）时才延迟加载
                delay_load = not getattr(config, 'continuous_training', False)

            ###微调（Finetune）和推理（Inference）这两阶段，操作是一模一样的：都是只搭一个架子
            if self.stage in ["finetune", "inference"]:
                delay_load = False  # 别延迟，立刻搭架子，迎接15GB权重  微调或推理：HF 的权重马上要进场，必须立刻搭好桶（Skeleton）
            else:
                delay_load = True   # 第一阶段预训练，可以偷懒，等官方送货等用到图片再去拉取官方权重。

            self.vision_tower = build_vision_tower(config, delay_load=delay_load, training_stage=self.stage)
            self.vision_resampler = build_vision_resampler(config, delay_load=delay_load,training_stage=self.stage)
            self.mm_projector = build_vision_projector(config, delay_load=delay_load, training_stage=self.stage)
            if getattr(config, 'continuous_training', False):
                config.continuous_training = False         


    #注意这里写法，其实不是获取命令行中的字符串
    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)  #这只是从self对象拿到vision_tower属性（模型对象），如果是list则取第一个，否则原样返回
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower # 也就说我们的双塔视觉编码器返回的是自己本身
    
    def initialize_vision_modules_stage3(self, model_args):
        """
        [Stage 3 专用版] 视觉模块初始化逻辑
        核心目标：
        1. 确保在 DeepSpeed 加载模型后，视觉塔架构已建立。
        2. 触发防御性加载机制（优先使用内存中来自 Stage 1 的权重）。
        3. 强制开启全量梯度（Full Fine-tuning）。
        """
        vision_tower_name = model_args.vision_tower
        self.config.mm_vision_tower = vision_tower_name
        
        # 1. 获取或创建视觉塔对象 (The Skeleton)
        vision_tower = self.get_vision_tower()
        
        if vision_tower is None:
            # 这种情况通常发生在没有通过 from_pretrained 加载，或者配置丢失时
            print(f"🏗️  [Stage 3] 视觉塔对象不存在，正在根据配置创建架构: {vision_tower_name}")
            vision_tower = build_vision_tower(model_args)
            self.vision_tower = vision_tower
        elif isinstance(vision_tower, str):
            # 兼容逻辑：如果 vision_tower 属性只是个路径字符串
            print(f"🏗️  [Stage 3] 探测到路径字符串，正在实例化视觉塔对象...")
            vision_tower = build_vision_tower(model_args)
            self.vision_tower = vision_tower

        # 2. 触发防御性权重加载 (The Soul)
        # 调用 AdaptiveConcatenationVisionTower 的 load_model()
        # 内部的 check_tower_valid 会判断：
        #   - 如果 Stage 1 权重已在内存：跳过官方权重，保护微调成果。
        #   - 如果是全新编码器（如 OpenVision2）：加载其官方底座。
        if hasattr(vision_tower, 'load_model'):
            print(f"💉 [Stage 3] 触发视觉塔防御性检测与加载逻辑...")
            vision_tower.load_model()

        # 3. 精度转换与设备对齐
        # 注意：在全解冻模式下，必须确保所有参数都在正确的设备和精度上
        compute_dtype = torch.float16 if getattr(model_args, 'fp16', False) else torch.bfloat16
        vision_tower.to(dtype=compute_dtype, device='cuda')

        # 4. 【核心区别】全量解冻 (Activate All Gradients)
        # Stage 3 的定义就是 Full Tuning，所以这里不再判断 freeze_mm_vision_tower
        print("🔥 [Stage 3] 正在解锁视觉塔全量参数梯度...")
        for p in vision_tower.parameters():
            p.requires_grad = True

        # 5. Projector 初始化与解冻
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'mlp2x_gelu')
        self.config.mm_hidden_size = vision_tower.hidden_size
        
        if getattr(self, 'mm_projector', None) is None:
            print("🏗️  正在构建 Projector...")
            self.mm_projector = build_vision_projector(self.config)
        
        # 强制解冻 Projector
        print("🔥 [Stage 3] 正在解锁 Projector 全量参数梯度...")
        for p in self.mm_projector.parameters():
            p.requires_grad = True

        # 6. 【Stage 3 特殊逻辑】忽略外部 adapter 文件
        # 在 Stage 3 中，我们直接使用 BASE_MODEL (Stage 1 checkpoint) 里的 model.safetensors。
        # 因此，通常不需要手动加载 pretrain_mm_mlp_adapter。
        # 只有当你发现某些 Key 没对上时，才需要手动补载，这里我们保持默认信任主模型文件。
        if model_args.pretrain_mm_mlp_adapter is not None:
            print("⚠️  [Stage 3 Warning] 探测到外部 Adapter 路径，但将优先使用主模型权重。")
            # 如果你确实需要从一个单独的 bin 加载 Projector，可以在这里保留你原有的 torch.load 逻辑

        print("✅ [Stage 3] 视觉模块初始化完成，准备进行全解冻微调。")


    def initialize_vision_modules_stage3_fsdp(self, model_args):
        vision_tower_name = model_args.vision_tower
        self.config.mm_vision_tower = vision_tower_name
        
        vision_tower = self.get_vision_tower()
        
        if vision_tower is None:
            vision_tower = build_vision_tower(model_args)
            self.vision_tower = vision_tower
        
        if hasattr(vision_tower, 'load_model'):
            vision_tower.load_model()

        # --- 关键修改：严禁在 FSDP 下调用 .to('cuda') ---
        # 仅设置精度，不搬运设备
        compute_dtype = torch.float16 if getattr(model_args, 'fp16', False) else torch.bfloat16
        # 只要不带 device='cuda'，.to(dtype) 在 FSDP 下是安全的
        vision_tower.to(dtype=compute_dtype) 

        # 解冻梯度
        for p in vision_tower.parameters():
            p.requires_grad = True

        # Projector 初始化
        self.config.use_mm_proj = True
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        
        for p in self.mm_projector.parameters():
            p.requires_grad = True    


    def initialize_vision_modules(self, model_args):
        """
        🚀 专门用于 Stage 1 (Pretrain) 的视觉模块初始化逻辑。
        """
        self.stage = "pretrain"
        vision_tower_name = model_args.vision_tower
        self.config.mm_vision_tower = vision_tower_name
        
        print(f"🔥 [Stage 1 Pretrain] 正在启动视觉系统初始化...")

        # 1. 构建视觉塔实体
        # 💡 修正点：直接通过 self.get_vision_tower() 检查，不需要 get_model()
        if self.get_vision_tower() is None:
            print("🚨 [Stage 1] 检测到视觉塔为空，正在构建实体...")
            vision_tower = build_vision_tower(model_args, delay_load=True, training_stage="pretrain")
            
            # 💡 修正点：直接赋值给 self
            self.vision_tower = vision_tower 
        else:
            vision_tower = self.get_vision_tower()
            print("🧊 [Stage 1] 视觉塔实体已存在。")

        # 2. 构建重采样器
        # 💡 修正点：直接通过 self 检查
        if getattr(self, 'vision_resampler', None) is None:
            print("🛠️  正在创建视觉重采样器 (Resampler)...")
            vision_resampler = build_vision_resampler(model_args, delay_load=True, training_stage="pretrain")
            
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)
            
            # 💡 修正点：直接赋值给 self
            self.vision_resampler = vision_resampler

        # 3. 🧠 强制物理加载驱动
        if hasattr(vision_tower, 'load_model'):
            print("⚡ [Power On] 正在从官方路径拉取 DINO/SigLIP 原始权重...")
            vision_tower.load_model()

        # 4. 构建投影层 (Projector)
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'mlp2x_gelu')
        self.config.mm_hidden_size = vision_tower.hidden_size
        
        if getattr(self, 'mm_projector', None) is None:
            print(f"🛠️  正在创建全新的投影层: {self.config.mm_projector_type}")
            self.mm_projector = build_vision_projector(self.config)
        
        # 5. 状态设置
        vision_tower.to(dtype=torch.float16, device='cuda')
        
        unfreeze_vt = getattr(model_args, "unfreeze_mm_vision_tower", False)
        for p in vision_tower.parameters():
            p.requires_grad = unfreeze_vt
            
        for p in self.mm_projector.parameters():
            p.requires_grad = True

        print(f"✅ [Stage 1] 初始化完毕！视觉特征维度: {self.config.mm_hidden_size}")

class BunnyMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def encode_images(self, images):
        #这里可以来控制,如果不是dynamic 
        vision_tower = self.get_model().get_vision_tower()

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if "AdaptiveConcatenationVisionTower" in str(type(vision_tower)):
            image_features, _ = vision_tower(images)
            image_features = self.get_model().mm_projector(image_features)
            return image_features
        mm_resampler_type = getattr(self.config, 'mm_resampler_type', None)

        if mm_resampler_type is None:  # 常规处理模式, 这里我们希望的
            image_features, _ = self.get_model().get_vision_tower()(images)  #这里是希望能返回中间层特征
            image_features = self.get_model().mm_projector(image_features)
            return image_features
        else:  #如果是那几个
            if mm_resampler_type=='dynamic_compressor':
                image_features, image_size, _ = self.get_model().get_vision_tower()(images)
                image_features,_ = self.get_model().vision_resampler(image_features, forward_type='image',image_size=image_size)
                image_features = self.get_model().mm_projector(image_features)
                return image_features
            else:
                image_features = self.get_model().get_vision_tower()(images)
                image_features = self.get_model().vision_resampler(image_features)
                image_features = self.get_model().mm_projector(image_features)
                return image_features    
                
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        # 1. 基础检查：推理阶段的流式生成直接跳过逻辑
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # --- [打印点 1：检查输入图像维度] ---
        # 确保进入视觉塔前，images 是你预期的 [B, 6, 2, 3, 378, 378]
        if local_rank == 0 and self.training:
            print(f"DEBUG: [Step 1] Input Images Shape: {images.shape if hasattr(images, 'shape') else 'List'}")

        # 2. 图像特征编码：你的混合塔直接吞 6D 吐 [B, 365, 1024]
        if isinstance(images, list) or images.ndim == 5:
            # 兼容多图输入情况
            concat_images = torch.cat([image for image in images], dim=0)
            raw_features = self.encode_images(concat_images)
        else:
            # 你的标准 AnyRes 路径
            raw_features = self.encode_images(images)
    
        # 3. 维度检查与拆分：确保每个样本独立
        if raw_features.ndim == 3: # [Batch, 365, 1024]
            image_features = [raw_features[i] for i in range(raw_features.shape[0])]
        else:
            image_features = raw_features # 已经是 list 则保持

        # DEBUG：确认视觉塔出来的东西对不对
        if local_rank== 0: # 只在主进程打印
            print(f"DEBUG: image_features type: {type(image_features)}")
            if isinstance(image_features, list):
                print(f"DEBUG: image_features[0] shape: {image_features[0].shape}")
            else:
                print(f"DEBUG: image_features shape: {image_features.shape}")

        # 4. 文本对齐处理
        _labels = labels
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # 改 -200 为 0 防止嵌入层报错
        input_ids_temp = input_ids.clone() 
        input_ids_temp[input_ids_temp == IMAGE_TOKEN_INDEX] = 0
        # --- [玛德，这就给你打印 input_ids_temp] ---
# # --- 强制打印调试法 ---
#         import torch.distributed as dist
#         # 获取当前进程的 rank，不依赖那个可能没赋值成功的 local_rank
#         try:
#             curr_rank = dist.get_rank()
#         except:
#             curr_rank = 0 # 非分布式模式

#         if curr_rank == 0:
#             # 使用内置 print，加上 flush=True 确保立即输出到屏幕
#             print(f"\n📊 [input_ids_temp 监控]", flush=True)
#             print(f"🔹 形状 (Shape): {input_ids_temp.shape}", flush=True)
            
#             # 看看第一条数据前 100 个 token
#             sample_0 = input_ids_temp[0, :100].tolist()
#             print(f"🔹 样本 0 前 100 个内容: {sample_0}", flush=True)
            
#             # 统计 0 的个数
#             num_zeros = (input_ids_temp == 0).sum().item()
#             print(f"🔹 当前 Batch 中 Token '0' (原-200) 的总数: {num_zeros}", flush=True)
            
#             # 顺便确认视觉特征有没有货
#             if len(image_features) > 0:
#                 print(f"📸 视觉特征已就绪，当前组数: {len(image_features)}", flush=True)
#             print("-" * 40, flush=True)

        # 去掉 Padding，转为变长列表，准备缝合
        input_ids_list = [cur_input_ids[cur_mask] for cur_input_ids, cur_mask in zip(input_ids, attention_mask)]
        labels_list = [cur_labels[cur_mask] for cur_labels, cur_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # --- 玛德，这是最关键的安全检查补丁 ---
        total_image_placeholders = sum((x == IMAGE_TOKEN_INDEX).sum().item() for x in input_ids_list)
        
        if local_rank == 0:
            print(f"🔍 [6D 缝合检查] 文本坑位: {total_image_placeholders}, 视觉特征块: {len(image_features)}")
            if len(image_features) > 0:
                 print(f"🔍 [特征维度] 单个块形状: {image_features[0].shape}") # 必须是 [365, 1024]

        # --- 替换结束 ---
        # ------------------------------------
        # 5. 核心缝合逻辑：将 365 个 Token 塞进每一个 IMAGE_TOKEN_INDEX 位置
        for batch_idx, cur_input_ids in enumerate(input_ids_list):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            cur_labels = labels_list[batch_idx]
            
            if num_images == 0:
                # 纯文本样本
                new_input_embeds.append(self.get_input_embeddings()(cur_input_ids))
                new_labels.append(cur_labels)
                continue

            # 寻找切口
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_new_input_embeds = []
            cur_new_labels = []
            
            for i in range(num_images + 1):
                # A. 提取并转换文本段向量
                text_seg = cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i+1]]
                label_seg = cur_labels[image_token_indices[i] + 1 : image_token_indices[i+1]]
                if text_seg.shape[0] > 0:
                    cur_new_input_embeds.append(self.get_input_embeddings()(text_seg))
                    cur_new_labels.append(label_seg)
                
                # B. 插入图像特征 (直接插入视觉塔返回的 365 个 tokens)
                if i < num_images:
                    cur_feat = image_features[cur_image_idx]
                    # --- 添加以下打印信息 ---
                    if local_rank == 0:
                         print(f"🚀 [物理注入] 样本 {batch_idx} 的第 {cur_image_idx} 组特征正在缝入第 {i} 个坑位")
                    cur_image_idx += 1
                    
                    cur_new_input_embeds.append(cur_feat)
                    # 图像对应的 Labels 全部设为 IGNORE_INDEX (-100)
                    cur_new_labels.append(
                        torch.full((cur_feat.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)
                    )

            # 拼接当前样本的所有片段
            new_input_embeds.append(torch.cat(cur_new_input_embeds))
            new_labels.append(torch.cat(cur_new_labels))

        # 6. 🛡️ 截断与 Padding：重新合体为标准 Batch 张量
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', 2048)
        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        
        target_dtype = new_input_embeds[0].dtype
        target_device = new_input_embeds[0].device
        
        # 初始化 Padding 容器
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=labels.dtype, device=target_device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=target_device)
        position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=target_device)
        new_input_embeds_padded = []

        for i, (cur_embed, cur_label) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_embed.shape[0]
            # 统一右填充
            padding_size = max_len - cur_len
            new_input_embeds_padded.append(torch.cat((
                cur_embed,
                torch.zeros((padding_size, cur_embed.shape[1]), dtype=target_dtype, device=target_device)
            ), dim=0))
            
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_label
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=torch.long, device=target_device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)


        if new_input_embeds.dtype == torch.float16:
            # 第一步：物理检查与填充
            if torch.isnan(new_input_embeds).any() or torch.isinf(new_input_embeds).any():
                if local_rank == 0:
                    print("⚠️ [Warning] 捕获到 NaN/Inf，执行紧急数值置换...")
                # 将异常值直接归零，防止污染整个 Batch
                new_input_embeds = torch.nan_to_num(new_input_embeds, nan=0.0, posinf=4096.0, neginf=-4096.0)
            
            # 第二步：强力限幅 (将阈值从 16384 压低到 4096)
            # 理由：Phi-1.5 的 Hidden Size 较小，4096 已经足够表达特征，
            # 留出更大的余量给后面的 Transformer 层计算。
            new_input_embeds = torch.clamp(new_input_embeds, min=-4096.0, max=4096.0)

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels_padded

