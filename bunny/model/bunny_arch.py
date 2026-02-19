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

#很明显要根据之前LLM基座的分隔符来设计
class VisualTokenStructurer:
    def __init__(self, config, embed_tokens_fn):
        self.config = config
        self.embed_tokens = embed_tokens_fn
        # 1. 根据你的探测，Phi-1.5 的换行符是 198
        self.newline_id = 198 
        # 2. 如果你想做更强的转场（比如全局到局部），可以用 ### (21017)
        self.sep_id = 21017 
        
    def __call__(self, visual_features):
        """
        输入: [358, 1024]
        输出: [365, 1024]
        """
        device = visual_features.device
        
        # 准备向量 (必须从 LLM 的 Embedding 层拿，才能保持语义空间一致)
        nl_emb = self.embed_tokens(torch.tensor([self.newline_id], device=device)) # [1, 1024]
        
        # 按照你的混合塔设计切分
        soul = visual_features[0:4, :]          # 4
        base = visual_features[4:148, :]        # 144
        detail = visual_features[148:358, :]    # 210
        
        # 组合成 365
        # 策略：Soul(4) + \n(1) + Base(144) + \n(1) + Detail(210) + \n*5(5)
        # 这里的 5 个换行是作为“图像结束”的标志，防止文本瞬间贴上来
        res = torch.cat([
            soul, 
            nl_emb, 
            base, 
            nl_emb, 
            detail, 
            nl_emb.repeat(5, 1)
        ], dim=0)
        
        return res


# 1. continuous_training 的核心逻辑是什么？
# 在你的代码中，这个参数决定了视觉塔权重的“初始来源”：

# 如果为 False (默认状态 - 延迟加载模式)：

# delay_load 变成 True。

# 意图：视觉塔对象先被创建出来（空架子），但不立即去读磁盘上的官方大模型文件。它在等你通过 from_pretrained 加载你自己的 model.safetensors（即 Stage 1 训练好的权重）。

# 场景：Stage 2、Stage 3 或者是推理阶段。

# 如果为 True (立即加载模式)：

# delay_load 变成 False。

# 意图：强制视觉塔在初始化时立即去 /mnt/facebook/... 加载官方原始权重。

# 场景：全新的 Stage 1 开始，此时你没有任何自己的 Checkpoint，必须靠官方权重初始化。
#直接强制选择这个indeity投影，直接在这个里面，在增加一个开源的openvision2的编码器，并初始化装载
class BunnyMetaModel:
    def __init__(self, config):
        super(BunnyMetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            model_path = getattr(config, "_name_or_path", "")
            print(f" 🔍 [BunnyMetaModel 探测] 路径: {model_path}")
            
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
            
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            
            if getattr(config, 'continuous_training', False):
                config.continuous_training = False
            self.mm_projector = build_vision_projector(config)
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

    def initialize_vision_modules(self, model_args):
        vision_tower = model_args.vision_tower
        #这个地方传递进了模型的视觉塔名称，我们要在这里加入
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter #这个其实就是预训练的适配器,在第二个阶段
        self.config.mm_vision_tower = vision_tower
        if self.get_vision_tower() is None:
            print("🚨🚨🚨 警告：进入了创建新塔的分支！！！！！") # 看看训练开始时这一行会不会打印
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)
            self.vision_tower = vision_tower            #给自身的vision_tower对象属性赋值
            self.vision_resampler = vision_resampler           
        else:
            # 1. 即使检测到有权重，也要调用 load_model！, 这个地方在推理的时候，可能存在一定的bug，因为会重新加载模型官方模型
            # 因为 load_model 里现在有“防御性逻辑”，它会自己判断是“跳过加载”还是“执行加载”
            # 关键是：它会执行 _set_subtower_grad_state() 来锁定 eval 模式
            current_vt = self.get_vision_tower()

            # 【核心修正】：如果它是 None，或者是字符串，说明都需要“实例化”
            if current_vt is None or isinstance(current_vt, str):
                # 走创建逻辑
                vision_tower = build_vision_tower(model_args)
                # ... 其他 build 逻辑 ...
                self.get_model().vision_tower = vision_tower # 确保存进 model 里的属性
            else:
                # 说明已经是一个真正的模型对象了（推理或重用场景）
                vision_tower = current_vt
                # 既然是对象，现在调用这些方法才是安全的
                if hasattr(vision_tower, 'load_model'):
                    vision_tower.load_model()
            
        # 此时再统一设置梯度和转换精度，就再也不会报错了
        vision_tower.to(dtype=torch.float16, device='cuda')
        for p in vision_tower.parameters():
            p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type')
        self.config.mm_hidden_size = vision_tower.hidden_size #投影层的输入特征维度要等于视觉编码器的特征维度
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True
        if pretrain_mm_mlp_adapter is not None and os.path.exists(pretrain_mm_mlp_adapter):
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_projector_w(weights):
                # 逻辑：如果有前缀就切前缀，没前缀且是数字开头就直接返回
                new_dict = {k.split('mm_projector.')[1]: v for k, v in weights.items() if 'mm_projector.' in k}
                if not new_dict and any(k.split('.')[0].isdigit() for k in weights.keys()):
                    return weights
                return new_dict

            self.mm_projector.load_state_dict(get_projector_w(mm_projector_weights))
            print("✅ [Success] Projector weights loaded.........................")

            # 2. 加载视觉塔融合层 (Vision Tower)
            vt_tuned_path = pretrain_mm_mlp_adapter.replace('mm_projector.bin', 'vision_tower_tuned.bin')
            if os.path.exists(vt_tuned_path):
                vt_weights = torch.load(vt_tuned_path, map_location='cpu')

                def get_vision_tower_w(weights):
                    # 逻辑：如果有 vision_tower. 前缀就切掉
                    new_dict = {k.split('vision_tower.')[1]: v for k, v in weights.items() if 'vision_tower.' in k}
                    # 【核心修改点】：如果没有前缀，但 key 包含你定义的融合层关键字（如 mlp_layers）
                    # 这种情况直接返回整个 weights，不要去管什么 0. 开头
                    if not new_dict:
                        fusion_keywords = ['mlp_layers', 'cross_attn', 'cls_weights', 'pseudo', 'score_predictor']
                        if any(any(kw in k for kw in fusion_keywords) for k in weights.keys()):
                            return weights
                    return new_dict

                vt_data = get_vision_tower_w(vt_weights)
                msg = self.get_vision_tower().load_state_dict(vt_data, strict=False)
                print(f"✅ [Success] Vision Tower Tuned loaded. Status: {msg}")

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
            


    # 其实代码里是有批次的。但因为 LLM 处理的是变长序列（不同句子的 Token 长度不一样），所以它采用了**“先打散、再组装、最后填充（Padding）”**的策略：
    # 打散（Flatten）：代码通过 attention_mask 把 Batch 里的每个样本取出来，变成一个一个独立的“变长列表”。
    # 图像替换：在一个循环里（for batch_idx, cur_input_ids in enumerate(input_ids):），逐个样本寻找图像占位符（IMAGE_TOKEN_INDEX），并把对应的视觉向量插进去。
    # 重新合体：在代码最后（new_input_embeds_padded 部分），它会找到这一批样本中最长的那一个，然后把其他短样本用 0 补齐，
    # 重新叠成一个 [Batch_Size, Max_Len, Hidden_Size] 的标准三维张量送给 LLM。

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels, images
        ):
            # ==========================================================
            # 1. 🔥【内核硬对齐】强制使用合法 ID 50295
            # ==========================================================
            image_token_index = 50295
            self.config.image_token_index = image_token_index
            IGNORE_INDEX = -100

            # 调试日志：确保内核此时看到的 input_ids 包含 50295
            # print(f"🕵️ [内核实测] input_ids: {input_ids[0].tolist()}")

            vision_tower = self.get_vision_tower()
            
            # --- 情况 A: 推理中的流式生成阶段 (KV Cache 阶段) ---
            # 如果是生成第二个字开始，input_ids 长度为 1，直接跳过图像处理
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

            # ==========================================================
            # 2. 视觉特征提取与格式化
            # ==========================================================
            if isinstance(images, list) or images.ndim == 5:
                concat_images = torch.cat([image for image in images], dim=0)
                raw_features = self.encode_images(concat_images)
            else:
                raw_features = self.encode_images(images)

            # 确保 image_features 是一个包含 [358, 1024] 元素的列表
            if raw_features.ndim == 3:
                image_features = [raw_features[i] for i in range(raw_features.shape[0])]
            else:
                image_features = raw_features 

            # ==========================================================
            # 3. 准备基础变量
            # ==========================================================
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                attention_mask = attention_mask.bool()
            
            if position_ids is None:
                position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            
            if labels is None:
                labels = torch.full_like(input_ids, IGNORE_INDEX)

            # 去掉 Padding，转为 List 进行变长处理
            input_ids_list = [cur_input_ids[cur_mask] for cur_input_ids, cur_mask in zip(input_ids, attention_mask)]
            labels_list = [cur_labels[cur_mask] for cur_labels, cur_mask in zip(labels, attention_mask)]

            # --- 关键检查：验证占位符数量 ---
            total_image_placeholders = sum((x == image_token_index).sum().item() for x in input_ids_list)
            if len(image_features) != total_image_placeholders:
                raise ValueError(f"特征数量({len(image_features)})与占位符数量({total_image_placeholders})不匹配！"
                                f"请检查输入文本是否正确包含了 <img_content> 标签。")

            # ==========================================================
            # 4. 🔥【核心手术】执行文本与 365 图像特征的缝合
            # ==========================================================
            token_sewer = VisualTokenStructurer(self.config, self.get_input_embeddings())
            new_input_embeds = []
            new_labels = []
            cur_image_idx = 0

            for batch_idx, cur_input_ids in enumerate(input_ids_list):
                num_images = (cur_input_ids == image_token_index).sum()
                cur_labels = labels_list[batch_idx]
                
                if num_images == 0:
                    # 纯文本样本
                    new_input_embeds.append(self.get_input_embeddings()(cur_input_ids))
                    new_labels.append(cur_labels)
                    continue

                # 寻找所有 50295 出现的切口位置
                image_token_indices = [-1] + torch.where(cur_input_ids == image_token_index)[0].tolist() + [cur_input_ids.shape[0]]
                cur_new_input_embeds = []
                cur_new_labels = []

                for i in range(num_images + 1):
                    # A. 切出文本片段并转向量
                    # 关键：切片时要 clone 并在映射前将 50295 归零防止 Embedding 层报错
                    text_seg = cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i+1]].clone()
                    label_seg = cur_labels[image_token_indices[i] + 1 : image_token_indices[i+1]]
                    
                    if text_seg.shape[0] > 0:
                        # 安全防御：将文本片段中的占位符 ID 抹掉（虽然理论上片段里不该有）
                        text_seg[text_seg == image_token_index] = 0 
                        cur_new_input_embeds.append(self.get_input_embeddings()(text_seg))
                        cur_new_labels.append(label_seg)
                    
                    # B. 插入图像特征：在这里执行 358 -> 365 的转换
                    if i < num_images:
                        raw_feat = image_features[cur_image_idx] 
                        cur_image_idx += 1
                        
                        # 使用 VisualTokenStructurer 进行 358 -> 365 缝合
                        structured_features = token_sewer(raw_feat) 
                        
                        cur_new_input_embeds.append(structured_features)
                        cur_new_labels.append(
                            torch.full((365,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)
                        )

                # 拼接当前样本的所有片段
                new_input_embeds.append(torch.cat(cur_new_input_embeds))
                new_labels.append(torch.cat(cur_new_labels))

            # ==========================================================
            # 5. 后处理：截断、对齐与 Padding
            # ==========================================================
            # 🛡️ 截断：防止超过模型最大长度（默认 2048）
            tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', 2048)
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

            # 计算 Padding 后的统一长度
            max_len = max(x.shape[0] for x in new_input_embeds)
            batch_size = len(new_input_embeds)
            
            # 初始化最终返回的张量容器
            final_input_embeds = torch.zeros(
                (batch_size, max_len, new_input_embeds[0].shape[-1]), 
                dtype=new_input_embeds[0].dtype, 
                device=new_input_embeds[0].device
            )
            final_labels = torch.full(
                (batch_size, max_len), IGNORE_INDEX, 
                dtype=new_labels[0].dtype, 
                device=new_labels[0].device
            )
            final_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=new_labels[0].device)
            final_position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=new_labels[0].device)

            # 填充数据
            for i, (cur_embed, cur_label) in enumerate(zip(new_input_embeds, new_labels)):
                cur_len = cur_embed.shape[0]
                final_input_embeds[i, :cur_len] = cur_embed
                final_labels[i, :cur_len] = cur_label
                final_attention_mask[i, :cur_len] = True
                final_position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=torch.long, device=cur_label.device)

            # 最终稳定性检查
            final_input_embeds = torch.nan_to_num(final_input_embeds, nan=0.0)

            # 根据 Transformers 规范，返回 embeds 时，input_ids 必须为 None
            return None, final_position_ids, final_attention_mask, past_key_values, final_input_embeds, final_labels

