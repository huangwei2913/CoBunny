from abc import ABC, abstractmethod
import torch
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector
from bunny.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
import os

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


#直接强制选择这个indeity投影，直接在这个里面，在增加一个开源的openvision2的编码器，并初始化装载
class BunnyMetaModel:
    def __init__(self, config):
        super(BunnyMetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):  #continuous_training通常都为False
            self.vision_tower = build_vision_tower(config, delay_load=not getattr(config, 'continuous_training', False))  #如果 continuous_training 是 False（通常是默认状态）： delay_load 变成 True。
            #这个地方可以添加的，因为不传递的话，就是indentifymap 
            #如果 continuous_training 是 True： delay_load 变成 False。发生什么： 视觉塔会立即去 /mnt/facebook/... 读原始权重。意图： 这通常用于你从零开始训练（比如刚从 LLM 接入 Vision Tower 的第一阶段），此时没有现成的 Checkpoint 可读，必须从官方权重开始。
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
        ###########################################
        #这段代码处理的是 LLM 模型在**推理（Inference）阶段的“流式生成”（Streaming Generation）**情况。
        ############################################################################
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        #images 是一个包含多张图像的 List (每个元素是 $B \times C \times H \times W$)
        #5D 张量 (视频格式 $B \times T \times C \times H \times W$)，则判断为多帧输入 
        #第一种情况：多图或视频输入，把所有图片堆叠在一起。
        # 1. 统一编码，拿到 [Total_Images, 358, 1024]
        if isinstance(images, list) or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            raw_features = self.encode_images(concat_images)
        else:
            raw_features = self.encode_images(images)
        # 2. 【防坑核心】强制转换为包含 [358, 1024] 元素的列表
        # 无论你是训练还是推理，无论是一张还是多张
        if raw_features.ndim == 3: # [Batch, 358, 1024]
            image_features = [raw_features[i] for i in range(raw_features.shape[0])]
        else:
            image_features = raw_features # 如果已经是 list 则保持
        # 调试监控眼
        if local_rank== 0: # 只在主进程打印
            print(f"DEBUG: image_features type: {type(image_features)}")
            if isinstance(image_features, list):
                print(f"DEBUG: image_features[0] shape: {image_features[0].shape}")
            else:
                print(f"DEBUG: image_features shape: {image_features.shape}")

        ###########################################
        #因为函数最后要返回这些值。如果用户进来时没传 attention_mask，函数会自己生成一个，
        # 最后返回时要根据这个备份判断是返回生成的掩码还是返回 None
        ############################################################################       
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        #改 Tensor 里的 -200 为 0：是为了让模型在把文本转向量（Embedding）时，别因为遇到负数而崩溃。
        input_ids_temp = input_ids # points to the actual input_ids tensor


        # # 形状 [2, 6] -> Batch_size=2, Max_Len=6
        # [
        # [-200, 10, 15, 12, 0, 0],  # 样本1: <image> What is this? (实际长度4, 补了2个0)
        # [10, 25, 0, 0, 0, 0]       # 样本2: Hello? (实际长度2, 补了4个0)
        # ]
        # attention_mask (Tensor):

        # Python
        # [
        # [1, 1, 1, 1, 0, 0], # 1代表真话，0代表Padding
        # [1, 1, 0, 0, 0, 0]
        # ]

        # 去掉 Padding（拆掉多余的线）
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
  
        input_ids_temp[input_ids_temp == IMAGE_TOKEN_INDEX] = 0

        # 执行后，原始 input_ids 变成：

        # Python
        # [
        #   [0, 10, 15, 12, 0, 0],  # 原来的 -200 变成了 0
        #   [10, 25, 0, 0, 0, 0]
        # ]

        token_sewer = VisualTokenStructurer(self.config, self.get_input_embeddings())
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # --- 2. 安全检查：统计占位符总数 (修正后的逻辑) ---
        total_image_placeholders = sum((x == IMAGE_TOKEN_INDEX).sum().item() for x in input_ids)
        if len(image_features) != total_image_placeholders:
            raise ValueError(f"特征数量({len(image_features)})与占位符数量({total_image_placeholders})不匹配！")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            cur_labels = labels[batch_idx]
            
            if num_images == 0:
                # 纯文本处理
                cur_input_embeds = self.get_input_embeddings()(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(cur_labels)
                continue

            # 寻找切口 [-1, img_pos, end_pos]
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                # A. 切出文本片段并转向量
                text_seg = cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i+1]]
                label_seg = cur_labels[image_token_indices[i] + 1 : image_token_indices[i+1]]
                
                if text_seg.shape[0] > 0:
                    cur_new_input_embeds.append(self.get_input_embeddings()(text_seg))
                    cur_new_labels.append(label_seg)
                
                # B. 插入结构化后的 365 图像特征
                if i < num_images:
                    raw_features = image_features[cur_image_idx] # [358, 2048]
                    cur_image_idx += 1
                    
                    # --- 讲究的 358 -> 365 转换 ---
                    # 4(Soul) + 1(n) + 144(Base) + 1(n) + 210(Detail) + 5(n) = 365
                    structured_features = token_sewer(raw_features) 
                    
                    cur_new_input_embeds.append(structured_features)
                    cur_new_labels.append(
                        torch.full((365,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)
                    )

            # 拼接单个样本
            new_input_embeds.append(torch.cat(cur_new_input_embeds))
            new_labels.append(torch.cat(cur_new_labels))

        # 3. 🛡️ 截断防御：解决 3052 > 2048 报错的核心逻辑
        # 必须在 Padding 之前截断，否则 Max Length 检查会报错
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', 2048)
        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # 4. 重新 Padding 成标准 Batch Tensor
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        target_dtype = new_labels[0].dtype
        target_device = new_labels[0].device
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=target_dtype, device=target_device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=target_device)
        position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=target_device)
        new_input_embeds_padded = []

        for i, (cur_embed, cur_label) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_embed.shape[0]
            # 统一右填充 (适合训练)
            new_input_embeds_padded.append(torch.cat((
                cur_embed,
                torch.zeros((max_len - cur_len, cur_embed.shape[1]), dtype=cur_embed.dtype, device=cur_embed.device)
            ), dim=0))
            
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_label
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=torch.long, device=target_device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # 5. 🔥 终极稳定性加固 (防止 NaN/Inf 毁掉模型)
        if torch.isnan(new_input_embeds).any() or torch.isinf(new_input_embeds).any():
            new_input_embeds = torch.nan_to_num(new_input_embeds, nan=0.0, posinf=65500, neginf=-65500)
        

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels_padded

