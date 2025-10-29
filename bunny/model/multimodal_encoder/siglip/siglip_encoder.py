import torch
import torch.nn as nn

from transformers import SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig
from bunny.util.s2wrapper import forward as multiscale_forward
from bunny.util.merge import bipartite_soft_matching_merge

# SiglipVisionTower 模块的功能和之前我们讨论的 CLIPVisionTower 基本类似，都是封装了一个预训练的视觉Transformer模型来做视觉特征提取。核心功能是：

# 从路径或名称加载预训练的SigLip视觉模型和对应的图片预处理器。

# 冻结模型参数，不参与训练，只进行推理。

# 通过指定的层(select_layer)和特征类型（patch或包含CLS token的cls_patch）选取需要的隐藏特征。

# 支持输入单张图像或图像列表，输出对应的视觉特征。

# 附带一些属性方便获得隐藏向量维度、设备、数据类型等信息。

# 和 CLIPVisionTower 不同的地方可能是模型具体是SigLip架构，而不是CLIP架构，这意味着视觉编码器是基于SigLip设计的。这两个视觉模型的结构细节和训练目标或预训练任务有所区别，但在多模态架构中作为视觉编码器的用途和接口设计基本类似。

# 所以总结：

# SiglipVisionTower 是用来获得SigLip视觉模型的表示，作为多模态学习中的视觉特征提取部分。

# 代码实现和CLIP视觉编码器封装类似，封装预训练模型，冻结权重，提供接口提取指定层的视觉token特征。

# 具体区别在于底层视觉编码器模型不同，SigLip是一个新兴的视觉Transformer模型。


class SiglipVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = -2

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        if self.is_loaded:
            return
        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        self.image_processor.crop_size = self.image_processor.size
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]

        return image_features


    def forward(self, images):
        
        if type(images) is list:
            all_hidden_states = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                      output_hidden_states=True)
                hidden_states = image_forward_out.hidden_states
                all_hidden_states.append(hidden_states)
            raise NotImplementedError("SiglipVisionTower should ideally process a single batch for simplicity, or have proper list handling.")

        else:
            image_forward_outs = self.vision_tower(
                    images.to(device=self.device, dtype=self.dtype),
                    output_hidden_states=True
            )
            # 🛠️ 返回完整的 hidden_states (一个 tuple/list of tensors)
            # image_forward_outs.hidden_states 是所有层的特征
            return image_forward_outs.hidden_states

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class SiglipVisionTowerS2(SiglipVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        self.s2_scales = getattr(args, 's2_scales', '384,768,1152')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        super().__init__(vision_tower, args, delay_load)

        self.multiscale_forward = multiscale_forward

        if not delay_load:
            self.image_processor.size['height'] = self.image_processor.size['width'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    # 1. 新增方法：获取所有层的特征
    def forward_all_features(self, images):
        """
        执行模型前向传播，返回所有 hidden states (一个 tuple/list of tensors)
        """
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                               output_hidden_states=True)
        return image_forward_outs.hidden_states


    def load_model(self):
        if self.is_loaded:
            return
        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        self.image_processor.crop_size = self.image_processor.size
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['height'] = self.image_processor.size['width'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                               output_hidden_states=True, interpolate_pos_encoding=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    def forward(self, images):
        if type(images) is list:
            all_processed_features = []
            for image in images:
                s2_feature = self.multiscale_forward(
                    self.forward_feature, 
                    image.unsqueeze(0), # 确保输入是 B=1
                    img_sizes=self.s2_scales, 
                    max_split_size=self.s2_split_size
                )
                all_processed_features.append(s2_feature)
            
            # 返回所有图像的 S2 特征列表 (如果需要返回所有层，列表处理需要更复杂的逻辑)
            return all_processed_features 

        else: 
            # 1. 获取所有中间层特征 (原始特征)
            # all_hidden_states 是一个 list 或 tuple of tensors
            all_hidden_states = list(self.forward_all_features(images))
            
            # 2. 提取 selected layer 的特征，进行 S2 处理
            # 传入的 model 是 forward_feature，它只返回 selected layer 的特征
            s2_feature = self.multiscale_forward(
                self.forward_feature, 
                images, 
                img_sizes=self.s2_scales,
                max_split_size=self.s2_split_size
                # 假设 Siglip 没有 prefix token
            )
            
            # 3. 计算 select_layer 在列表中的正确索引
            num_layers = len(all_hidden_states)
            # 例如：如果 select_layer=-2, num_layers=13，则 target_idx=11
            target_idx = num_layers + self.select_layer if self.select_layer < 0 else self.select_layer
            
            # 4. 创建要返回的第二个值：替换后的所有层特征
            # 我们对 all_hidden_states 进行复制并替换，以防外部代码意外修改原始列表
            combined_hidden_states = list(all_hidden_states) 
            combined_hidden_states[target_idx] = s2_feature # 替换！
            
            # 5. 返回两个值
            # 第一个值：S2 处理后的特征
            # 第二个值：包含 S2 特征和所有中间层特征的列表
            return s2_feature, combined_hidden_states

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
    

    @property
    def patch_size(self):
        return self.config.patch_size
